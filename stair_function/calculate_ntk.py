#!/usr/bin/env python3
import torch
import os
import glob
from torch.func import functional_call, vmap, jacrev
import numpy as np
from typing import Tuple, Dict, List
from functools import partial
import re
import torch.nn as nn
from mpi4py import MPI

print = partial(print, flush=True)

class DeepNN(nn.Module):
   def __init__(self, d: int, hidden_size: int, depth: int, mode: str = 'special'):
       super().__init__()
       
       torch.set_default_dtype(torch.float32)
       self.mode = mode
       self.depth = depth
       self.hidden_size = hidden_size
       self.input_dim = d
       
       layers = []
       prev_dim = d
       self.layer_lrs = []
       
       for layer_idx in range(depth):
           linear = nn.Linear(prev_dim, hidden_size)
           
           if mode == 'special':
               gain = nn.init.calculate_gain('relu')
               std = gain / np.sqrt(prev_dim)
               nn.init.normal_(linear.weight, mean=0.0, std=std)
               nn.init.zeros_(linear.bias)
               self.layer_lrs.append(1.0)
           elif mode == 'mup_pennington':
               if layer_idx == 0:  
                   std = 1.0 / np.sqrt(prev_dim)
                   lr_scale = 1.0  
               else:  
                   std = 1.0 / np.sqrt(prev_dim)
                   lr_scale = 1.0 / prev_dim  
               nn.init.normal_(linear.weight, mean=0.0, std=std)
               nn.init.zeros_(linear.bias)
               self.layer_lrs.append(lr_scale)
           else:  # standard
               nn.init.xavier_uniform_(linear.weight)
               nn.init.zeros_(linear.bias)
               self.layer_lrs.append(1.0)
           
           layers.extend([linear, nn.ReLU()])
           prev_dim = hidden_size
       
       final_layer = nn.Linear(prev_dim, 1)
       if mode == 'special':
           nn.init.normal_(final_layer.weight, std=0.01)
           self.layer_lrs.append(1.0)
       elif mode == 'mup_pennington':
           std = 1.0 / np.sqrt(prev_dim)
           lr_scale = 1.0 / prev_dim
           nn.init.normal_(final_layer.weight, std=std)
           self.layer_lrs.append(lr_scale)
       else:
           nn.init.xavier_uniform_(final_layer.weight)
           self.layer_lrs.append(1.0)
           
       nn.init.zeros_(final_layer.bias)
       layers.append(final_layer)
       
       self.network = nn.Sequential(*layers)

   def forward(self, x: torch.Tensor) -> torch.Tensor:
       return self.network(x).squeeze()
       
   def compute_feature_map(self, x: torch.Tensor) -> torch.Tensor:
       features = x
       for layer in list(self.network.children())[:-1]:
           features = layer(features)
       return features

def compute_empirical_kernels(model: nn.Module, X1: torch.Tensor, X2: torch.Tensor, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
   """Compute empirical NNGP and NTK kernels using the Jacobian contraction method"""
   model.eval()
   device = X1.device
   
   def fnet_single(params, x):
       return functional_call(model, params, (x.unsqueeze(0),)).squeeze(0)
   
   def compute_kernel_batch(x1_batch: torch.Tensor, x2_batch: torch.Tensor):
       params = {k: v for k, v in model.named_parameters()}
       jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1_batch)
       jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2_batch)
       
       jac1_flat = []
       jac2_flat = []
       for j1, j2 in zip(jac1.values(), jac2.values()):
           j1_shape = j1.shape
           j2_shape = j2.shape
           j1_reshaped = j1.reshape(j1_shape[0], -1)
           j2_reshaped = j2.reshape(j2_shape[0], -1)
           jac1_flat.append(j1_reshaped)
           jac2_flat.append(j2_reshaped)
       
       ntk_result = sum(torch.matmul(j1, j2.t()) for j1, j2 in zip(jac1_flat, jac2_flat))
       
       with torch.no_grad():
           feat1 = model.compute_feature_map(x1_batch)
           feat2 = model.compute_feature_map(x2_batch)
           nngp_result = torch.matmul(feat1, feat2.T) / feat1.shape[1]
       
       return nngp_result.detach(), ntk_result.detach()
   
   n1, n2 = X1.shape[0], X2.shape[0]
   nngp = torch.zeros((n1, n2), device=device)
   ntk = torch.zeros((n1, n2), device=device)
   
   for i in range(0, n1, batch_size):
       i_end = min(i + batch_size, n1)
       for j in range(0, n2, batch_size):
           j_end = min(j + batch_size, n2)
           
           nngp_batch, ntk_batch = compute_kernel_batch(
               X1[i:i_end], 
               X2[j:j_end]
           )
           nngp[i:i_end, j:j_end] = nngp_batch
           ntk[i:i_end, j:j_end] = ntk_batch
   
   return nngp.cpu().numpy(), ntk.cpu().numpy()

def save_kernels(nngp: np.ndarray, ntk: np.ndarray, base_path: str):
   """Save kernels using numpy format"""
   os.makedirs(os.path.dirname(base_path), exist_ok=True)
   np.save(f"{base_path}_nngp.npy", nngp)
   np.save(f"{base_path}_ntk.npy", ntk)

def parse_model_params(filename: str) -> dict:
   """Extract model parameters from filename"""
   pattern = r'h(\d+)_d(\d+)_n(\d+)_lr([\d\.]+)_(\w+)'
   match = re.search(pattern, filename)
   if match:
       params = {
           'hidden_size': int(match.group(1)),
           'depth': int(match.group(2)),
           'n_train': int(match.group(3)),
           'lr': float(match.group(4)),
           'mode': match.group(5),
           'd': 35  # Fixed input dimension
       }
       return params
   raise ValueError(f"Couldn't parse parameters from filename: {filename}")

def find_matching_files(directory: str) -> List[Dict]:
   """Find matching initial/final models and their training datasets"""
   matches = []
   
   initial_models = glob.glob(os.path.join(directory, 'initial_model_*.pt'))
   
   for initial_model in initial_models:
       base_name = os.path.basename(initial_model)
       config = base_name.replace('initial_model_', '').replace('.pt', '')
       
       final_model = os.path.join(directory, f'final_model_{config}.pt')
       train_dataset = os.path.join(directory, f'train_dataset_{config}.pt')
       
       if os.path.exists(final_model) and os.path.exists(train_dataset):
           matches.append({
               'initial_model': initial_model,
               'final_model': final_model,
               'train_dataset': train_dataset,
               'config': config
           })
   
   return matches

def process_model_pair(initial_model_path: str, final_model_path: str, dataset_path: str, 
                     save_dir: str, rank: int):
   """Process a pair of initial and final models"""
   device = torch.device('cpu')
   
   try:
       # Load dataset
       train_data = torch.load(dataset_path, map_location=device)
       if isinstance(train_data, dict):
           X = train_data['X']
       else:
           X = train_data
       
       # Extract parameters and create model
       params = parse_model_params(os.path.basename(initial_model_path))
       model = DeepNN(
           d=params['d'],
           hidden_size=params['hidden_size'],
           depth=params['depth'],
           mode=params['mode']
       ).to(device)
       
       # Process initial model
       print(f"Rank {rank}: Processing initial model: {initial_model_path}")
       model.load_state_dict(torch.load(initial_model_path, map_location=device))
       initial_nngp, initial_ntk = compute_empirical_kernels(model, X, X)
       
       base_name = os.path.basename(initial_model_path).replace('initial_model_', '').replace('.pt', '')
       initial_kernel_path = os.path.join(save_dir, f'{base_name}_initial_kernels')
       save_kernels(initial_nngp, initial_ntk, initial_kernel_path)
       
       # Process final model
       print(f"Rank {rank}: Processing final model: {final_model_path}")
       model.load_state_dict(torch.load(final_model_path, map_location=device))
       final_nngp, final_ntk = compute_empirical_kernels(model, X, X)
       
       final_kernel_path = os.path.join(save_dir, f'{base_name}_final_kernels')
       save_kernels(final_nngp, final_ntk, final_kernel_path)
       
       return True, base_name
       
   except Exception as e:
       return False, f"Error processing {os.path.basename(initial_model_path)}: {str(e)}"

def main():
   # Initialize MPI
   comm = MPI.COMM_WORLD
   rank = comm.Get_rank()
   size = comm.Get_size()
   
   # Configuration
   models_dir = "/mnt/users/goringn/NNs_vs_Kernels/stair_function/results/msp_standard_0401_shuffled_mup_false_antro_newmsp"  # Replace with your directory
   save_dir = os.path.join(models_dir, "kernel_analysis")  # Save kernels in a subdirectory
   
   # Only rank 0 creates the save directory and finds matching files
   if rank == 0:
       os.makedirs(save_dir, exist_ok=True)
       print("Finding matching model files...")
       matches = find_matching_files(models_dir)
       print(f"Found {len(matches)} matching model sets")
   else:
       matches = None
   
   # Broadcast matches to all workers
   matches = comm.bcast(matches, root=0)
   
   # Distribute work among workers
   worker_matches = []
   for i in range(len(matches)):
       if i % size == rank:
           worker_matches.append(matches[i])
   
   print(f"Rank {rank} processing {len(worker_matches)} model pairs")
   
   # Process assigned model pairs
   results = []
   for match in worker_matches:
       try:
           success, msg = process_model_pair(
               match['initial_model'],
               match['final_model'],
               match['train_dataset'],
               save_dir,
               rank
           )
           if not success:
               print(f"Rank {rank}: {msg}")
           results.append((success, msg))
       except Exception as e:
           print(f"Rank {rank} - Error processing {match['config']}: {str(e)}")
           results.append((False, str(e)))
   
   # Wait for all processes to complete
   comm.Barrier()
   
   # Gather results to rank 0
   all_results = comm.gather(results, root=0)
   
   # Print summary on rank 0
   if rank == 0:
       successful = 0
       failed = 0
       for worker_results in all_results:
           for success, msg in worker_results:
               if success:
                   successful += 1
               else:
                   failed += 1
                   print(msg)
       
       print(f"\nKernel computation complete!")
       print(f"Successfully processed: {successful}")
       print(f"Failed: {failed}")

if __name__ == "__main__":
   main()