import torch
import torch.nn as nn
from typing import List, Dict
import os
from mpi4py import MPI
import json
from datetime import datetime
import numpy as np
from functools import partial

print = partial(print, flush=True)


class DeepNN(nn.Module):
    def __init__(self, d: int, hidden_size: int, depth: int, mode: str = 'special', gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma
        # ... rest of initialization code stays the same ...

        
        torch.set_default_dtype(torch.float32)
        
        self.mode = mode
        self.depth = depth
        self.hidden_size = hidden_size
        self.input_dim = d
        
        layers = []
        prev_dim = d
        self.layer_lrs = []  # Store layerwise learning rates
        
        for layer_idx in range(depth):
            linear = nn.Linear(prev_dim, hidden_size)
            
            if mode == 'special':
                # Special initialization as in original code
                gain = nn.init.calculate_gain('relu')
                std = gain / np.sqrt(prev_dim)
                nn.init.normal_(linear.weight, mean=0.0, std=std)
                nn.init.zeros_(linear.bias)
                self.layer_lrs.append(1.0)
                
            elif mode == 'spectral':
                # Implement spectral initialization
                fan_in = prev_dim
                fan_out = hidden_size
                std = (1.0 / np.sqrt(fan_in)) * min(1.0, np.sqrt(fan_out / fan_in))
                nn.init.normal_(linear.weight, mean=0.0, std=std)
                nn.init.zeros_(linear.bias)
                self.layer_lrs.append(float(fan_out) / fan_in)
                
            elif mode == 'mup_pennington':
                # muP initialization and learning rates from the paper
                if layer_idx == 0:  # Embedding layer
                    std = 1.0 / np.sqrt(prev_dim)
                    lr_scale = 1.0  # O(1) learning rate for embedding
                else:  # Hidden layers
                    std = 1.0 / np.sqrt(prev_dim)
                    lr_scale = 1.0 / prev_dim  # O(1/n) learning rate for hidden
                nn.init.normal_(linear.weight, mean=0.0, std=std)
                nn.init.zeros_(linear.bias)
                self.layer_lrs.append(lr_scale)
                
            else:  # standard
                nn.init.xavier_uniform_(linear.weight)
                nn.init.zeros_(linear.bias)
                self.layer_lrs.append(1.0)
            
            layers.extend([
                linear,
                nn.ReLU()
            ])
            prev_dim = hidden_size
        
        # Final layer
        final_layer = nn.Linear(prev_dim, 1)
        if mode == 'special':
            nn.init.normal_(final_layer.weight, std=0.01)
            self.layer_lrs.append(1.0)
        elif mode == 'spectral':
            fan_in = prev_dim
            fan_out = 1
            std = (1.0 / np.sqrt(fan_in)) * min(1.0, np.sqrt(fan_out / fan_in))
            nn.init.normal_(final_layer.weight, std=std)
            self.layer_lrs.append(float(fan_out) / fan_in)
        elif mode == 'mup_pennington':
            std = 1.0 / np.sqrt(prev_dim)
            lr_scale = 1.0 / prev_dim  # O(1/n) learning rate for readout
            nn.init.normal_(final_layer.weight, std=std)
            self.layer_lrs.append(lr_scale)
        else:
            nn.init.xavier_uniform_(final_layer.weight)
            self.layer_lrs.append(1.0)
            
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma = float(self.gamma)  # Ensure gamma is a float
        return self.network(x).squeeze() / gamma
    
    def get_layer_learning_rates(self, base_lr: float) -> List[float]:
        """Return list of learning rates for each layer"""
        return [base_lr * lr for lr in self.layer_lrs]

def load_model_and_data(model_path: str, data_path: str, hidden_size: int, depth: int, device: torch.device):
    """Load model and corresponding dataset"""
    # Load model state
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    
    # Initialize model
    model = DeepNN(d=20, hidden_size=hidden_size, depth=depth, mode='mup_pennington').to(device)
    model.load_state_dict(state_dict)
    
    # Load dataset
    data_dict = torch.load(data_path, map_location=device)
    X = data_dict['X'].to(device)
    y = data_dict['y'].to(device)
    
    return model, X, y

def perturb_model(model: nn.Module, sigma: float) -> nn.Module:
    """Add Gaussian noise to model parameters"""
    perturbed_model = type(model)(
        d=model.input_dim, 
        hidden_size=model.hidden_size, 
        depth=model.depth,
        mode='mup_pennington'
    ).to(next(model.parameters()).device)
    
    perturbed_model.load_state_dict(model.state_dict())
    
    with torch.no_grad():
        for param in perturbed_model.parameters():
            noise = torch.randn_like(param) * sigma
            param.add_(noise)
    
    return perturbed_model

def evaluate_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    """Compute MSE loss for model"""
    with torch.no_grad():
        pred = model(X)
        loss = torch.mean((pred - y) ** 2).item()
    return loss

def process_model_pair(
    model_path: str,
    data_path: str,
    hidden_size: int,
    depth: int,
    sigmas: List[float],
    n_perturbations: int,
    device: torch.device
) -> Dict:
    """Process a single model-data pair"""
    # Extract n_train from path using regex
    import re
    n_match = re.search(r'_n(\d+)_', data_path)
    if n_match:
        n_train = int(n_match.group(1))
    else:
        raise ValueError(f"Could not extract n_train from path: {data_path}")
        
    print(f"\nStarting model with n_train={n_train}")
    
    # Load model and data
    model, X, y = load_model_and_data(model_path, data_path, hidden_size, depth, device)
    
    # Get base loss
    base_loss = evaluate_model(model, X, y)
    print(f"Base model loss: {base_loss:.6f}")
    
    results = {
        'hidden_size': hidden_size,
        'depth': depth,
        'base_loss': base_loss,
        'n_train': n_train,
        'perturbations': {}
    }
    
    # For each sigma value
    total_perturbations = len(sigmas) * n_perturbations
    perturbation_count = 0
    
    for sigma in sigmas:
        print(f"\nProcessing sigma={sigma}")
        losses = []
        
        for i in range(n_perturbations):
            perturbed_model = perturb_model(model, sigma)
            loss = evaluate_model(perturbed_model, X, y)
            losses.append(loss)
            
            perturbation_count += 1
            if i % 100 == 0:  # Print every 100 perturbations
                print(f"n_train={n_train}, sigma={sigma}: {i}/{n_perturbations} perturbations complete")
                print(f"Total progress: {perturbation_count}/{total_perturbations} [{perturbation_count/total_perturbations*100:.1f}%]")
            
            # Clear GPU memory
            del perturbed_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        # Print summary statistics for this sigma
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        print(f"Completed sigma={sigma}: Mean loss={mean_loss:.6f}, Std loss={std_loss:.6f}")
            
        results['perturbations'][str(sigma)] = losses
    
    print(f"\nCompleted all perturbations for n_train={n_train}")
    return results

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Better GPU assignment
    n_gpus = torch.cuda.device_count()
    if rank == 0:
        print(f"Number of GPUs available: {n_gpus}")
    
    if n_gpus > 0:
        # Divide processes evenly between GPUs
        processes_per_gpu = size // n_gpus
        gpu_id = rank // processes_per_gpu if rank < (processes_per_gpu * n_gpus) else (n_gpus - 1)
        device = torch.device(f'cuda:{gpu_id}')
        
        # Print GPU assignment
        print(f"Rank {rank} assigned to GPU {gpu_id}")
    else:
        device = torch.device('cpu')
        print(f"Rank {rank} using CPU")
    
    # Set this GPU as the default
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    
    # Parameters
    hidden_size = 2000
    depth = 4
    sigmas = [0.000001,0.000005,0.00001,0.00005, 0.0001,0.0005,0.001,0.005,0.01,0.1,0.2]
    n_perturbations = 1000
    experiment_name = "stability_test_pp_false_h2000_d4_g1_lr0001_all1_locnorm"
    
    # Model paths
#     model_paths = [
#        "/mnt/users/goringn/NNs_vs_Kernels/low_dim_poly/results/low_dim_poly_prune_2801_false/initial_model_h1000_d4_n10_lr0.001_mup_pennington_20250128_001331_rank0.pt",
# "/mnt/users/goringn/NNs_vs_Kernels/low_dim_poly/results/low_dim_poly_prune_2801_false/initial_model_h1000_d4_n100_lr0.001_mup_pennington_20250128_001331_rank1.pt",
# "/mnt/users/goringn/NNs_vs_Kernels/low_dim_poly/results/low_dim_poly_prune_2801_false/initial_model_h1000_d4_n500_lr0.001_mup_pennington_20250128_001331_rank0.pt",
# "/mnt/users/goringn/NNs_vs_Kernels/low_dim_poly/results/low_dim_poly_prune_2801_false/initial_model_h1000_d4_n1000_lr0.001_mup_pennington_20250128_001331_rank1.pt",
# "/mnt/users/goringn/NNs_vs_Kernels/low_dim_poly/results/low_dim_poly_prune_2801_false/initial_model_h1000_d4_n2500_lr0.001_mup_pennington_20250128_001331_rank0.pt",
# "/mnt/users/goringn/NNs_vs_Kernels/low_dim_poly/results/low_dim_poly_prune_2801_false/initial_model_h1000_d4_n5000_lr0.001_mup_pennington_20250128_001331_rank1.pt",
# "/mnt/users/goringn/NNs_vs_Kernels/low_dim_poly/results/low_dim_poly_prune_2801_false/initial_model_h1000_d4_n10000_lr0.001_mup_pennington_20250128_001331_rank0.pt",
# "/mnt/users/goringn/NNs_vs_Kernels/low_dim_poly/results/low_dim_poly_prune_2801_false/initial_model_h1000_d4_n20000_lr0.001_mup_pennington_20250128_001331_rank1.pt",
# "/mnt/users/goringn/NNs_vs_Kernels/low_dim_poly/results/low_dim_poly_prune_2801_false/initial_model_h1000_d4_n40000_lr0.001_mup_pennington_20250128_001331_rank0.pt"
#     ]
    
#     # Data paths
#     data_paths = [
#        "/mnt/users/goringn/NNs_vs_Kernels/low_dim_poly/results/low_dim_poly_prune_2801_false/dataset_h1000_d4_n10_lr0.001_mup_pennington_20250128_001331_rank0.pt",
# "/mnt/users/goringn/NNs_vs_Kernels/low_dim_poly/results/low_dim_poly_prune_2801_false/dataset_h1000_d4_n100_lr0.001_mup_pennington_20250128_001331_rank1.pt",
# "/mnt/users/goringn/NNs_vs_Kernels/low_dim_poly/results/low_dim_poly_prune_2801_false/dataset_h1000_d4_n500_lr0.001_mup_pennington_20250128_001331_rank0.pt",
# "/mnt/users/goringn/NNs_vs_Kernels/low_dim_poly/results/low_dim_poly_prune_2801_false/dataset_h1000_d4_n1000_lr0.001_mup_pennington_20250128_001331_rank1.pt",
# "/mnt/users/goringn/NNs_vs_Kernels/low_dim_poly/results/low_dim_poly_prune_2801_false/dataset_h1000_d4_n2500_lr0.001_mup_pennington_20250128_001331_rank0.pt",
# "/mnt/users/goringn/NNs_vs_Kernels/low_dim_poly/results/low_dim_poly_prune_2801_false/dataset_h1000_d4_n5000_lr0.001_mup_pennington_20250128_001331_rank1.pt",
# "/mnt/users/goringn/NNs_vs_Kernels/low_dim_poly/results/low_dim_poly_prune_2801_false/dataset_h1000_d4_n10000_lr0.001_mup_pennington_20250128_001331_rank0.pt",
# "/mnt/users/goringn/NNs_vs_Kernels/low_dim_poly/results/low_dim_poly_prune_2801_false/dataset_h1000_d4_n20000_lr0.001_mup_pennington_20250128_001331_rank1.pt",
# "/mnt/users/goringn/NNs_vs_Kernels/low_dim_poly/results/low_dim_poly_prune_2801_false/dataset_h1000_d4_n40000_lr0.001_mup_pennington_20250128_001331_rank0.pt"
#     ]

  
    # Model paths
    model_paths = [
     "/mnt/users/goringn/NNs_vs_Kernels/low_dim_poly/results/low_dim_poly_prune_2801_false_lr_localnorm/final_model_h2000_d4_n40000_lr0.001_mup_pennington_20250129_053049_rank1.pt",
"/mnt/users/goringn/NNs_vs_Kernels/low_dim_poly/results/low_dim_poly_prune_2801_false_lr_localnorm/final_model_h2000_d4_n20000_lr0.001_mup_pennington_20250129_053049_rank5.pt",
"/mnt/users/goringn/NNs_vs_Kernels/low_dim_poly/results/low_dim_poly_prune_2801_false_lr_localnorm/final_model_h2000_d4_n10000_lr0.001_mup_pennington_20250129_053049_rank9.pt",
"/mnt/users/goringn/NNs_vs_Kernels/low_dim_poly/results/low_dim_poly_prune_2801_false_lr_localnorm/final_model_h2000_d4_n2500_lr0.001_mup_pennington_20250129_053049_rank5.pt",
"/mnt/users/goringn/NNs_vs_Kernels/low_dim_poly/results/low_dim_poly_prune_2801_false_lr_localnorm/final_model_h2000_d4_n1000_lr0.001_mup_pennington_20250129_053049_rank9.pt",
"/mnt/users/goringn/NNs_vs_Kernels/low_dim_poly/results/low_dim_poly_prune_2801_false_lr_localnorm/final_model_h2000_d4_n100_lr0.001_mup_pennington_20250129_053049_rank5.pt",
"/mnt/users/goringn/NNs_vs_Kernels/low_dim_poly/results/low_dim_poly_prune_2801_false_lr_localnorm/final_model_h2000_d4_n10_lr0.001_mup_pennington_20250129_053049_rank9.pt"
    ]
    
    # Data paths
    data_paths = [
"/mnt/users/goringn/NNs_vs_Kernels/low_dim_poly/results/low_dim_poly_prune_2801_false_lr_localnorm/dataset_h2000_d4_n40000_lr0.001_mup_pennington_20250129_053049_rank1.pt",
"/mnt/users/goringn/NNs_vs_Kernels/low_dim_poly/results/low_dim_poly_prune_2801_false_lr_localnorm/dataset_h2000_d4_n20000_lr0.001_mup_pennington_20250129_053049_rank5.pt",
"/mnt/users/goringn/NNs_vs_Kernels/low_dim_poly/results/low_dim_poly_prune_2801_false_lr_localnorm/dataset_h2000_d4_n10000_lr0.001_mup_pennington_20250129_053049_rank9.pt",
"/mnt/users/goringn/NNs_vs_Kernels/low_dim_poly/results/low_dim_poly_prune_2801_false_lr_localnorm/dataset_h2000_d4_n5000_lr0.001_mup_pennington_20250129_053049_rank1.pt",
"/mnt/users/goringn/NNs_vs_Kernels/low_dim_poly/results/low_dim_poly_prune_2801_false_lr_localnorm/dataset_h2000_d4_n2500_lr0.001_mup_pennington_20250129_053049_rank5.pt",
"/mnt/users/goringn/NNs_vs_Kernels/low_dim_poly/results/low_dim_poly_prune_2801_false_lr_localnorm/dataset_h2000_d4_n1000_lr0.001_mup_pennington_20250129_053049_rank9.pt",
"/mnt/users/goringn/NNs_vs_Kernels/low_dim_poly/results/low_dim_poly_prune_2801_false_lr_localnorm/dataset_h2000_d4_n100_lr0.001_mup_pennington_20250129_053049_rank5.pt",
"/mnt/users/goringn/NNs_vs_Kernels/low_dim_poly/results/low_dim_poly_prune_2801_false_lr_localnorm/dataset_h2000_d4_n10_lr0.001_mup_pennington_20250129_053049_rank9.pt"
    ]
    # Create results directory
    results_dir = f"results/{experiment_name}"
    if rank == 0:
        os.makedirs(results_dir, exist_ok=True)
        
    # Get timestamp
    if rank == 0:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    else:
        timestamp = None
    timestamp = comm.bcast(timestamp, root=0)
    
    # Distribute work
    worker_pairs = []
    for i in range(len(model_paths)):
        if i % size == rank:
            worker_pairs.append((model_paths[i], data_paths[i]))
    
    # Process assigned pairs
    results = []
    for model_path, data_path in worker_pairs:
        print(f"Rank {rank} processing: {model_path}")
        
        result = process_model_pair(
            model_path,
            data_path,
            hidden_size,
            depth,
            sigmas,
            n_perturbations,
            device
        )
        results.append(result)
        
        # Save intermediate results
        worker_results_path = os.path.join(
            results_dir, 
            f'stability_results_{timestamp}_rank{rank}.json'
        )
        with open(worker_results_path, 'w') as f:
            json.dump(results, f, indent=4)
    
    # Wait for all workers
    comm.Barrier()
    
    # Gather results
    all_results = comm.gather(results, root=0)
    
    # Combine and save final results
    if rank == 0:
        combined_results = []
        for worker_results in all_results:
            combined_results.extend(worker_results)
            
        final_results_path = os.path.join(
            results_dir,
            f'stability_results_final_{timestamp}.json'
        )
        with open(final_results_path, 'w') as f:
            json.dump(combined_results, f, indent=4)
        
        print("All results saved successfully")
if __name__ == "__main__":
    main()