#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
from mpi4py import MPI
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import List, Set, Tuple, Dict
import re
from functools import partial
import random
import os

# Define print globally first
print = partial(print, flush=True)

class MSPFunction:
    def __init__(self, P: int, sets: List[Set[int]]):
        self.P = P
        self.sets = sets
    
    def evaluate(self, z: torch.Tensor) -> torch.Tensor:
        device = z.device
        batch_size = z.shape[0]
        result = torch.zeros(batch_size, dtype=torch.float32, device=device)
        
        for S in self.sets:
            term = torch.ones(batch_size, dtype=torch.float32, device=device)
            for idx in S:
                term = term * z[:, idx]
            result = result + term
            
        return result

def process_single_model_pair(initial_state: Dict[str, torch.Tensor],
                            final_state: Dict[str, torch.Tensor],
                            n_train: int,
                            params: Dict,
                            msp: MSPFunction,
                            d: int,
                            base_seed: int) -> Dict:
    """Process a single model pair efficiently"""
    # Calculate parameter norm difference
    initial_vec = torch.cat([p.flatten() for p in initial_state.values()])
    final_vec = torch.cat([p.flatten() for p in final_state.values()])
    param_diff = torch.norm(final_vec - initial_vec).item()
    
    # Generate training data
    torch.manual_seed(base_seed)
    X_train = (2 * torch.bernoulli(0.5 * torch.ones((n_train, d), dtype=torch.float32)) - 1)
    
    # Compute activations more efficiently using batch processing
    def get_activations(state: Dict[str, torch.Tensor]) -> torch.Tensor:
        layers = []
        for i in range(params['depth']):
            linear = nn.Linear(d if i == 0 else params['hidden_size'], 
                             params['hidden_size'])
            linear.weight = nn.Parameter(state[f'network.{i*2}.weight'])
            linear.bias = nn.Parameter(state[f'network.{i*2}.bias'])
            layers.extend([linear, nn.ReLU()])
        network = nn.Sequential(*layers)
        with torch.no_grad():
            return network(X_train)
    
    # Compute kernel alignment
    initial_activations = get_activations(initial_state)
    final_activations = get_activations(final_state)
    
    # Compute kernel matrices and alignment efficiently
    K_initial = torch.matmul(initial_activations, initial_activations.t())
    K_final = torch.matmul(final_activations, final_activations.t())
    
    # Center kernels efficiently
    n = K_initial.shape[0]
    H = torch.eye(n) - torch.ones((n, n)) / n
    K_initial_centered = H @ K_initial @ H
    K_final_centered = H @ K_final @ H
    
    alignment = torch.sum(K_initial_centered * K_final_centered)
    norm1 = torch.sqrt(torch.sum(K_initial_centered * K_initial_centered))
    norm2 = torch.sqrt(torch.sum(K_final_centered * K_final_centered))
    kernel_align = (alignment / (norm1 * norm2)).item()
    
    return {
        'hidden_size': params['hidden_size'],
        'depth': params['depth'],
        'n_train': n_train,
        'learning_rate': params['lr'],
        'param_norm_diff': param_diff,
        'kernel_alignment': kernel_align
    }


def create_separate_plots(all_results: List[Dict]):
    """Create separate grid plots for parameter differences and kernel alignments"""
    # Extract unique parameters
    depths = sorted(set(r['depth'] for r in all_results))
    learning_rates = sorted(set(r['learning_rate'] for r in all_results))
    hidden_sizes = sorted(set(r['hidden_size'] for r in all_results))
    
    # Create color maps
    n_sizes = len(hidden_sizes)
    colors = plt.cm.viridis(np.linspace(0, 1, n_sizes))
    
    metrics = [
        ('param_norm_diff', 'Parameter Vector Difference', True),  # True for log scale
        ('kernel_alignment', 'Kernel Alignment', False)  # False for linear scale
    ]
    
    figures = []
    
    # Create separate plots for each metric
    for metric, title, use_log in metrics:
        # Create figure
        n_rows = len(depths)
        n_cols = len(learning_rates)
        fig = plt.figure(figsize=(4*n_cols, 3*n_rows))
        fig.suptitle(f'{title} vs Training Size', y=1.02, fontsize=16)
        
        for i, depth in enumerate(depths):
            for j, lr in enumerate(learning_rates):
                ax = plt.subplot(n_rows, n_cols, i*n_cols + j + 1)
                
                # Plot for each hidden size
                for idx, hidden_size in enumerate(hidden_sizes):
                    relevant_results = [
                        r for r in all_results 
                        if r['depth'] == depth and 
                           r['learning_rate'] == lr and 
                           r['hidden_size'] == hidden_size
                    ]
                    
                    if relevant_results:
                        x = [r['n_train'] for r in relevant_results]
                        y = [r[metric] for r in relevant_results]
                        points = sorted(zip(x, y))
                        if points:
                            x, y = zip(*points)
                            ax.plot(x, y, '-o', color=colors[idx], 
                                  label=f'h={hidden_size}', 
                                  markersize=4)
                
                ax.set_xscale('log')
                if use_log:
                    ax.set_yscale('log')
                
                # Set labels
                if i == n_rows-1:
                    ax.set_xlabel('Training Size')
                if j == 0:
                    ax.set_ylabel(title)
                
                ax.text(0.05, 0.95, f'd={depth}\nlr={lr:.1e}', 
                       transform=ax.transAxes, 
                       verticalalignment='top',
                       bbox=dict(facecolor='white', alpha=0.8))
                
                # Only add legend to first subplot
                if i == 0 and j == 0:
                    ax.legend(fontsize='small', bbox_to_anchor=(1.05, 1), 
                            loc='upper left', borderaxespad=0.)
                
                ax.grid(True, which="both", ls="-", alpha=0.2)
        
        plt.tight_layout()
        figures.append((fig, f"{metric}_analysis_grid.png"))
    
    return figures

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Constants
    P = 8
    d = 30
    msp_sets = [{7},{2,7},{0,2,7},{5,7,4},{1},{0,4},{3,7},{0,1,2,3,4,6,7}]
    n_train_sizes = [10,50,100,200,300,400,500,800,1000,2500,5000,8000,10000,15000,20000]
    base_seed = 42
    
    # Initialize MSP function
    msp = MSPFunction(P, msp_sets)
    
    # Set paths
    results_dir = Path("/mnt/users/goringn/NNs_vs_Kernels/stair_function/results/msp_NN_grid_0912_cpuparallel")
    
    if rank == 0:
        # Create work items (list of tasks to be distributed)
        work_items = []
        model_pattern = r'(?:initial|final)_model_h(\d+)_d(\d+)_n(\d+)_lr([\d.e-]+)_(\w+)_(\d+)_rank(\d+).pt'
        
        # Group files by parameters
        model_groups = {}
        for model_file in results_dir.glob("*model_*.pt"):
            match = re.match(model_pattern, model_file.name)
            if match:
                h, d, n, lr, mode, timestamp, r = match.groups()
                key = (h, d, n, lr, mode, timestamp, r)
                if key not in model_groups:
                    model_groups[key] = {'initial': None, 'final': None}
                if 'initial' in model_file.name:
                    model_groups[key]['initial'] = model_file
                else:
                    model_groups[key]['final'] = model_file
        
        # Create work items from complete pairs
        for key, files in model_groups.items():
            if files['initial'] and files['final']:
                h, d, n, lr, _, _, _ = key
                params = {
                    'hidden_size': int(h),
                    'depth': int(d),
                    'n_train': int(n),
                    'lr': float(lr)
                }
                if int(n) in n_train_sizes:  # Only include specified training sizes
                    work_items.append((files['initial'], files['final'], params))
        
        print(f"Found {len(work_items)} model pairs to process")
        
        # Split work among processes
        chunks = np.array_split(work_items, size)
    else:
        chunks = None
    
    # Scatter work to all processes
    chunk = comm.scatter(chunks if rank == 0 else None, root=0)
    
    # Process assigned chunk
    results = []
    for initial_file, final_file, params in chunk:
        try:
            # Load models
            initial_state = torch.load(str(initial_file), map_location='cpu')
            final_state = torch.load(str(final_file), map_location='cpu')
            
            # Process the model pair
            result = process_single_model_pair(
                initial_state, final_state,
                params['n_train'], params, msp, d, base_seed
            )
            results.append(result)
            
            print(f"Rank {rank}: Processed model pair with n_train={params['n_train']}")
            
        except Exception as e:
            print(f"Rank {rank}: Error processing models: {e}")
            continue
    
    # Gather results
    all_results = comm.gather(results, root=0)
    
    if rank == 0:
        # Combine results
        combined_results = []
        for worker_results in all_results:
            combined_results.extend(worker_results)
        
        # Save results
        with open('model_analysis_results.json', 'w') as f:
            json.dump(combined_results, f, indent=4)
        
        # Create plots (plotting code remains the same)
        create_separate_plots(combined_results)
        print("Analysis complete. Results saved.")

if __name__ == "__main__":
    main()