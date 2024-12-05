#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
import random
from mpi4py import MPI


from functools import partial

# Define print globally first
print = partial(print, flush=True)

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def generate_random_msp(P: int, random_seed: int = 41) -> List[set]:
    """Generate MSP with fixed seed for reproducibility"""
    random.seed(random_seed)
    
    sets = []
    num_sets = random.randint(1, P)
    
    size = random.randint(1, min(3, P))
    sets.append(set(random.sample(range(P), size)))
    
    for _ in range(num_sets - 1):
        prev_union = set().union(*sets)
        remaining = set(range(P)) - prev_union
        
        if remaining and random.random() < 0.7:
            new_elem = random.choice(list(remaining))
            base_elems = random.sample(list(prev_union), random.randint(0, len(prev_union)))
            new_set = set(base_elems + [new_elem])
        else:
            size = random.randint(1, len(prev_union))
            new_set = set(random.sample(list(prev_union), size))
        
        sets.append(new_set)
    
    return sets

class MSPFunction:
    def __init__(self, P: int, sets: List[set]):
        self.P = P
        self.sets = sets
        
    def evaluate(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[0]
        result = torch.zeros(batch_size, device=z.device)
        
        for S in self.sets:
            term = torch.ones(batch_size, device=z.device)
            for idx in S:
                term = term * z[:, idx]
            result = result + term
            
        return result

class DeepNN(nn.Module):
    def __init__(self, d: int, hidden_dims: List[int]):
        super().__init__()
        layers = []
        prev_dim = d
        for hidden_dim in hidden_dims:
            linear = nn.Linear(prev_dim, hidden_dim)
            gain = nn.init.calculate_gain('relu')
            std = gain / np.sqrt(prev_dim)
            nn.init.normal_(linear.weight, mean=0.0, std=std)
            nn.init.zeros_(linear.bias)
            layers.extend([linear, nn.ReLU()])
            prev_dim = hidden_dim
        
        final_layer = nn.Linear(prev_dim, 1)
        nn.init.normal_(final_layer.weight, std=0.01)
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze()

def collect_training_outputs(model: nn.Module,
                           msp: MSPFunction,
                           X_train: torch.Tensor,
                           y_train: torch.Tensor,
                           X_test: torch.Tensor,
                           y_test: torch.Tensor,
                           batch_size: int,
                           epochs: int,
                           lr: float,
                           seed: int,
                           weight_decay: float = 0.0) -> Tuple[torch.Tensor, List[float], List[float]]:
    """Train network and collect outputs for a single seed"""
    torch.manual_seed(seed)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    test_outputs = torch.zeros(epochs, len(X_test))
    train_errors = []
    test_errors = []
    
    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            output = model(batch_X)
            loss = torch.mean((output - batch_y) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        
        with torch.no_grad():
            train_pred = model(X_train)
            test_pred = model(X_test)
            train_error = torch.mean((train_pred - y_train) ** 2).item()
            test_error = torch.mean((test_pred - y_test) ** 2).item()
            train_errors.append(train_error)
            test_errors.append(test_error)
            test_outputs[epoch] = test_pred
            
            if epoch % 100 == 0 and rank == 0:
                print(f"Seed {seed}, Epoch {epoch}: Train Error = {train_error:.6f}, Test Error = {test_error:.6f}")
    
    return test_outputs, train_errors, test_errors

def analyze_ergodicity(outputs_matrix: torch.Tensor, 
                      epoch_windows: List[int]) -> Dict:
    """Analyze ergodicity of network outputs"""
    n_seeds, total_epochs, n_test = outputs_matrix.shape
    results = {
        'window_sizes': epoch_windows,
        'empirical_variances': []
    }
    
    for window_size in epoch_windows:
        n_windows = total_epochs // window_size
        if n_windows == 0:
            continue
            
        # Reshape to organize data into windows
        reshaped = outputs_matrix[:, :(n_windows*window_size), :].reshape(
            n_seeds, n_windows, window_size, n_test
        )
        
        # Use permute instead of transpose for multi-dimensional tensors
        reshaped = reshaped.permute(1, 0, 2, 3)
        
        # Calculate means for each window
        window_means = reshaped.mean(dim=(1, 2))  # Average over seeds and time
        
        # Calculate empirical variance across windows
        empirical_var = window_means.var(dim=0).mean()  # Average over test points
        
        results['empirical_variances'].append(empirical_var.item())
    
    return results

def main():
    # Parameters
    P = 9
    d = 54
    hidden_dims = [1000, 500, 250, 100]
    n_test = 10
    batch_size = 64
    epochs = 5000
    lr = 0.001
    weight_decay = 1e-4
    n_seeds = 50
    n_train_sizes = [100, 500, 1000]
    epoch_windows = [50, 100, 200, 500, 1000]
    
    # Create the ground truth MSP function with seed 41
    sets = generate_random_msp(P, random_seed=41)
    if rank == 0:
        print("Ground Truth MSP sets:", sets)
    msp = MSPFunction(P, sets)
    
    # Only rank 0 creates directories and manages file output
    if rank == 0:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs('results', exist_ok=True)
    else:
        timestamp = None
    
    timestamp = comm.bcast(timestamp, root=0)
    
    # Process different training sizes
    for n_train in n_train_sizes:
        if rank == 0:
            print(f"\nProcessing n_train = {n_train}")
        
        # Generate random training data (different for each n_train)
        torch.manual_seed(n_train)  # Different seed for each training size
        X_train = 2 * torch.bernoulli(0.5 * torch.ones((n_train, d))) - 1
        X_test = 2 * torch.bernoulli(0.5 * torch.ones((n_test, d))) - 1
        y_train = msp.evaluate(X_train)
        y_test = msp.evaluate(X_test)
        
        # Distribute seeds across ranks
        seeds_per_rank = n_seeds // size
        local_seeds = range(rank * seeds_per_rank, (rank + 1) * seeds_per_rank)
        
        # Local storage
        local_outputs = torch.zeros(seeds_per_rank, epochs, n_test)
        local_train_errors = []
        local_test_errors = []
        
        # Process local seeds
        for i, seed in enumerate(local_seeds):
            model = DeepNN(d, hidden_dims)
            outputs, train_errs, test_errs = collect_training_outputs(
                model, msp, X_train, y_train, X_test, y_test,
                batch_size, epochs, lr, seed, weight_decay
            )
            local_outputs[i] = outputs
            local_train_errors.append(train_errs)
            local_test_errors.append(test_errs)
        
        # Gather results to rank 0
        all_outputs = comm.gather(local_outputs, root=0)
        all_train_errors = comm.gather(local_train_errors, root=0)
        all_test_errors = comm.gather(local_test_errors, root=0)
        
        if rank == 0:
            # Combine gathered results
            combined_outputs = torch.cat(all_outputs)
            combined_train_errors = [err for sublist in all_train_errors for err in sublist]
            combined_test_errors = [err for sublist in all_test_errors for err in sublist]
            
            # Calculate averages
            avg_train_errors = np.mean(combined_train_errors, axis=0)
            avg_test_errors = np.mean(combined_test_errors, axis=0)
            
            # Analyze ergodicity
            ergodicity_results = analyze_ergodicity(combined_outputs, epoch_windows)
            
            # Save results
            results = {
                'n_train': n_train,
                'msp_sets': [list(s) for s in sets],  # Convert sets to lists for JSON
                'ergodicity_analysis': ergodicity_results,
                'final_train_error': float(avg_train_errors[-1]),
                'final_test_error': float(avg_test_errors[-1])
            }
            
            # Save to files
            with open(f'results/ergodicity_results_{n_train}_{timestamp}.json', 'w') as f:
                json.dump(results, f, indent=4)
            
            # Create plots
            plt.figure(figsize=(10, 6))
            plt.semilogy(avg_train_errors, label='Train Error')
            plt.semilogy(avg_test_errors, label='Test Error')
            plt.xlabel('Epochs')
            plt.ylabel('MSE')
            plt.title(f'Training Curves (n_train={n_train})')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'results/training_curves_n{n_train}_{timestamp}.png')
            plt.close()
            
            plt.figure(figsize=(10, 6))
            plt.loglog(ergodicity_results['window_sizes'], 
                      ergodicity_results['empirical_variances'], 
                      'x-', label=f'n_train={n_train}')
            plt.xlabel('Window Size (epochs)')
            plt.ylabel('Empirical Variance')
            plt.title('Ergodicity Analysis')
            plt.grid(True)
            plt.savefig(f'results/ergodicity_n{n_train}_{timestamp}.png')
            plt.close()
            
            print(f"\nFinal Results for n_train = {n_train}:")
            print(f"Average Final Train Error: {avg_train_errors[-1]:.6f}")
            print(f"Average Final Test Error: {avg_test_errors[-1]:.6f}")

if __name__ == "__main__":
    main()