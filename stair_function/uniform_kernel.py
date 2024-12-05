#!/usr/bin/env python3
import numpy as np
import torch
from typing import List, Set, Tuple
import random
from functools import partial
import json
from datetime import datetime
import os

# Define print globally first
print = partial(print, flush=True)

class MSPFunction:
    def __init__(self, P: int, sets: List[Set[int]]):
        self.P = P
        self.sets = sets
            
        # Verify MSP property
        for i in range(1, len(sets)):
            prev_union = set().union(*sets[:i])
            diff = sets[i] - prev_union
            if len(diff) > 1:
                raise ValueError(f"Not an MSP: Set {sets[i]} adds {len(diff)} new elements: {diff}")
    
    def evaluate(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[0]
        result = torch.zeros(batch_size, device=z.device)
        
        for S in self.sets:
            term = torch.ones(batch_size, device=z.device)
            for idx in S:
                term = term * z[:, idx]
            result = result + term
            
        return result

def generate_random_msp(P: int) -> List[Set[int]]:
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

def create_uniform_spectrum_kernel(X: torch.Tensor) -> torch.Tensor:
    """
    Creates a kernel matrix with uniform eigenspectrum on the given data.
    """
    n = X.shape[0]
    
    # Create orthogonal matrix using SVD of a random matrix
    random_matrix = torch.randn(n, n)
    U, _, V = torch.linalg.svd(random_matrix)
    Q = U @ V  # Orthogonal matrix
    
    # Create diagonal matrix with uniform eigenvalues
    uniform_eigenvalue = 1.0
    D = uniform_eigenvalue * torch.ones(n)
    
    # Construct kernel matrix K = QDQ^T
    K = Q @ torch.diag(D) @ Q.T
    
    return K

def compute_kernel_prediction(K_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor, X_train: torch.Tensor) -> torch.Tensor:
    """
    Compute kernel regression predictions for test points.
    """
    # Create kernel between test and training points
    n_test = X_test.shape[0]
    n_train = X_train.shape[0]
    
    # Create cross-kernel matrix using the same uniform spectrum approach
    random_matrix = torch.randn(n_test, n_train)
    U, _, V = torch.linalg.svd(random_matrix, full_matrices=False)
    K_test_train = U @ V  # This ensures consistent scaling with training kernel
    
    # Compute predictions
    pred = K_test_train @ torch.linalg.solve(K_train + 1e-8 * torch.eye(K_train.shape[0]), y_train)
    
    return pred

def main():
    # Parameters
    P = 9
    d = 54
    n_test = 1000
    n_train_sizes = [10, 50, 100, 200, 300, 400, 500, 800, 1000, 5000, 10000, 20000]
    
    # Create timestamp for saving results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(41)
    
    # Generate random MSP function
    sets = generate_random_msp(P)
    msp = MSPFunction(P, sets)
    
    # Generate test data (fixed)
    X_test = 2 * torch.bernoulli(0.5 * torch.ones((n_test, d))) - 1
    y_test = msp.evaluate(X_test)
    
    # Results storage
    results = []
    
    # For each training size
    for n_train in n_train_sizes:
        print(f"\nProcessing n_train = {n_train}")
        
        # Generate training data
        X_train = 2 * torch.bernoulli(0.5 * torch.ones((n_train, d))) - 1
        y_train = msp.evaluate(X_train)
        
        # Create uniform spectrum kernel
        K_train = create_uniform_spectrum_kernel(X_train)
        
        # Verify uniform spectrum
        eigenvalues = torch.linalg.eigvalsh(K_train)
        print(f"Eigenvalue statistics:")
        print(f"Mean: {eigenvalues.mean():.6f}")
        print(f"Std: {eigenvalues.std():.6f}")
        print(f"Min: {eigenvalues.min():.6f}")
        print(f"Max: {eigenvalues.max():.6f}")
        
        # Compute predictions and test error
        y_pred = compute_kernel_prediction(K_train, y_train, X_test, X_train)
        test_error = torch.mean((y_pred - y_test) ** 2).item()
        
        print(f"Test Error for n_train = {n_train}: {test_error:.6f}")
        
        # Store results
        result = {
            'n_train': n_train,
            'test_error': test_error,
            'eigenvalue_stats': {
                'mean': float(eigenvalues.mean()),
                'std': float(eigenvalues.std()),
                'min': float(eigenvalues.min()),
                'max': float(eigenvalues.max())
            }
        }
        results.append(result)
        
        # Save results after each iteration
        os.makedirs('uniform_kernel_results', exist_ok=True)
        with open(f'uniform_kernel_results/results_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Results saved for n_train = {n_train}")

if __name__ == "__main__":
    main()