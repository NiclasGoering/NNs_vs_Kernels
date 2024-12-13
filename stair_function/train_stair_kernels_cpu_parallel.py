#!/usr/bin/env python3

import os
import numpy as np
from typing import List, Set
from functools import partial
import json
from datetime import datetime
from mpi4py import MPI
# Define print globally first
print = partial(print, flush=True)

import jax
import jax.numpy as jnp
import neural_tangents as nt
from jax import random as jrandom 

class MSPFunction:
    def __init__(self, P: int, sets: List[Set[int]]):
        self.P = P
        self.sets = sets
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        batch_size = x.shape[0]
        result = np.zeros(batch_size, dtype=np.float64)
        
        for S in self.sets:
            term = np.ones(batch_size, dtype=np.float64)
            for idx in S:
                term = term * x[:, idx]
            result = result + term
            
        return result.reshape(-1, 1)

def create_neural_tangent_model(d: int, hidden_size: int, depth: int):
    """Create a neural tangent kernel model"""
    layers = []
    for _ in range(depth):
        layers.extend([
            nt.stax.Dense(hidden_size), 
            nt.stax.Relu()
        ])
    layers.append(nt.stax.Dense(1))
    return nt.stax.serial(*layers)

def create_kernel_matrix_batched(X1: np.ndarray, X2: np.ndarray = None, k: int = 50000, batch_size: int = 10000, matmul_batch: int = 1000) -> np.ndarray:
    """Creates a kernel matrix using random Gaussian features with batched computation."""
    if X2 is None:
        X2 = X1
    
    n1, n2 = X1.shape[0], X2.shape[0]
    result = np.zeros((n1, n2))
    
    for i in range(0, k, batch_size):
        k_batch = min(batch_size, k - i)
        
        Z1_batch = np.random.randn(n1, k_batch).astype(np.float64)
        Z1_batch = Z1_batch / np.sqrt(k)
        norms1 = np.linalg.norm(Z1_batch, axis=1, keepdims=True)
        Z1_batch = Z1_batch / norms1
        
        if X2 is X1:
            Z2_batch = Z1_batch
        else:
            Z2_batch = np.random.randn(n2, k_batch).astype(np.float64)
            Z2_batch = Z2_batch / np.sqrt(k)
            norms2 = np.linalg.norm(Z2_batch, axis=1, keepdims=True)
            Z2_batch = Z2_batch / norms2
        
        for j in range(0, n1, matmul_batch):
            j_end = min(j + matmul_batch, n1)
            Z1_sub = Z1_batch[j:j_end]
            
            for l in range(0, n2, matmul_batch):
                l_end = min(l + matmul_batch, n2)
                Z2_sub = Z2_batch[l:l_end]
                result[j:j_end, l:l_end] += np.dot(Z1_sub, Z2_sub.T)
    
    return result

def create_random_limit_kernel(X1: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
    """Creates a kernel matrix for infinite random features limit case."""
    if X2 is None:
        X2 = X1
        return np.eye(X1.shape[0])
    else:
        return np.zeros((X1.shape[0], X2.shape[0]))

def train_kernel(X_train, y_train, X_test, y_test, init_fn, apply_fn, kernel_fn, kernel_type, k=50000, n_samples=None):
    """Train using CPU with batched computation"""
    if kernel_type in ['ntk', 'nngp']:
        kernel_train = kernel_fn(X_train, None, kernel_type)
        kernel_test = kernel_fn(X_test, X_train, kernel_type)
        
        diag_reg = 1e-2 if (n_samples is not None and n_samples > 8000) else 1e-4
        kernel_train = kernel_train + diag_reg * np.eye(kernel_train.shape[0])
        predictions = np.dot(kernel_test, np.linalg.solve(kernel_train, y_train))
    
    elif kernel_type == 'random':
        feature_batch_size = 10000
        matmul_batch_size = 1000
        
        K_train = create_kernel_matrix_batched(
            X_train, None, k, 
            batch_size=feature_batch_size,
            matmul_batch=matmul_batch_size
        )
        K_test_train = create_kernel_matrix_batched(
            X_test, X_train, k,
            batch_size=feature_batch_size,
            matmul_batch=matmul_batch_size
        )
        
        lambda_reg = 1e-2 if (n_samples is not None and n_samples > 8000) else 1e-4
        reg_matrix = K_train + lambda_reg * np.eye(K_train.shape[0])
        predictions = np.dot(K_test_train, np.linalg.solve(reg_matrix, y_train))
    
    elif kernel_type == 'random_limit':
        K_train = create_random_limit_kernel(X_train)
        K_test_train = create_random_limit_kernel(X_test, X_train)
        
        lambda_reg = 1e-2 if (n_samples is not None and n_samples > 8000) else 1e-4
        reg_matrix = K_train + lambda_reg * np.eye(K_train.shape[0])
        predictions = np.dot(K_test_train, np.linalg.solve(reg_matrix, y_train))
    
    test_error = np.mean((predictions - y_test) ** 2)
    print(f"{kernel_type.upper()} Test Error: {test_error:.6f}")
    
    return float(test_error)

def get_parameter_combinations(hidden_sizes, depths, n_train_sizes, training_modes):
    """Generate all possible hyperparameter combinations"""
    combinations = []
    for hidden_size in hidden_sizes:
        for depth in depths:
            for n_train in n_train_sizes:
                for mode in training_modes:
                    combinations.append({
                        'hidden_size': hidden_size,
                        'depth': depth,
                        'n_train': n_train,
                        'training_mode': mode
                    })
    return combinations

def save_results(results: List[dict], results_dir: str, timestamp: str):
    """Helper function to save results with error handling"""
    try:
        results_path = os.path.join(results_dir, f'results_{timestamp}.json')
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print(f"Error saving results: {e}")

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Set random seeds
    np.random.seed(42 + rank)  # Different seed for each worker
    key = jrandom.PRNGKey(42 + rank)
    
    # Parameters
    experiment_name = "msp_NN_grid_1012_biggrid_kernel_cpu"
    P = 8 
    d = 30
    hidden_sizes = [10,20,30,40,50,75,85,100,120,150,200,300,400,500,600,800,1000,1500,2000,3000,4000]
    depths = [1,2,4,6,8]  
    n_test = 1000
    n_train_sizes = [10,50,100,200,300,400,500,800,1000,2500,5000,8000,10000,15000,20000]
    training_modes = ['ntk', 'nngp', 'random_limit']
    k = 200000  # number of random features
    
    # Define the specific MSP sets
    msp_sets = [{7},{2, 7}, {0, 2, 7},{5, 7,4}, {1}, {0, 4}, {3, 7}, {0, 1, 2, 3, 4,6, 7}]
    
    # Create results directory (only master process)
    results_dir = f"stair_function/results/{experiment_name}"
    if rank == 0:
        os.makedirs(results_dir, exist_ok=True)
        
        # Save hyperparameters
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        hyperparams = {
            'P': P,
            'd': d,
            'hidden_sizes': hidden_sizes,
            'depths': depths,
            'n_test': n_test,
            'n_train_sizes': n_train_sizes,
            'msp_sets': [list(s) for s in msp_sets],
            'training_modes': training_modes,
            'random_features': k,
            'num_workers': size
        }
        
        hyperparams_path = os.path.join(results_dir, f'hyperparameters_{timestamp}.json')
        with open(hyperparams_path, 'w') as f:
            json.dump(hyperparams, f, indent=4)
    else:
        timestamp = None
    
    # Broadcast timestamp to all workers
    timestamp = comm.bcast(timestamp, root=0)
    
    # Initialize MSP function
    msp = MSPFunction(P, msp_sets)
    
    # Generate test data (same for all workers)
    key, subkey = jrandom.split(key)
    X_test = 2 * jrandom.bernoulli(subkey, shape=(n_test, d), p=0.5).astype(np.float32) - 1
    y_test = msp.evaluate(X_test)
    
    # Generate all parameter combinations
    all_combinations = get_parameter_combinations(hidden_sizes, depths, n_train_sizes, training_modes)
    
    # Distribute work among workers
    worker_combinations = []
    for i in range(len(all_combinations)):
        if i % size == rank:
            worker_combinations.append(all_combinations[i])
    
    if rank == 0:
        print(f"\nStarting experiment: {experiment_name}")
        print(f"Results will be saved to: {results_dir}")
        print(f"Timestamp: {timestamp}")
        print(f"Number of workers: {size}")
    
    # Process combinations assigned to this worker
    results = []
    for params in worker_combinations:
        print(f"Worker {rank} processing: {params}")
        
        # Generate training data
        key, subkey = jrandom.split(key)
        X_train = 2 * jrandom.bernoulli(subkey, shape=(params['n_train'], d), p=0.5).astype(np.float32) - 1
        y_train = msp.evaluate(X_train)
        
        # Create the model functions
        init_fn, apply_fn, kernel_fn = create_neural_tangent_model(d, params['hidden_size'], params['depth'])
        
        try:
            # Train and evaluate
            test_error = train_kernel(
                X_train, y_train, X_test, y_test,
                init_fn, apply_fn, kernel_fn,
                params['training_mode'], k, params['n_train']
            )
            
            # Store results
            result = {
                'hidden_size': params['hidden_size'],
                'depth': params['depth'],
                'n_train': params['n_train'],
                'training_mode': params['training_mode'],
                'test_error': test_error,
                'status': 'success',
                'worker_rank': rank
            }
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            result = {
                'hidden_size': params['hidden_size'],
                'depth': params['depth'],
                'n_train': params['n_train'],
                'training_mode': params['training_mode'],
                'test_error': None,
                'status': 'failed',
                'error': str(e),
                'worker_rank': rank
            }
        
        results.append(result)
        
        # Save intermediate results for this worker
        save_results(results, results_dir, f'{timestamp}_rank{rank}')
        print(f"Worker {rank} completed {params}")
    
    # Wait for all workers to complete
    comm.Barrier()
    
    # Gather all results to master process
    all_results = comm.gather(results, root=0)
    
    # Master process combines and saves all results
    if rank == 0:
        combined_results = []
        for worker_results in all_results:
            combined_results.extend(worker_results)
        
        final_results_path = os.path.join(results_dir, f'final_results_{timestamp}.json')
        with open(final_results_path, 'w') as f:
            json.dump(combined_results, f, indent=4)
        
        print("\nExperiment completed!")
        print(f"Final results saved to: {results_dir}/final_results_{timestamp}.json")

if __name__ == "__main__":
    main()