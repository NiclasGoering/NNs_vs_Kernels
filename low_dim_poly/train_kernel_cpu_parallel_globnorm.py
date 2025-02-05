#!/usr/bin/env python3


import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # Force JAX to use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from typing import List, Set
from functools import partial
import json
from datetime import datetime
from mpi4py import MPI
from itertools import product

# Define print globally first
print = partial(print, flush=True)

import jax
import jax.numpy as jnp
import neural_tangents as nt
from jax import random as jrandom

def generate_polynomials(r, d):
    """Generate all multi-indices where sum(alpha) <= d"""
    indices = [alpha for alpha in product(range(d + 1), repeat=r) if sum(alpha) <= d]
    return indices

def generate_latent_poly_data(n_samples, ambient_dim, latent_dim, degree, U=None, coeff_vec=None, noise_std=0.1, random_state=None):
    """Generate synthetic data without normalization"""
    if random_state is not None:
        np.random.seed(random_state)
        
    # Generate random orthogonal matrix for latent directions if not provided
    if U is None:
        U, _ = np.linalg.qr(np.random.randn(ambient_dim, latent_dim))
    
    # Generate input data from N(0, Id)
    X = np.random.randn(n_samples, ambient_dim)
    
    # Project onto latent space
    X_latent = X @ U
    
    # Generate all polynomial terms
    terms = generate_polynomials(latent_dim, degree)
    
    # Initialize output
    y = np.zeros(n_samples)
    
    if coeff_vec is None:
        coeff_vec = []
        # Generate new coefficients
        for term in terms:
            if sum(term) > 0:  # Skip constant term
                coef = np.random.randn()
                coeff_vec.append(coef)
    
    # Add each polynomial term using provided or generated coefficients
    coeff_idx = 0
    for term in terms:
        if sum(term) > 0:  # Skip constant term
            term_value = np.ones(n_samples)
            for dim, power in enumerate(term):
                if power > 0:
                    term_value *= X_latent[:, dim] ** power
            y += coeff_vec[coeff_idx] * term_value
            coeff_idx += 1
    
    # Add symmetric noise
    noise = noise_std * np.random.choice([-1, 1], size=n_samples)
    y = y + noise
    
    return X, y, U, coeff_vec

def generate_and_normalize_data(n_total, ambient_dim, latent_dim, degree, U=None, coeff_vec=None, noise_std=0.1, random_state=None):
    """Generate training and test data with proper normalization"""
    # Calculate train size (80% of total)
    n_train = int(0.8 * n_total)
    n_test = n_total - n_train
    
    # Generate data using provided U and coefficients
    X, y, U, coeff_vec = generate_latent_poly_data(
        n_samples=n_total,
        ambient_dim=ambient_dim,
        latent_dim=latent_dim,
        degree=degree,
        U=U,
        coeff_vec=coeff_vec,
        noise_std=noise_std,
        random_state=random_state
    )
    
    # Split into train and test
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    # Calculate normalization parameters from training data
    y_train_mean = np.mean(y_train)
    y_train_std = np.std(y_train)
    
    # Normalize both sets using training statistics
    y_train_normalized = (y_train - y_train_mean) / y_train_std
    y_test_normalized = (y_test - y_train_mean) / y_train_std
    
    return X_train, y_train_normalized.reshape(-1, 1), X_test, y_test_normalized.reshape(-1, 1)

def create_neural_tangent_model(d: int, hidden_size: int, depth: int,key=None):
    """Create a neural tangent kernel model"""
    layers = []
    for _ in range(depth):
        layers.extend([
            nt.stax.Dense(hidden_size), 
            nt.stax.Relu()
        ])
    layers.append(nt.stax.Dense(1))
    return nt.stax.serial(*layers)

def create_kernel_matrix_batched(X1: np.ndarray, X2: np.ndarray = None, k: int = 50000, 
                               batch_size: int = 10000, matmul_batch: int = 1000, 
                               random_seed: int = None) -> np.ndarray:
    """Creates a kernel matrix using random Gaussian features with batched computation."""
    if random_seed is not None:
        np.random.seed(random_seed)
        
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

def train_kernel(X_train, y_train, X_test, y_test, init_fn, apply_fn, kernel_fn, 
                kernel_type, k=50000, n_samples=None, random_seed=None):
    """Train using CPU with batched computation and consistent regularization"""
    # Use fixed regularization parameter
    lambda_reg = 1e-4  # Consistent regularization strength
    
    if kernel_type in ['ntk', 'nngp']:
        kernel_train = kernel_fn(X_train, None, kernel_type)
        kernel_test = kernel_fn(X_test, X_train, kernel_type)
        
        # Apply consistent regularization
        kernel_train = kernel_train + lambda_reg * np.eye(kernel_train.shape[0])
        predictions = np.dot(kernel_test, np.linalg.solve(kernel_train, y_train))
    
    elif kernel_type == 'random':
        feature_batch_size = 10000
        matmul_batch_size = 1000
        
        K_train = create_kernel_matrix_batched(
            X_train, None, k, 
            batch_size=feature_batch_size,
            matmul_batch=matmul_batch_size,
            random_seed=random_seed
        )
        K_test_train = create_kernel_matrix_batched(
            X_test, X_train, k,
            batch_size=feature_batch_size,
            matmul_batch=matmul_batch_size,
            random_seed=random_seed
        )
        
        # Apply consistent regularization
        reg_matrix = K_train + lambda_reg * np.eye(K_train.shape[0])
        predictions = np.dot(K_test_train, np.linalg.solve(reg_matrix, y_train))
    
    elif kernel_type == 'random_limit':
        K_train = create_random_limit_kernel(X_train)
        K_test_train = create_random_limit_kernel(X_test, X_train)
        
        # Apply consistent regularization
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
    
    # Fixed seed for data generation
    data_seed = 42
    
    # Parameters for low dimensional polynomials
    ambient_dim = 20
    latent_dim = 3
    degree = 5
    noise_std = 0.0
    
    # Parameters for training
    experiment_name = "low_dim_poly_kernels_normalized2"  # Changed name to reflect modifications
    hidden_sizes = [10,20,30,40,50,75,85,100,120,150,200,300,400,500,600,800,1000,1500,2000,3000,4000,5000,8000]
    hidden_sizes.reverse()
    depths = [4,1]
    n_train_sizes = [10,50,100,200,300,400,500,800,1000,2500,5000,8000,10000,15000,20000,30000,40000,60000]
    n_train_sizes.reverse()
    training_modes = ['ntk', 'nngp', 'random_limit']
    k = 200000  # number of random features
    
    # Generate all data once with fixed seed
    if rank == 0:
        # Generate training data
        max_train_size = max(n_train_sizes)
        X_full, y_full, U, coeff_vec = generate_latent_poly_data(
            n_samples=max_train_size,
            ambient_dim=ambient_dim,
            latent_dim=latent_dim,
            degree=degree,
            noise_std=noise_std,
            random_state=data_seed
        )
        
        # Generate fixed test data using same U and coeff_vec
        n_test = 10000  # Fixed test set size
        X_test, y_test = generate_latent_poly_data(
            n_samples=n_test,
            ambient_dim=ambient_dim,
            latent_dim=latent_dim,
            degree=degree,
            U=U,
            coeff_vec=coeff_vec,
            noise_std=noise_std,
            random_state=data_seed + 1  # Different seed but same polynomial
        )[:2]  # Only take X and y, not U and coeff_vec
        
        # Calculate global normalization statistics
        global_mean = float(np.mean(y_full))
        global_std = float(np.std(y_full))
    else:
        X_full = None
        y_full = None
        X_test = None
        y_test = None
        global_mean = None
        global_std = None
    
    # Broadcast datasets and normalization statistics to all workers
    X_full = comm.bcast(X_full, root=0)
    y_full = comm.bcast(y_full, root=0)
    X_test = comm.bcast(X_test, root=0)
    y_test = comm.bcast(y_test, root=0)
    global_mean = comm.bcast(global_mean, root=0)
    global_std = comm.bcast(global_std, root=0)
    
    # Create results directory (only master process)
    results_dir = f"low_dim_poly/results/{experiment_name}"
    if rank == 0:
        os.makedirs(results_dir, exist_ok=True)
        
        # Save hyperparameters
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        hyperparams = {
            'ambient_dim': ambient_dim,
            'latent_dim': latent_dim,
            'degree': degree,
            'noise_std': noise_std,
            'hidden_sizes': hidden_sizes,
            'depths': depths,
            'n_train_sizes': n_train_sizes,
            'training_modes': training_modes,
            'random_features': k,
            'num_workers': size,
            'data_seed': data_seed,
            'test_set_size': n_test,
            'global_mean': global_mean,
            'global_std': global_std,
            'regularization': 1e-4  # Added to track the fixed regularization
        }
        
        hyperparams_path = os.path.join(results_dir, f'hyperparameters_{timestamp}.json')
        with open(hyperparams_path, 'w') as f:
            json.dump(hyperparams, f, indent=4)
    else:
        timestamp = None
    
    # Broadcast timestamp to all workers
    timestamp = comm.bcast(timestamp, root=0)
    
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
        print(f"Total parameter combinations: {len(all_combinations)}")
        print(f"Combinations per worker: ~{len(all_combinations)/size}")
        print(f"Global normalization stats - mean: {global_mean:.6f}, std: {global_std:.6f}")
    
    # Process combinations assigned to this worker
    results = []
    for params in worker_combinations:
        print(f"Worker {rank} processing: {params}")
        
        # Get training data subset
        n_train = params['n_train']
        X_train = X_full[:n_train]
        y_train = y_full[:n_train]
        
        # Use global normalization statistics
        y_train_normalized = (y_train - global_mean) / global_std
        y_test_normalized = (y_test - global_mean) / global_std
        
        # Different initialization for each combination
        kernel_seed = hash((params['n_train'], params['hidden_size'], params['depth'], rank)) % 2**32
        key = jrandom.PRNGKey(kernel_seed)
        
        # Create the model functions with specific initialization
        init_fn, apply_fn, kernel_fn = create_neural_tangent_model(
            ambient_dim, 
            params['hidden_size'], 
            params['depth']
        )
        
        try:
            # Train and evaluate with different kernel initialization
            test_error = train_kernel(
                X_train, y_train_normalized.reshape(-1, 1),
                X_test, y_test_normalized.reshape(-1, 1),
                init_fn, apply_fn, kernel_fn,
                params['training_mode'], 
                k=k, 
                n_samples=params['n_train'],
                random_seed=kernel_seed
            )
            
            # Store results
            result = {
                'hidden_size': params['hidden_size'],
                'depth': params['depth'],
                'n_train': params['n_train'],
                'n_test': X_test.shape[0],
                'training_mode': params['training_mode'],
                'test_error': test_error,
                'status': 'success',
                'worker_rank': rank,
                'kernel_seed': kernel_seed,
                'global_mean': float(global_mean),  # Store global stats instead of local
                'global_std': float(global_std)
            }
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            result = {
                'hidden_size': params['hidden_size'],
                'depth': params['depth'],
                'n_train': params['n_train'],
                'n_test': X_test.shape[0],
                'training_mode': params['training_mode'],
                'test_error': None,
                'status': 'failed',
                'error': str(e),
                'worker_rank': rank,
                'kernel_seed': kernel_seed,
                'global_mean': float(global_mean),  # Store global stats instead of local
                'global_std': float(global_std)
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