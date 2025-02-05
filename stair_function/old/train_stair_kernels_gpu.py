#!/usr/bin/env python3

import os
# Set XLA flags for stable memory management
# os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_force_compilation_parallelism=1'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Reduce log spam
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['JAX_PLATFORM_NAME'] = 'gpu'

import numpy as np
from typing import List, Set
from functools import partial
import json
from datetime import datetime
# Define print globally first
print = partial(print, flush=True)

import jax
print("JAX version:", jax.__version__)
print("Available devices:", jax.devices())
print("GPU device count:", jax.device_count())

import jax
import jax.numpy as jnp
import neural_tangents as nt
from jax import random as jrandom 

 # Using jax.random consistently


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

def compute_random_features(X: np.ndarray, k: int) -> np.ndarray:
    """Map each input to an independent random Gaussian vector, normalized."""
    n_samples = X.shape[0]
    # Generate independent random Gaussian vectors
    Z = np.random.randn(n_samples, k).astype(np.float64)
    # Normalize by sqrt(k)
    Z = Z / np.sqrt(k)
    # Normalize each row
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / norms

def create_kernel_matrix(X1: np.ndarray, X2: np.ndarray = None, k: int = 50000) -> np.ndarray:
    """Creates a kernel matrix using random Gaussian features."""
    if X2 is None:
        X2 = X1
    
    # Compute random features for both sets
    Z1 = compute_random_features(X1, k)
    Z2 = Z1 if X2 is X1 else compute_random_features(X2, k)
    
    # Compute kernel matrix as K = ZZ^T
    return np.dot(Z1, Z2.T)


def compute_random_features_batched(X: np.ndarray, k: int, batch_size: int = 10000) -> np.ndarray:
    """Map each input to batches of independent random Gaussian vectors, normalized."""
    n_samples = X.shape[0]
    feature_batches = []
    
    # Generate and process random features in batches
    for i in range(0, k, batch_size):
        k_batch = min(batch_size, k - i)
        # Generate batch of random features
        Z_batch = np.random.randn(n_samples, k_batch).astype(np.float64)
        # Normalize by sqrt(k) for proper scaling
        Z_batch = Z_batch / np.sqrt(k)
        # Normalize each row
        norms = np.linalg.norm(Z_batch, axis=1, keepdims=True)
        Z_batch = Z_batch / norms
        feature_batches.append(Z_batch)
    
    return np.concatenate(feature_batches, axis=1)



def save_results(results: List[dict], results_dir: str, timestamp: str):
    """Helper function to save results with error handling"""
    try:
        results_path = os.path.join(results_dir, f'results_{timestamp}.json')
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print(f"Error saving results: {e}")


# def train_kernel(X_train, y_train, X_test, y_test, init_fn, apply_fn, kernel_fn, kernel_type, k=50000, n_samples=None):
#     """Train using hybrid GPU-CPU approach"""
#     if kernel_type in ['ntk', 'nngp']:
#         # Compute kernel on GPU
#         kernel_train = kernel_fn(X_train, None, kernel_type)  # Use GPU for kernel computation
#         kernel_test = kernel_fn(X_test, X_train, kernel_type)
        
#         # Move to CPU for solving
#         with jax.default_device(jax.devices('cpu')[0]):
#             # Transfer kernels to CPU (happens automatically with device context)
#             diag_reg = 1e-2 if (n_samples is not None and n_samples > 8000) else 1e-4
#             kernel_train = kernel_train + diag_reg * jnp.eye(kernel_train.shape[0])
            
#             # Solve on CPU
#             predictions = jnp.dot(kernel_test, jnp.linalg.solve(kernel_train, y_train))
    
#     elif kernel_type == 'random':
#         # Keep random feature kernel as is
#         K_train = create_kernel_matrix(X_train, None, k)
#         K_test_train = create_kernel_matrix(X_test, X_train, k)
        
#         # Move to CPU for solving
#         with jax.default_device(jax.devices('cpu')[0]):
#             lambda_reg = 1e-2 if (n_samples is not None and n_samples > 8000) else 1e-4
#             reg_matrix = K_train + lambda_reg * jnp.eye(K_train.shape[0])
#             predictions = jnp.dot(K_test_train, jnp.linalg.solve(reg_matrix, y_train))
    
#     test_error = jnp.mean((predictions - y_test) ** 2)
#     print(f"{kernel_type.upper()} Test Error: {test_error:.6f}")
    
#     return float(test_error)

def create_random_limit_kernel(X1: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
    """Creates a kernel matrix for infinite random features limit case."""
    if X2 is None:
        X2 = X1
        return np.eye(X1.shape[0])
    else:
        return np.zeros((X1.shape[0], X2.shape[0]))

def create_kernel_matrix_batched(X1: np.ndarray, X2: np.ndarray = None, k: int = 50000, batch_size: int = 10000, matmul_batch: int = 1000) -> np.ndarray:
    """Creates a kernel matrix using random Gaussian features with batched computation and batched matmul."""
    if X2 is None:
        X2 = X1
    
    n1, n2 = X1.shape[0], X2.shape[0]
    result = np.zeros((n1, n2))
    
    # Compute features in batches and immediately use them for matrix multiplication
    for i in range(0, k, batch_size):
        k_batch = min(batch_size, k - i)
        
        # Generate batch of random features for X1
        Z1_batch = np.random.randn(n1, k_batch).astype(np.float64)
        Z1_batch = Z1_batch / np.sqrt(k)
        norms1 = np.linalg.norm(Z1_batch, axis=1, keepdims=True)
        Z1_batch = Z1_batch / norms1
        
        # Generate batch of random features for X2
        if X2 is X1:
            Z2_batch = Z1_batch
        else:
            Z2_batch = np.random.randn(n2, k_batch).astype(np.float64)
            Z2_batch = Z2_batch / np.sqrt(k)
            norms2 = np.linalg.norm(Z2_batch, axis=1, keepdims=True)
            Z2_batch = Z2_batch / norms2
        
        # Batch the matrix multiplication
        for j in range(0, n1, matmul_batch):
            j_end = min(j + matmul_batch, n1)
            Z1_sub = Z1_batch[j:j_end]
            
            for l in range(0, n2, matmul_batch):
                l_end = min(l + matmul_batch, n2)
                Z2_sub = Z2_batch[l:l_end]
                
                # Update result matrix block
                result[j:j_end, l:l_end] += np.dot(Z1_sub, Z2_sub.T)
    
    return result

def train_kernel(X_train, y_train, X_test, y_test, init_fn, apply_fn, kernel_fn, kernel_type, k=50000, n_samples=None):
    """Train using hybrid GPU-CPU approach with fully batched random features"""
    if kernel_type in ['ntk', 'nngp']:
        # Compute kernel on GPU
        kernel_train = kernel_fn(X_train, None, kernel_type)
        kernel_test = kernel_fn(X_test, X_train, kernel_type)
        
        # Move to CPU for solving
        with jax.default_device(jax.devices('cpu')[0]):
            diag_reg = 1e-2 if (n_samples is not None and n_samples > 8000) else 1e-4
            kernel_train = kernel_train + diag_reg * jnp.eye(kernel_train.shape[0])
            predictions = jnp.dot(kernel_test, jnp.linalg.solve(kernel_train, y_train))
    
    elif kernel_type == 'random':
        # Parameters for batched computation
        feature_batch_size = 10000  # For random feature generation
        matmul_batch_size = 1000    # For matrix multiplication
        
        with jax.default_device(jax.devices('cpu')[0]):
            # Compute kernel matrices with batched matmul
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
            reg_matrix = K_train + lambda_reg * jnp.eye(K_train.shape[0])
            predictions = jnp.dot(K_test_train, jnp.linalg.solve(reg_matrix, y_train))

    elif kernel_type == 'random_limit':
        with jax.default_device(jax.devices('cpu')[0]):
            K_train = create_random_limit_kernel(X_train)
            K_test_train = create_random_limit_kernel(X_test, X_train)
            
            lambda_reg = 1e-2 if (n_samples is not None and n_samples > 8000) else 1e-4
            reg_matrix = K_train + lambda_reg * jnp.eye(K_train.shape[0])
            predictions = jnp.dot(K_test_train, jnp.linalg.solve(reg_matrix, y_train))
    
    test_error = jnp.mean((predictions - y_test) ** 2)
    print(f"{kernel_type.upper()} Test Error: {test_error:.6f}")
    
    return float(test_error)


def main():
    # Set random seeds
    np.random.seed(42)
    #random.seed(42)
    key = jrandom.PRNGKey(42)
    
    # Parameters
    experiment_name = "msp_NN_grid_1012_biggrid_kernel_2"
    P = 8 
    d = 30
    hidden_sizes = [10,20,30,40,50,75,85,100,120,150,200,300,400,500,600,800,1000,1500,2000,3000,4000]
    depths = [1,2,4,6,8]  
    n_test = 1000
    n_train_sizes = [10,50,100,200,300,400,500,800,1000,2500,5000,8000,10000,15000,20000]
    training_modes = ['ntk', 'nngp', 'random_limit'] #random
    k = 200000  # number of random features
    
    # Define the specific MSP sets
    msp_sets = [{7},{2, 7}, {0, 2, 7},{5, 7,4}, {1}, {0, 4}, {3, 7}, {0, 1, 2, 3, 4,6, 7}]
    
    # Create results directory
    results_dir = f"stair_function/results/{experiment_name}"
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
        'random_features': k
    }
    
    hyperparams_path = os.path.join(results_dir, f'hyperparameters_{timestamp}.json')
    with open(hyperparams_path, 'w') as f:
        json.dump(hyperparams, f, indent=4)
    
    # Initialize MSP function
    msp = MSPFunction(P, msp_sets)
    
    # Generate test data
    key, subkey = jrandom.split(key)
    X_test = 2 * jrandom.bernoulli(subkey, shape=(n_test, d), p=0.5).astype(np.float32) - 1
    y_test = msp.evaluate(X_test)
    
    # Results storage
    results = []
    
    print(f"Starting experiment: {experiment_name}")
    print(f"Results will be saved to: {results_dir}")
    print(f"Timestamp: {timestamp}")
    
    # Iterate over architectures and training sizes
    for hidden_size in hidden_sizes:
        for depth in depths:
            print(f"\nProcessing architecture: hidden_size={hidden_size}, depth={depth}")
            
            # Create the model functions
            init_fn, apply_fn, kernel_fn = create_neural_tangent_model(d, hidden_size, depth)
            
            for n_train in n_train_sizes:
                print(f"\nProcessing n_train = {n_train}")
                
                # Generate training data
                key, subkey = jrandom.split(key)
                X_train = 2 * jrandom.bernoulli(subkey, shape=(n_train, d), p=0.5).astype(np.float32) - 1
                y_train = msp.evaluate(X_train)
                
                for training_mode in training_modes:
                    print(f"\nTraining mode: {training_mode}")
                    
                    try:
                        # Train and evaluate
                        test_error = train_kernel(
                            X_train, y_train, X_test, y_test,
                            init_fn, apply_fn, kernel_fn,
                            training_mode, k, n_train
                        )
                        
                        # Store results
                        result = {
                            'hidden_size': hidden_size,
                            'depth': depth,
                            'n_train': n_train,
                            'training_mode': training_mode,
                            'test_error': test_error,
                            'status': 'success'
                        }
                        
                    except Exception as e:
                        print(f"Error during training: {str(e)}")
                        # Store error result
                        result = {
                            'hidden_size': hidden_size,
                            'depth': depth,
                            'n_train': n_train,
                            'training_mode': training_mode,
                            'test_error': None,
                            'status': 'failed',
                            'error': str(e)
                        }
                    
                    results.append(result)
                    
                    # Save results after each iteration
                    try:
                        save_results(results, results_dir, timestamp)
                        print(f"Results saved for architecture h{hidden_size}_d{depth}, "
                              f"n_train = {n_train}, mode = {training_mode}")
                        if result['status'] == 'success':
                            print(f"Test Error: {test_error:.6f}")
                    except Exception as e:
                        print(f"Error saving results: {str(e)}")

    print("\nExperiment completed!")
    print(f"Final results saved to: {results_dir}/results_{timestamp}.json")
    
if __name__ == "__main__":
    main()