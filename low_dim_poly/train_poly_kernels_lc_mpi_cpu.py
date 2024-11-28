#!/usr/bin/env python3
import os
import numpy as np
from mpi4py import MPI
from itertools import product
from functools import partial
import neural_tangents as nt
from jax import random
import jax.numpy as jnp
from sklearn.model_selection import train_test_split
from datetime import datetime
import json
from math import sqrt, ceil

def initialize_mpi():
    """Initialize MPI and return rank and size"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    return comm, rank, size

# Initialize MPI globally
COMM, RANK, SIZE = initialize_mpi()
print = partial(print, flush=True)

def generate_polynomials(r, d):
    """
    Generate all multi-indices where sum(alpha) <= d
    r: number of variables
    d: maximum degree
    """
    indices = [alpha for alpha in product(range(d + 1), repeat=r) if sum(alpha) <= d]
    return indices

def generate_latent_poly_data(n_samples, ambient_dim, latent_dim, degree, noise_std=0.1, random_state=None):
    """
    Generate synthetic data for learning polynomials with low-dimensional structure.
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    # Generate random orthogonal matrix for latent directions
    U, _ = np.linalg.qr(np.random.randn(ambient_dim, latent_dim))
    
    # Generate input data from N(0, Id)
    X = np.random.randn(n_samples, ambient_dim)
    
    # Project onto latent space
    X_latent = X @ U
    
    # Generate all polynomial terms
    terms = generate_polynomials(latent_dim, degree)
    
    # Initialize output
    y = np.zeros(n_samples)
    
    coeff_vec = []
    # Add each polynomial term
    for i, term in enumerate(terms):
        if sum(term) > 0:  # Skip constant term
            coef = np.random.randn()
            coeff_vec.append(coef)
            term_value = np.ones(n_samples)
            for dim, power in enumerate(term):
                if power > 0:
                    term_value *= X_latent[:, dim] ** power
            y += coef * term_value
            if RANK == 0:
                print(f"Term {i}: {term} (powers for each dimension)")
    
    if RANK == 0:
        # Print statistics
        total_terms = len(terms)
        non_constant_terms = len([t for t in terms if sum(t) > 0])
        print(f"\nStatistics:")
        print(f"Total terms (including constant): {total_terms}")
        print(f"Non-constant terms: {non_constant_terms}")
    
    # Normalize y to have unit variance
    y = y / np.std(y)
    
    # Add symmetric noise
    noise = noise_std * np.random.choice([-1, 1], size=n_samples)
    y = y + noise
    
    return X, y, U, coeff_vec

def compute_kernel_chunk(kernel_fn, x1_chunk, x2_chunk, get='ntk'):
    """Compute kernel for a chunk of data"""
    kernel = kernel_fn(x1_chunk, x2_chunk, get)
    return np.array(kernel)  # Convert JAX array to numpy array

def distributed_kernel_computation(kernel_fn, X, chunk_size=1000):
    """Compute kernel matrix in a distributed manner using MPI"""
    n_samples = X.shape[0]
    n_chunks = ceil(n_samples / chunk_size)
    
    # Determine chunks for this rank
    chunks_per_rank = ceil(n_chunks / SIZE)
    start_chunk = RANK * chunks_per_rank
    end_chunk = min((RANK + 1) * chunks_per_rank, n_chunks)
    
    kernel_chunks = []
    
    # Compute assigned chunks
    for i in range(start_chunk, end_chunk):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_samples)
        chunk = jnp.array(X[start_idx:end_idx])  # Ensure chunk is JAX array
        
        row_chunks = []
        for j in range(n_chunks):
            start_idx_col = j * chunk_size
            end_idx_col = min((j + 1) * chunk_size, n_samples)
            chunk_col = jnp.array(X[start_idx_col:end_idx_col])  # Ensure chunk_col is JAX array
            
            k_chunk = compute_kernel_chunk(kernel_fn, chunk, chunk_col)
            row_chunks.append(k_chunk)
        
        kernel_chunks.append(np.hstack(row_chunks))
    
    # Gather results from all ranks
    all_chunks = COMM.gather(kernel_chunks, root=0)
    
    if RANK == 0:
        # Combine chunks into final kernel matrix
        kernel_matrix = np.vstack([np.vstack(chunks) for chunks in all_chunks if chunks])
        return kernel_matrix
    return None

def create_network(input_dim, hidden_dim=1000):
    """Create neural network using neural_tangents matching the original architecture"""
    return nt.stax.serial(
        nt.stax.Dense(hidden_dim), nt.stax.Relu(),
        nt.stax.Dense(hidden_dim), nt.stax.Relu(),
        nt.stax.Dense(hidden_dim), nt.stax.Relu(),
        nt.stax.Dense(hidden_dim), nt.stax.Relu(),
        nt.stax.Dense(hidden_dim), nt.stax.Relu(),
        nt.stax.Dense(1)
    )

def main():
    # Training configurations (matching original exactly)
    train_sizes = [10, 500, 1000, 5000, 10000, 20000, 50000, 100000]
    test_size = 20000
    ambient_dim = 20
    latent_dim = 3
    degree = 5
    noise_std = 0.0
    hidden_dim = 400
    chunk_size = 1000
    
    if RANK == 0:
        # Create results directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_dir = os.path.join('results', f'kernel_training_run_{timestamp}')
        os.makedirs(base_dir, exist_ok=True)
        
        # Save hyperparameters (matching original format)
        hyperparams = {
            'train_sizes': train_sizes,
            'test_size': test_size,
            'ambient_dim': ambient_dim,
            'latent_dim': latent_dim,
            'degree': degree,
            'noise_std': noise_std,
            'model_architecture': {
                'input_dim': ambient_dim,
                'hidden_dim': hidden_dim,
                'num_layers': 5
            },
            'kernel_params': {
                'chunk_size': chunk_size,
                'n_mpi_processes': SIZE
            }
        }
        
        with open(os.path.join(base_dir, 'hyperparameters.json'), 'w') as f:
            json.dump(hyperparams, f, indent=4)
    
    # Initialize network
    init_fn, apply_fn, kernel_fn = create_network(ambient_dim, hidden_dim)
    
    # Results storage (matching original format)
    if RANK == 0:
        results = np.zeros((4, len(train_sizes)))  # [initial_train_error, initial_test_error, best_train_error, best_test_error]
    
    # Training loop for different dataset sizes
    for size_idx, train_size in enumerate(train_sizes):
        if RANK == 0:
            print(f"\nTraining with {train_size} samples")
        
        # Generate new dataset for this experiment
        total_size = train_size + test_size
        X, y, U, coeff_vec = generate_latent_poly_data(
            n_samples=total_size,
            ambient_dim=ambient_dim, 
            latent_dim=latent_dim,
            degree=degree,
            noise_std=noise_std,
            random_state=42
        )
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Convert to jax arrays
        X_train = jnp.array(X_train)
        y_train = jnp.array(y_train).reshape(-1, 1)
        X_test = jnp.array(X_test)
        y_test = jnp.array(y_test).reshape(-1, 1)
        
        if RANK == 0:
            print("Computing initial predictions...")
        
        # Compute initial predictions using NTK
        predict_fn = nt.predict.gradient_descent_mse_ensemble(
            kernel_fn,
            X_train,
            y_train,
            diag_reg=1e-4
        )
        
        # Get initial predictions
        if RANK == 0:
            initial_train_pred = predict_fn(x_test=X_train, get='ntk')
            initial_test_pred = predict_fn(x_test=X_test, get='ntk')
            
            # Calculate initial errors
            initial_train_error = np.mean((y_train - initial_train_pred) ** 2)
            initial_test_error = np.mean((y_test - initial_test_pred) ** 2)
            
            # Store initial results
            results[0, size_idx] = initial_train_error
            results[1, size_idx] = initial_test_error
            
            print("Computing kernel predictions...")
        
                # In the main function, replace the test kernel computation with:
        K_train = distributed_kernel_computation(kernel_fn, X_train, chunk_size)
        K_test = None
        if RANK == 0:
            # Compute test kernel matrix
            K_test = np.array(kernel_fn(X_test, X_train, 'ntk'))  # Fixed conversion from JAX array
            
            # Add ridge regularization
            K_train += 1e-4 * np.eye(K_train.shape[0])
            
            # Solve for predictions
            alpha = np.linalg.solve(K_train, y_train)
            y_pred_train = K_train @ alpha
            y_pred_test = K_test @ alpha
            
            # Compute final MSE
            final_train_error = np.mean((y_train - y_pred_train) ** 2)
            final_test_error = np.mean((y_test - y_pred_test) ** 2)
            
            # Store final results
            results[2, size_idx] = final_train_error
            results[3, size_idx] = final_test_error
            
            print(f"Initial - Train MSE: {initial_train_error:.6f}, Test MSE: {initial_test_error:.6f}")
            print(f"Final   - Train MSE: {final_train_error:.6f}, Test MSE: {final_test_error:.6f}")
    
    if RANK == 0:
        # Save results
        np.save(os.path.join(base_dir, 'training_results.npy'), results)
        
        # Print final results (matching original format)
        print("\nFinal Results:")
        print("Train Size | Initial Train | Initial Test | Best Train | Best Test")
        print("-" * 65)
        for i, size in enumerate(train_sizes):
            print(f"{size:9d} | {results[0,i]:11.6f} | {results[1,i]:11.6f} | "
                  f"{results[2,i]:10.6f} | {results[3,i]:9.6f}")

if __name__ == "__main__":
    main()