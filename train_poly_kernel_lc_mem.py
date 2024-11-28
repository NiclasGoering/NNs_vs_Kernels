#!/usr/bin/env python3
import os
import numpy as np
from itertools import product
from functools import partial
import json
from datetime import datetime
import neural_tangents as nt
import jax.numpy as jnp
from mpi4py import MPI
import gc
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print = partial(print, flush=True)

def generate_polynomials(r, d):
    """Generate all multi-indices where sum(alpha) <= d"""
    indices = [alpha for alpha in product(range(d + 1), repeat=r) if sum(alpha) <= d]
    return indices

def parallel_generate_latent_poly_data(n_samples, ambient_dim, latent_dim, degree, 
                                     noise_std=0.1, random_state=None):
    """Generate synthetic data with MPI parallelization"""
    if random_state is not None:
        np.random.seed(random_state + rank)  # Different seed for each process
        
    # Process 0 generates shared components
    if rank == 0:
        U, _ = np.linalg.qr(np.random.randn(ambient_dim, latent_dim))
        terms = generate_polynomials(latent_dim, degree)
        coeff_vec = np.random.randn(len(terms))
    else:
        U = None
        terms = None
        coeff_vec = None
    
    # Broadcast shared components
    U = comm.bcast(U, root=0)
    terms = comm.bcast(terms, root=0)
    coeff_vec = comm.bcast(coeff_vec, root=0)
    
    # Calculate local portion size
    local_size = n_samples // size
    if rank < (n_samples % size):
        local_size += 1
    
    # Generate local portion of data
    X_local = np.random.randn(local_size, ambient_dim)
    X_latent = X_local @ U
    
    # Generate polynomial terms locally
    y_local = np.zeros(local_size)
    for i, term in enumerate(terms):
        if sum(term) > 0:  # Skip constant term
            term_value = np.ones(local_size)
            for dim, power in enumerate(term):
                if power > 0:
                    term_value *= X_latent[:, dim] ** power
            y_local += coeff_vec[i] * term_value
    
    # Gather all data at rank 0
    X_gathered = comm.gather(X_local, root=0)
    y_gathered = comm.gather(y_local, root=0)
    
    if rank == 0:
        X = np.concatenate(X_gathered)
        y = np.concatenate(y_gathered)
        
        # Normalize y to have unit variance
        y = y / np.std(y)
        
        # Add symmetric noise
        if noise_std > 0:
            noise = noise_std * np.random.choice([-1, 1], size=n_samples)
            y = y + noise
            
        return X, y, U
    return None, None, None

def parallel_kernel_computation(kernel_fn, X1, X2, batch_size=256):
    """Compute kernel matrix in parallel using MPI"""
    local_X1 = distribute_array(X1, comm)
    results = np.zeros((len(local_X1), len(X2)))
    
    for i in range(0, len(local_X1), batch_size):
        for j in range(0, len(X2), batch_size):
            i_end = min(i + batch_size, len(local_X1))
            j_end = min(j + batch_size, len(X2))
            
            batch_kernel = kernel_fn(local_X1[i:i_end], X2[j:j_end], 'nngp')
            results[i:i_end, j:j_end] = batch_kernel
            
            gc.collect()
    
    # Gather results
    gathered_results = comm.gather(results, root=0)
    if rank == 0:
        return np.concatenate(gathered_results, axis=0)
    return None

def distribute_array(array, comm):
    """Distribute array across MPI processes"""
    local_size = len(array) // comm.Get_size()
    remainder = len(array) % comm.Get_size()
    
    if rank < remainder:
        local_size += 1
        offset = rank * local_size
    else:
        offset = rank * local_size + remainder
    
    local_array = array[offset:offset + local_size]
    return local_array

def main():
    # Training configurations
    train_sizes = [10, 500, 1000, 5000, 10000, 20000, 50000, 100000]
    test_size = 20000
    n_samples = 2 * max(train_sizes)
    ambient_dim = 20
    latent_dim = 3
    degree = 5
    noise_std = 0.0
    hidden_dim = 400
    batch_size = 256
    
    if rank == 0:
        # Create results directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_dir = os.path.join('results', f'training_run_{timestamp}')
        os.makedirs(base_dir, exist_ok=True)
        
        # Save hyperparameters
        hyperparams = {
            'train_sizes': train_sizes,
            'test_size': test_size,
            'total_samples': n_samples,
            'ambient_dim': ambient_dim,
            'latent_dim': latent_dim,
            'degree': degree,
            'noise_std': noise_std,
            'model_architecture': {
                'input_dim': ambient_dim,
                'hidden_dim': hidden_dim,
                'num_layers': 5
            },
            'mpi_processes': size
        }
        
        with open(os.path.join(base_dir, 'hyperparameters.json'), 'w') as f:
            json.dump(hyperparams, f, indent=4)
    
    # Define neural network architecture
    init_fn, apply_fn, kernel_fn = nt.stax.serial(
        nt.stax.Dense(hidden_dim), nt.stax.Relu(),
        nt.stax.Dense(hidden_dim), nt.stax.Relu(),
        nt.stax.Dense(hidden_dim), nt.stax.Relu(),
        nt.stax.Dense(hidden_dim), nt.stax.Relu(),
        nt.stax.Dense(hidden_dim), nt.stax.Relu(),
        nt.stax.Dense(1)
    )
    
    # Generate dataset in parallel
    if rank == 0:
        print("Generating dataset...")
    X, y, U = parallel_generate_latent_poly_data(
        n_samples=n_samples,
        ambient_dim=ambient_dim,
        latent_dim=latent_dim,
        degree=degree,
        noise_std=noise_std,
        random_state=42
    )
    
    if rank == 0:
        # Split data
        X_remaining, X_test, y_remaining, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        # Initialize results array
        results = np.zeros((4, len(train_sizes)))
    else:
        X_remaining = y_remaining = X_test = y_test = None
    
    # Broadcast test data to all processes
    X_test = comm.bcast(X_test, root=0)
    y_test = comm.bcast(y_test, root=0)
    
    # Training loop
    for size_idx, train_size in enumerate(train_sizes):
        if rank == 0:
            print(f"\nTraining with {train_size} samples")
            indices = np.random.choice(len(X_remaining), train_size, replace=False)
            X_train = X_remaining[indices]
            y_train = y_remaining[indices]
        else:
            X_train = y_train = None
        
        # Broadcast training data
        X_train = comm.bcast(X_train, root=0)
        y_train = comm.bcast(y_train, root=0)
        
        # Parallel kernel computation
        K_train_train = parallel_kernel_computation(kernel_fn, X_train, X_train, batch_size)
        K_test_train = parallel_kernel_computation(kernel_fn, X_test, X_train, batch_size)
        
        if rank == 0:
            # Compute predictions
            predict_fn = nt.predict.gradient_descent_mse_ensemble(
                kernel_fn=kernel_fn,
                x_train=X_train,
                y_train=y_train.reshape(-1, 1),
                diag_reg=1e-4
            )
            
            # Get predictions
            train_predictions = predict_fn(x_test=X_train, get='nngp')
            test_predictions = predict_fn(x_test=X_test, get='nngp')
            
            # Calculate errors
            results[0, size_idx] = np.mean((train_predictions - y_train.reshape(-1, 1))**2)
            results[1, size_idx] = np.mean((test_predictions - y_test.reshape(-1, 1))**2)
            
            print(f"NNGP - Train Error: {results[0, size_idx]:.6f}, Test Error: {results[1, size_idx]:.6f}")
            
            # Save intermediate results
            np.save(os.path.join(base_dir, f'kernel_results_{size_idx}.npy'), results)
            
        comm.Barrier()
        gc.collect()
    
    if rank == 0:
        # Save final results
        np.save(os.path.join(base_dir, 'kernel_results.npy'), results)
        
        # Print final results
        print("\nFinal Results:")
        print("Train Size | NNGP Train | NNGP Test")
        print("-" * 45)
        for i, size in enumerate(train_sizes):
            print(f"{size:9d} | {results[0,i]:10.6f} | {results[1,i]:10.6f}")

if __name__ == "__main__":
    main()