#!/usr/bin/env python3

"""
Example code using JAX + Neural Tangents (stax) to train an MLP on an MSP function.
We compute and save empirical NTKs at initial and final parameters,
as well as the test MSE from the initial/final NTK.
Modified to include batched NTK computation and multiple experiment runs.
"""

import os
import json
import numpy as np
from datetime import datetime
from functools import partial
from typing import List, Set, Tuple, Any

# MPI
from mpi4py import MPI

# JAX / neural-tangents / optax
import jax
import jax.numpy as jnp
from jax import random as jax_random
import optax
import neural_tangents as nt
from neural_tangents import stax as nt_stax

# Flush-print
print = partial(print, flush=True)


# -------------------
# MSP Function
# -------------------
class MSPFunction:
    """
    MSP function: sum of products of certain subsets of input coordinates.
    Each subset is in 'sets'.
    """
    def __init__(self, P: int, sets: List[Set[int]]):
        self.P = P
        self.sets = sets

    def evaluate(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate MSP function on a batch of inputs z of shape (batch_size, d).
        Each subset S in self.sets contributes product(z[:, idx in S]).
        """
        batch_size = z.shape[0]
        result = jnp.zeros(batch_size, dtype=jnp.float32)

        for S in self.sets:
            subset_vals = jnp.prod(z[:, list(S)], axis=1)
            result = result + subset_vals
        return result


def center_matrix(K):
    """Center a kernel matrix K"""
    n = K.shape[0]
    unit = jnp.ones([n, n])
    I = jnp.eye(n)
    H = I - unit / n
    return H @ K @ H

def compute_cka_batched(K1, K2, batch_size=1000):
    """Compute CKA in a memory-efficient batched manner"""
    n = K1.shape[0]
    
    # Initialize accumulators for HSIC and norms
    hsic_sum = 0.0
    norm1_sum = 0.0
    norm2_sum = 0.0
    
    # Compute means for centering
    K1_mean = jnp.mean(K1)
    K2_mean = jnp.mean(K2)
    K1_row_means = jnp.mean(K1, axis=1)
    K2_row_means = jnp.mean(K2, axis=1)
    
    # Process in batches
    for i in range(0, n, batch_size):
        end_idx = min(i + batch_size, n)
        batch_size_i = end_idx - i
        
        # Get batches
        K1_batch = K1[i:end_idx]
        K2_batch = K2[i:end_idx]
        
        # Center batches
        K1_centered = (K1_batch - K1_row_means[i:end_idx, None] - 
                      K1_row_means[None, :] + K1_mean)
        K2_centered = (K2_batch - K2_row_means[i:end_idx, None] - 
                      K2_row_means[None, :] + K2_mean)
        
        # Accumulate HSIC
        hsic_sum += jnp.sum(K1_centered * K2_centered)
        
        # Accumulate norms
        norm1_sum += jnp.sum(K1_centered * K1_centered)
        norm2_sum += jnp.sum(K2_centered * K2_centered)
    
    # Compute final CKA
    return hsic_sum / (jnp.sqrt(norm1_sum) * jnp.sqrt(norm2_sum))


# def compute_frob_norm_batched_fixed(K1, K2, batch_size=1000):
#     """
#     Compute normalized Frobenius norm between two matrices in batches.
#     Returns a value between 0 and 1.
#     """
#     n = K1.shape[0]
    
#     # First pass: compute norms
#     norm1_sq = 0.0
#     norm2_sq = 0.0
#     for i in range(0, n, batch_size):
#         end = min(i + batch_size, n)
#         K1_batch = K1[i:end]
#         K2_batch = K2[i:end]
#         norm1_sq += jnp.sum(K1_batch * K1_batch)
#         norm2_sq += jnp.sum(K2_batch * K2_batch)
    
#     # Add small epsilon to avoid division by zero
#     eps = 1e-10
#     norm1 = jnp.sqrt(norm1_sq + eps)
#     norm2 = jnp.sqrt(norm2_sq + eps)
    
#     # Second pass: compute similarity
#     similarity = 0.0
#     for i in range(0, n, batch_size):
#         end = min(i + batch_size, n)
#         K1_batch = K1[i:end] / norm1
#         K2_batch = K2[i:end] / norm2
#         similarity += jnp.sum(K1_batch * K2_batch)
    
#     # The normalized distance is sqrt(2 - 2*cosine_similarity)
#     # where cosine_similarity = similarity/(norm1*norm2)
#     # This ensures the result is between 0 and 1
#     distance = jnp.sqrt(2.0 - 2.0 * similarity)
    
#     # Clip to [0,1] to handle any numerical instability
#     return jnp.clip(distance, 0.0, 1.0)


import jax
import jax.numpy as jnp
def frob_norm_batched(K, batch_size=1024):
    """
    Compute the Frobenius norm of a 2D matrix K in batches:
        ||K||_F = sqrt( sum_{i,j} K[i,j]^2 ).
    """
    n = K.shape[0]
    total_sq = 0.0
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        # Sum of squares in this batch of rows
        batch = K[i:end]
        total_sq += jnp.sum(batch * batch)
    return jnp.sqrt(total_sq)

def frob_norm_diff_batched(K1, K2, batch_size=1024):
    """
    Compute the Frobenius norm of (K1 - K2) in batches:
        ||K1 - K2||_F = sqrt( sum_{i,j} (K1[i,j] - K2[i,j])^2 ).
    """
    n = K1.shape[0]
    total_sq = 0.0
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        diff = K1[i:end] - K2[i:end]
        total_sq += jnp.sum(diff * diff)
    return jnp.sqrt(total_sq)

def compute_frob_norm_batched_fixed(K1, K2, batch_size=1024):
    """
    Compute the relative Frobenius difference between two matrices:
        ||K1 - K2||_F / (0.5 * (||K1||_F + ||K2||_F))
    """
    eps = 1e-12  # Keep small epsilon to avoid division by zero
    
    # Compute Frobenius norms
    norm1 = frob_norm_batched(K1, batch_size)
    norm2 = frob_norm_batched(K2, batch_size)
    diff = frob_norm_diff_batched(K1, K2, batch_size)
    
    # Use average of norms for normalization
    denominator = 0.5 * (norm1 + norm2) + eps
    ratio = diff / denominator
    
    # Optional: clip to [0,2] since with this normalization, 
    # the maximum possible value is 2 (when one matrix is the negative of the other)
    return jnp.clip(ratio, 0.0, 2.0)



def compute_frob_norm_batched_fixed(K1, K2, batch_size=1024):
    """
    Compute a relative element-wise difference:
    mean(|K1[i,j] - K2[i,j]| / max(|K1[i,j]|, |K2[i,j]|))
    """
    eps = 1e-12
    n = K1.shape[0]
    total_diff = 0.0
    count = 0
    
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        K1_batch = K1[i:end]
        K2_batch = K2[i:end]
        
        # Element-wise absolute differences
        diff = jnp.abs(K1_batch - K2_batch)
        # Element-wise maximum magnitudes
        max_vals = jnp.maximum(jnp.abs(K1_batch), jnp.abs(K2_batch)) + eps
        # Relative differences
        rel_diff = diff / max_vals
        
        total_diff += jnp.sum(rel_diff)
        count += rel_diff.size
    
    return total_diff / count


def compute_ntk_metrics_batched(initial_ntk, final_ntk, batch_size=1000, k_top=10):
    """
    Compute multiple NTK similarity metrics:
    1. Standard normalized Frobenius norm (fixed)
    2. Eigenvalue distribution comparison (Wasserstein distance)
    3. Dominant eigenvector changes (subspace alignment)
    """
    if initial_ntk is None or final_ntk is None:
        return {
            'cka': float('nan'),
            'frob_norm': float('nan'),
            'eig_dist': float('nan'),
            'vec_align': float('nan')
        }
    
    try:
        # 1. Proper normalized Frobenius norm
        diff_norm = jnp.linalg.norm(initial_ntk - final_ntk, ord='fro')
        init_norm = jnp.linalg.norm(initial_ntk, ord='fro')
        final_norm = jnp.linalg.norm(final_ntk, ord='fro')
        frob_norm = diff_norm / jnp.maximum(init_norm, final_norm)
        
        # 2. Eigenvalue distribution comparison
        init_eigs = jnp.linalg.eigvalsh(initial_ntk)
        final_eigs = jnp.linalg.eigvalsh(final_ntk)
        
        # Sort and normalize eigenvalues
        init_eigs = jnp.sort(init_eigs)[::-1] / jnp.sum(jnp.abs(init_eigs))
        final_eigs = jnp.sort(final_eigs)[::-1] / jnp.sum(jnp.abs(final_eigs))
        
        # Compute approximate Wasserstein distance (1D case)
        eig_dist = jnp.mean(jnp.abs(init_eigs - final_eigs))
        
        # 3. Dominant eigenvector alignment
        _, init_vecs = jnp.linalg.eigh(initial_ntk)
        _, final_vecs = jnp.linalg.eigh(final_ntk)
        
        # Take top k eigenvectors
        init_space = init_vecs[:, -k_top:]
        final_space = final_vecs[:, -k_top:]
        
        # Compute principal angles using SVD
        cross = init_space.T @ final_space
        s = jnp.linalg.svd(cross, compute_uv=False)
        vec_align = jnp.mean(s)  # 1 means perfectly aligned, 0 means orthogonal
        
        # Compute CKA as before
        cka = float(compute_cka_batched(initial_ntk, final_ntk, batch_size))
        
        return {
            'cka': cka,
            'frob_norm': float(frob_norm),
            'eig_dist': float(eig_dist),
            'vec_align': float(vec_align)
        }
        
    except Exception as e:
        print(f"Error computing batched NTK metrics: {e}")
        return {
            'cka': float('nan'),
            'frob_norm': float('nan'),
            'eig_dist': float('nan'),
            'vec_align': float('nan')
        }

# -------------------
# Data Generation
# -------------------
def generate_master_dataset(P: int, d: int, master_size: int, n_test: int, 
                          msp: MSPFunction, seed: int = 42):
    """Generate master training set + test set with a fixed seed"""
    np.random.seed(seed)
    X_train_master_np = 2.0 * np.random.binomial(1, 0.5, size=(master_size, d)) - 1.0
    X_test_np = 2.0 * np.random.binomial(1, 0.5, size=(n_test, d)) - 1.0

    X_train_master = jnp.array(X_train_master_np, dtype=jnp.float32)
    X_test = jnp.array(X_test_np, dtype=jnp.float32)
    y_train_master = msp.evaluate(X_train_master)
    y_test = msp.evaluate(X_test)
    return X_train_master, y_train_master, X_test, y_test


def shuffle_labels(y: jnp.ndarray, key: Any) -> jnp.ndarray:
    """Shuffle labels randomly using a JAX random key"""
    perm = jax_random.permutation(key, len(y))
    return y[perm]


# -------------------
# Create Network
# -------------------
def create_mlp(d: int, hidden_size: int, depth: int):
    """Create MLP with ReLU hidden layers using nt_stax"""
    layers = []
    W_std = 1.0 / (d ** 0.5)  # He initialization scaling
    
    for _ in range(depth):
        layers += [
            nt_stax.Dense(hidden_size, W_std=W_std),
            nt_stax.Relu()
        ]
    
    layers.append(nt_stax.Dense(1, W_std=W_std))
    init_fn, apply_fn, kernel_fn = nt_stax.serial(*layers)
    return init_fn, apply_fn, kernel_fn


# -------------------
# Loss and NTK Functions
# -------------------
def compute_mse(params: Any, apply_fn: Any, inputs: jnp.ndarray, 
                targets: jnp.ndarray, gamma: float = 1.0) -> float:
    """Compute MSE loss with gamma scaling"""
    predictions = apply_fn(params, inputs).squeeze(-1)
    # Scale predictions by 1/gamma before computing loss
    scaled_predictions = predictions / gamma
    return jnp.mean((scaled_predictions - targets) ** 2)

def compute_empirical_ntk_batched(apply_fn, params, x1, x2=None, batch_size=512):
    """Compute empirical NTK in batches to avoid memory issues."""
    ntk_fn = nt.empirical_ntk_fn(
        apply_fn,
        vmap_axes=0,
        trace_axes=(-1,)
    )
    
    n1 = len(x1)
    n2 = len(x2) if x2 is not None else n1
    
    # Initialize output matrix
    ntk = jnp.zeros((n1, n2))
    
    # Compute NTK in batches
    for i in range(0, n1, batch_size):
        i_end = min(i + batch_size, n1)
        x1_batch = x1[i:i_end]
        
        if x2 is None:
            # For train-train kernel
            for j in range(0, i_end, batch_size):
                j_end = min(j + batch_size, i_end)
                x2_batch = x1[j:j_end]
                ntk_batch = ntk_fn(x1_batch, x2_batch, params)
                ntk = ntk.at[i:i_end, j:j_end].set(ntk_batch)
                # Mirror the computation due to symmetry
                if i != j:
                    ntk = ntk.at[j:j_end, i:i_end].set(ntk_batch.T)
        else:
            # For test-train kernel
            for j in range(0, n2, batch_size):
                j_end = min(j + batch_size, n2)
                x2_batch = x2[j:j_end]
                ntk_batch = ntk_fn(x1_batch, x2_batch, params)
                ntk = ntk.at[i:i_end, j:j_end].set(ntk_batch)
    
    return ntk


# -------------------
# Training
# -------------------
def train_and_evaluate(init_fn: Any, apply_fn: Any, msp: MSPFunction,
                      X_train: jnp.ndarray, y_train: jnp.ndarray, 
                      X_test: jnp.ndarray, y_test: jnp.ndarray,
                      batch_size: int, epochs: int, lr: float, 
                      weight_decay: float, gamma: float, key: Any):
    """Train with AdamW and cosine annealing, compute empirical NTKs"""
    
    # Initialize model
    output_shape, params = init_fn(key, (-1,) + X_train.shape[1:])
    
    # Compute initial empirical NTK
    print("Computing initial NTK...")
    try:
        initial_ntk = compute_empirical_ntk_batched(apply_fn, params, X_train)
        initial_ntk_test = compute_empirical_ntk_batched(apply_fn, params, X_test, X_train)
        
        # Solve ridge regression with NTK
        reg = 1e-6 * jnp.trace(initial_ntk) / len(X_train)
        reg_ntk = initial_ntk + reg * jnp.eye(len(X_train))
        alpha = jnp.linalg.solve(reg_ntk, y_train)
        y_pred_ntk = initial_ntk_test @ alpha
        initial_ntk_error = jnp.mean((y_pred_ntk - y_test) ** 2)
        print(f"Initial NTK test error: {initial_ntk_error:.6f}")
    except Exception as e:
        print(f"Warning: Initial NTK computation failed: {e}")
        print("Setting initial NTK error to NaN")
        initial_ntk_error = float('nan')
        initial_ntk = None
    
    # Create optimizer with cosine decay schedule
    steps_per_epoch = max(1, len(X_train) // batch_size)
    total_steps = epochs * steps_per_epoch
    
    scheduler = optax.cosine_decay_schedule(
        init_value=lr,
        decay_steps=total_steps,
        alpha=0.0
    )
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=scheduler,
            weight_decay=weight_decay
        )
    )
    
    opt_state = optimizer.init(params)
    
    # JIT compile training step
    @jax.jit
    def train_step(params, opt_state, batch_x, batch_y):
        def loss_fn(p):
            return compute_mse(p, apply_fn, batch_x, batch_y, gamma)
        
        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_val
    
    # Training loop
    best_test_error = float('inf')
    training_history = {
        'train_errors': [],
        'test_errors': [],
        'epochs': []
    }
    
    print("Starting training...")
    
    for epoch in range(epochs):
        # Shuffle training data
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, len(X_train))
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]
        
        # Mini-batch updates
        epoch_losses = []
        for i in range(0, len(X_train), batch_size):
            batch_x = X_train_shuffled[i:i + batch_size]
            batch_y = y_train_shuffled[i:i + batch_size]
            params, opt_state, loss = train_step(params, opt_state, batch_x, batch_y)
            epoch_losses.append(loss)
        
        # Evaluate periodically
        if epoch % 100 == 0 or epoch == epochs - 1:
            train_error = jnp.mean(jnp.array(epoch_losses))
            test_pred = apply_fn(params, X_test).squeeze(-1) / gamma  # Scale predictions
            test_error = jnp.mean((test_pred - y_test) ** 2)
            best_test_error = min(best_test_error, test_error)
            
            training_history['train_errors'].append(float(train_error))
            training_history['test_errors'].append(float(test_error))
            training_history['epochs'].append(epoch)
            
            print(f"Epoch {epoch}")
            print(f"Train Error: {train_error:.6f}")
            print(f"Test Error: {test_error:.6f}")
            print(f"Best Test Error: {best_test_error:.6f}")
    
    print("Computing final NTK...")
    # Compute final empirical NTK
    try:
        final_ntk = compute_empirical_ntk_batched(apply_fn, params, X_train)
        final_ntk_test = compute_empirical_ntk_batched(apply_fn, params, X_test, X_train)
        
        # Solve ridge regression with final NTK
        reg = 1e-6 * jnp.trace(final_ntk) / len(X_train)
        reg_ntk = final_ntk + reg * jnp.eye(len(X_train))
        alpha = jnp.linalg.solve(reg_ntk, y_train)
        y_pred_ntk = final_ntk_test @ alpha
        final_ntk_error = jnp.mean((y_pred_ntk - y_test) ** 2)
        print(f"Final NTK test error: {final_ntk_error:.6f}")
    except Exception as e:
        print(f"Warning: Final NTK computation failed: {e}")
        print("Setting final NTK error to NaN")
        final_ntk_error = float('nan')
        final_ntk = None
    
    print("Computing NTK similarity metrics...")
    ntk_metrics = compute_ntk_metrics_batched(initial_ntk, final_ntk, batch_size=1024)
    print(f"CKA between initial and final NTK: {ntk_metrics['cka']:.6f}")
    print(f"Normalized Frobenius norm: {ntk_metrics['frob_norm']:.6f}")
    print(f"Eigenvalue distribution distance: {ntk_metrics['eig_dist']:.6f}")
    print(f"Top eigenvector alignment: {ntk_metrics['vec_align']:.6f}")
    
    return (params, best_test_error, training_history, 
            initial_ntk_error, final_ntk_error, 
            initial_ntk, final_ntk,
            ntk_metrics)


# -------------------
# Save Functions
# -------------------
def save_results(results: List[dict], results_dir: str, filename_suffix: str):
    """Save results to JSON"""
    try:
        path = os.path.join(results_dir, f'results_{filename_suffix}.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print(f"Error saving results: {e}")


def save_dataset(X: jnp.ndarray, y: jnp.ndarray, path: str, rank: int):
    """Save dataset as NPZ"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path,
                 X=np.array(X),
                 y=np.array(y),
                 saved_by_rank=rank,
                 timestamp=datetime.now().isoformat())
        return True
    except Exception as e:
        print(f"Error saving dataset: {e}")
        return False


# -------------------
# Main
# -------------------
def main():
    # MPI initialization
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Parameters
    base_experiment_name = "jax_mspgpu_20k_h1000_lr0005_metrictest"
    P = 8
    d = 30
    master_size = 35000
    hidden_sizes = [400,1000]
    hidden_sizes.reverse()
    depths = [4]
    n_test = 5000
    batch_size = 64
    epochs = 3000
    learning_rates = [0.005]
    weight_decay = 1e-4
    gammas = [0.01]  # New gamma parameter
    shuffled = False
    n_train_sizes = [10, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 20000]
    #n_train_sizes.reverse()
    num_experiments = 3

    # MSP sets
    msp_sets = [{7}, {2,7}, {0,2,7}, {5,7,4}, {1}, {0,4}, {3,7}, 
                {0,1,2,3,4,6,7}]
    msp = MSPFunction(P, msp_sets)

    # Run multiple experiments
    for exp_num in range(1, num_experiments + 1):
        if rank == 0:
            print(f"\n{'='*50}")
            print(f"Starting experiment run {exp_num}/{num_experiments}")
            print(f"{'='*50}\n")

        # Create experiment-specific name and directories
        experiment_name = f"{base_experiment_name}_{exp_num}"
        results_dir = os.path.join("/mnt/users/goringn/NNs_vs_Kernels", "/mnt/users/goringn/NNs_vs_Kernels/NTK/results", experiment_name)
        data_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        # Generate timestamp
        if rank == 0:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            hyperparams = {
                'experiment_run': exp_num,
                'P': P,
                'd': d,
                'hidden_sizes': hidden_sizes,
                'depths': depths,
                'n_test': n_test,
                'batch_size': batch_size,
                'epochs': epochs,
                'learning_rates': learning_rates,
                'weight_decay': weight_decay,
                'gammas': gammas,
                'shuffled': shuffled,
                'n_train_sizes': n_train_sizes,
                'msp_sets': [list(s) for s in msp_sets],
                'num_workers': size
            }
            
            hyperparams_path = os.path.join(results_dir, 
                                          f'hyperparameters_{timestamp}.json')
            with open(hyperparams_path, 'w') as f:
                json.dump(hyperparams, f, indent=4)
        else:
            timestamp = None

        timestamp = comm.bcast(timestamp, root=0)

        # Generate datasets (rank 0)
        if rank == 0:
            print("Master generating datasets...")
            key = jax_random.PRNGKey(exp_num)
            X_train_master, y_train_master, X_test, y_test = generate_master_dataset(
                P, d, master_size, n_test, msp)
            print("Data generation complete")
            print(f"Master set shape: {X_train_master.shape}")
            print(f"Test set shape: {X_test.shape}")

            # Save master dataset
            master_data_path = os.path.join(data_dir, f'master_data_{timestamp}.npz')
            if not save_dataset(X_train_master, y_train_master, master_data_path, rank):
                raise RuntimeError("Failed to save master dataset")

            # Save test dataset
            test_data_path = os.path.join(data_dir, f'test_data_{timestamp}.npz')
            if not save_dataset(X_test, y_test, test_data_path, rank):
                raise RuntimeError("Failed to save test dataset")

            print("Master saved all datasets successfully")
        else:
            X_train_master = None
            y_train_master = None
            X_test = None
            y_test = None

        # Broadcast datasets to all workers
        X_train_master = comm.bcast(X_train_master, root=0)
        y_train_master = comm.bcast(y_train_master, root=0)
        X_test = comm.bcast(X_test, root=0)
        y_test = comm.bcast(y_test, root=0)

        # Generate parameter combinations
        all_combinations = [
            {
                'hidden_size': h,
                'depth': d,
                'n_train': n,
                'lr': lr,
                'gamma': g
            }
            for h in hidden_sizes
            for d in depths
            for n in n_train_sizes
            for lr in learning_rates
            for g in gammas
        ]

        # Distribute work among ranks
        worker_combinations = [
            comb for i, comb in enumerate(all_combinations) if i % size == rank
        ]

        results = []
        
        # Process combinations
        for params in worker_combinations:
            print(f"Worker {rank} processing: {params}")

            # Sample training data
            key = jax_random.PRNGKey(hash(f"{exp_num}_{str(params)}") & 0xffffffff)
            indices = jax_random.permutation(key, master_size)[:params['n_train']]
            X_train = X_train_master[indices]
            y_train = y_train_master[indices]

            # If shuffling is enabled, shuffle the labels
            if shuffled:
                shuffle_key = jax_random.PRNGKey(
                    hash(f"shuffle_{params['n_train']}_{timestamp}_{rank}_{exp_num}") & 0xffffffff
                )
                y_train = shuffle_labels(y_train, shuffle_key)
                print(f"Labels shuffled with key: {shuffle_key}")
                params['shuffled'] = True
                params['shuffle_seed'] = int(shuffle_key[0])
            else:
                params['shuffled'] = False
                params['shuffle_seed'] = None

            # Save training subset
            subset_path = os.path.join(
                data_dir,
                f"train_subset_h{params['hidden_size']}_d{params['depth']}_"
                f"n{params['n_train']}_lr{params['lr']}_g{params['gamma']}_{timestamp}_rank{rank}.npz"
            )
            if not save_dataset(X_train, y_train, subset_path, rank):
                print(f"Failed to save training subset for {params}")
                continue

            # Create and initialize network
            init_fn, apply_fn, kernel_fn = create_mlp(
                d, params['hidden_size'], params['depth']
            )

            # Train model
            try:
                (final_params, best_test_error, training_history,
                 init_ntk_error, final_ntk_error,
                 init_ntk, final_ntk,
                 ntk_metrics) = train_and_evaluate(
                    init_fn, apply_fn, msp,
                    X_train, y_train,
                    X_test, y_test,
                    batch_size, epochs,
                    params['lr'], weight_decay,
                    params['gamma'], key
                )

                # Store results
                result = {
                    'experiment_run': exp_num,
                    'hidden_size': params['hidden_size'],
                    'depth': params['depth'],
                    'n_train': params['n_train'],
                    'learning_rate': params['lr'],
                    'gamma': params['gamma'],
                    'shuffled': params['shuffled'],
                    'shuffle_seed': params['shuffle_seed'],
                    'test_error': float(best_test_error),
                    'init_ntk_test_error': float(init_ntk_error),
                    'final_ntk_test_error': float(final_ntk_error),
                    'ntk_cka': ntk_metrics['cka'],
                    'ntk_frob_norm': ntk_metrics['frob_norm'],
                    'ntk_eig_dist': ntk_metrics['eig_dist'],
                    'ntk_vec_align': ntk_metrics['vec_align'],
                    'training_history': training_history,
                    'worker_rank': rank
                }
                results.append(result)

                # Save intermediate results
                save_results(results, results_dir, f'{timestamp}_rank{rank}')

            except Exception as e:
                print(f"Error processing {params}: {e}")
                continue

            print(f"Worker {rank} completed {params}")

        # Wait for all workers
        comm.Barrier()

        # Gather results for this experiment
        all_results = comm.gather(results, root=0)

        # Combine and save final results (rank 0)
        if rank == 0:
            combined_results = []
            for worker_results in all_results:
                combined_results.extend(worker_results)

            final_results_path = os.path.join(results_dir, 
                                            f'final_results_{timestamp}.json')
            with open(final_results_path, 'w') as f:
                json.dump(combined_results, f, indent=4)

            print(f"Experiment {exp_num} completed. Results saved.")

    if rank == 0:
        print("\nAll experiments completed successfully.")


if __name__ == "__main__":
    main()