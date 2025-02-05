#!/usr/bin/env python3

"""
Example code using JAX + Neural Tangents (stax) to train an MLP on low-dimensional polynomials.
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
from itertools import product

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
# Polynomial Generation Functions
# -------------------
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
    Generate synthetic data without normalization
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
    
    # Add symmetric noise
    noise = noise_std * np.random.choice([-1, 1], size=n_samples)
    y = y + noise
    
    return X, y, U, coeff_vec

def generate_fixed_test_data(ambient_dim, latent_dim, degree, n_test=10000, noise_std=0.1, U=None, coeff_vec=None, random_state=None):
    """
    Generate test set using same U matrix and coefficients as training data
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    X_test = np.random.randn(n_test, ambient_dim)
    
    # Project onto same latent space
    X_test_latent = X_test @ U
    
    # Initialize output
    y_test = np.zeros(n_test)
    
    # Generate all polynomial terms
    terms = generate_polynomials(latent_dim, degree)
    
    # Use same coefficients as training data
    coeff_idx = 0
    for term in terms:
        if sum(term) > 0:  # Skip constant term
            term_value = np.ones(n_test)
            for dim, power in enumerate(term):
                if power > 0:
                    term_value *= X_test_latent[:, dim] ** power
            y_test += coeff_vec[coeff_idx] * term_value
            coeff_idx += 1
    
    # Add symmetric noise
    noise = noise_std * np.random.choice([-1, 1], size=n_test)
    y_test = y_test + noise
    
    return X_test, y_test

def center_matrix(K):
    """Center a kernel matrix K"""
    n = K.shape[0]
    unit = jnp.ones([n, n])
    I = jnp.eye(n)
    H = I - unit / n
    return H @ K @ H

def normalized_frobenius_norm(A, B):
    norm_A = jnp.linalg.norm(A, ord='fro')
    norm_B = jnp.linalg.norm(B, ord='fro')
    A_normalized = A / (norm_A + 1e-10)  
    B_normalized = B / (norm_B + 1e-10)
    return jnp.linalg.norm(A_normalized - B_normalized, ord='fro')

def compute_cka(K1, K2):
    """Compute Centered Kernel Alignment (CKA)"""
    K1_centered = center_matrix(K1)
    K2_centered = center_matrix(K2)
    hsic = jnp.sum(K1_centered * K2_centered)
    norm1 = jnp.sqrt(jnp.sum(K1_centered * K1_centered))
    norm2 = jnp.sqrt(jnp.sum(K2_centered * K2_centered))
    return hsic / (norm1 * norm2)

def compute_ntk_metrics(initial_ntk, final_ntk):
    """Compute similarity metrics between initial and final NTK matrices."""
    if initial_ntk is None or final_ntk is None:
        return {
            'cka': float('nan'),
            'norm_frob': float('nan')
        }
    
    try:
        cka = float(compute_cka(initial_ntk, final_ntk))
        norm_frob = float(normalized_frobenius_norm(initial_ntk, final_ntk))
        
        return {
            'cka': cka,
            'norm_frob': norm_frob
        }
    except Exception as e:
        print(f"Error computing NTK metrics: {e}")
        return {
            'cka': float('nan'),
            'norm_frob': float('nan')
        }

# -------------------
# Data Generation
# -------------------
def generate_master_dataset(ambient_dim, latent_dim, degree, master_size, n_test, 
                          noise_std=0.1, seed=42):
    """Generate master training set + test set with a fixed seed"""
    np.random.seed(seed)
    
    # Generate training data
    X_train_master, y_train_master, U, coeff_vec = generate_latent_poly_data(
        n_samples=master_size,
        ambient_dim=ambient_dim,
        latent_dim=latent_dim,
        degree=degree,
        noise_std=noise_std,
        random_state=seed
    )
    
    # Generate test data using same polynomial
    X_test, y_test = generate_fixed_test_data(
        ambient_dim=ambient_dim,
        latent_dim=latent_dim,
        degree=degree,
        n_test=n_test,
        noise_std=noise_std,
        U=U,
        coeff_vec=coeff_vec,
        random_state=seed
    )
    
    # Convert to JAX arrays
    X_train_master = jnp.array(X_train_master, dtype=jnp.float32)
    y_train_master = jnp.array(y_train_master, dtype=jnp.float32)
    X_test = jnp.array(X_test, dtype=jnp.float32)
    y_test = jnp.array(y_test, dtype=jnp.float32)
    
    return X_train_master, y_train_master, X_test, y_test

def generate_gaussian_noise_labels(y: jnp.ndarray, key: Any, global_mean: float, global_std: float) -> jnp.ndarray:
    """Generate Gaussian noise with same mean and variance as normalized labels"""
    noise = jax_random.normal(key, y.shape)
    return global_mean + global_std * noise

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
                targets: jnp.ndarray) -> float:
    """Compute MSE loss"""
    predictions = apply_fn(params, inputs).squeeze(-1)
    return jnp.mean((predictions - targets) ** 2)

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
def train_and_evaluate(init_fn: Any, apply_fn: Any,
                      X_train: jnp.ndarray, y_train: jnp.ndarray, 
                      X_test: jnp.ndarray, y_test: jnp.ndarray,
                      batch_size: int, epochs: int, lr: float, 
                      weight_decay: float, key: Any):
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
            return compute_mse(p, apply_fn, batch_x, batch_y)
        
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
            test_pred = apply_fn(params, X_test).squeeze(-1)
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
    ntk_metrics = compute_ntk_metrics(initial_ntk, final_ntk)
    print(f"CKA between initial and final NTK: {ntk_metrics['cka']:.6f}")
    print(f"Normalized Frobenius norm between initial and final NTK: {ntk_metrics['norm_frob']:.6f}")
    
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
    base_experiment_name = "jax_polynn_h1000_lr0005_experiment"
    ambient_dim = 20
    latent_dim = 3
    degree = 5
    noise_std = 0.0
    master_size = 50000
    hidden_sizes = [400, 1000]
    hidden_sizes.reverse()
    depths = [4]
    n_test = 10000
    batch_size = 64
    epochs = 5000
    learning_rates = [0.1,0.01,0.05,0.001,0.005,0.0001,0.0005,0.00001]
    weight_decay = 1e-4
    shuffled = False
    n_train_sizes = [20000]
    n_train_sizes.reverse()
    num_experiments = 2  # Number of experimental runs

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
                'ambient_dim': ambient_dim,
                'latent_dim': latent_dim,
                'degree': degree,
                'noise_std': noise_std,
                'hidden_sizes': hidden_sizes,
                'depths': depths,
                'n_test': n_test,
                'batch_size': batch_size,
                'epochs': epochs,
                'learning_rates': learning_rates,
                'weight_decay': weight_decay,
                'shuffled': shuffled,
                'n_train_sizes': n_train_sizes,
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
            # Use different seed for each experiment
            key = jax_random.PRNGKey(exp_num)
            X_train_master, y_train_master, X_test, y_test = generate_master_dataset(
                ambient_dim, latent_dim, degree, master_size, n_test, noise_std, exp_num)
            
            # Calculate global normalization statistics
            y_mean = float(jnp.mean(y_train_master))
            y_std = float(jnp.std(y_train_master))
            
            # Normalize the data
            y_train_master = (y_train_master - y_mean) / y_std
            y_test = (y_test - y_mean) / y_std
            
            print("Data generation complete")
            print(f"Master set shape: {X_train_master.shape}")
            print(f"Test set shape: {X_test.shape}")
            print(f"Global stats - mean: {y_mean:.6f}, std: {y_std:.6f}")

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
            y_mean = None
            y_std = None

        # Broadcast datasets and normalization stats to all workers
        X_train_master = comm.bcast(X_train_master, root=0)
        y_train_master = comm.bcast(y_train_master, root=0)
        X_test = comm.bcast(X_test, root=0)
        y_test = comm.bcast(y_test, root=0)
        y_mean = comm.bcast(y_mean, root=0)
        y_std = comm.bcast(y_std, root=0)

        # Generate parameter combinations
        all_combinations = [
            {
                'hidden_size': h,
                'depth': d,
                'n_train': n,
                'lr': lr
            }
            for h in hidden_sizes
            for d in depths
            for n in n_train_sizes
            for lr in learning_rates
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

            # If shuffling is enabled, replace labels with Gaussian noise
            if shuffled:
                shuffle_key = jax_random.PRNGKey(
                    hash(f"shuffle_{params['n_train']}_{timestamp}_{rank}_{exp_num}") & 0xffffffff
                )
                # Calculate mean and variance of the actual labels before replacing
                local_mean = jnp.mean(y_train)
                local_std = jnp.std(y_train)
                
                # Generate Gaussian noise with matched statistics
                y_train = local_mean + local_std * jax_random.normal(shuffle_key, y_train.shape)
                print(f"Labels replaced with Gaussian noise (mean={local_mean:.6f}, std={local_std:.6f}) using key: {shuffle_key}")
                params['shuffled'] = True
                params['shuffle_seed'] = int(shuffle_key[0])
                params['noise_mean'] = float(local_mean)
                params['noise_std'] = float(local_std)
            else:
                params['shuffled'] = False
                params['shuffle_seed'] = None
                params['noise_mean'] = None
                params['noise_std'] = None

            # Save training subset
            subset_path = os.path.join(
                data_dir,
                f"train_subset_h{params['hidden_size']}_d{params['depth']}_"
                f"n{params['n_train']}_lr{params['lr']}_{timestamp}_rank{rank}.npz"
            )
            if not save_dataset(X_train, y_train, subset_path, rank):
                print(f"Failed to save training subset for {params}")
                continue

            # Create and initialize network
            init_fn, apply_fn, kernel_fn = create_mlp(
                ambient_dim, params['hidden_size'], params['depth']
            )

            # Train model
            try:
                (final_params, best_test_error, training_history,
                 init_ntk_error, final_ntk_error,
                 init_ntk, final_ntk,
                 ntk_metrics) = train_and_evaluate(
                    init_fn, apply_fn,
                    X_train, y_train,
                    X_test, y_test,
                    batch_size, epochs,
                    params['lr'], weight_decay,
                    key
                )

                # Store results
                result = {
                    'experiment_run': exp_num,
                    'hidden_size': params['hidden_size'],
                    'depth': params['depth'],
                    'n_train': params['n_train'],
                    'learning_rate': params['lr'],
                    'shuffled': params['shuffled'],
                    'shuffle_seed': params['shuffle_seed'],
                    'test_error': float(best_test_error),
                    'init_ntk_test_error': float(init_ntk_error),
                    'final_ntk_test_error': float(final_ntk_error),
                    'ntk_cka': ntk_metrics['cka'],
                    'ntk_norm_frob': ntk_metrics['norm_frob'],
                    'training_history': training_history,
                    'worker_rank': rank,
                    'normalization_mean': float(y_mean),
                    'normalization_std': float(y_std)
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