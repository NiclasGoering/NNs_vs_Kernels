#!/usr/bin/env python3

"""
Example code using JAX + Neural Tangents (stax) to train an MLP on an MSP function.
We compute and save empirical NTKs at initial and final parameters,
as well as the test MSE from the initial/final NTK.
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


# Add these imports at the top of the file
import jax.numpy as jnp

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
    A_normalized = A / (norm_A + 1e-10)  # avoid division by zero
    B_normalized = B / (norm_B + 1e-10)
    return jnp.linalg.norm(A_normalized - B_normalized, ord='fro')

def compute_cka(K1, K2):
    """
    Compute Centered Kernel Alignment (CKA) between kernel matrices K1 and K2.
    CKA is invariant to orthogonal transformation and isotropic scaling.
    """
    # Center the kernel matrices
    K1_centered = center_matrix(K1)
    K2_centered = center_matrix(K2)
    
    # Compute HSIC
    hsic = jnp.sum(K1_centered * K2_centered)
    
    # Normalize
    norm1 = jnp.sqrt(jnp.sum(K1_centered * K1_centered))
    norm2 = jnp.sqrt(jnp.sum(K2_centered * K2_centered))
    
    return hsic / (norm1 * norm2)

def compute_ntk_metrics(initial_ntk, final_ntk):
    """
    Compute similarity metrics between initial and final NTK matrices.
    Returns CKA and normalized Frobenius norm.
    """
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


# -------------------
# Create Network
# -------------------
def create_mlp(d: int, hidden_size: int, depth: int):
    """Create MLP with ReLU hidden layers using nt_stax"""
    layers = []
    
    # Standard initialization
    W_std = 1.0 / (d ** 0.5)  # He initialization scaling
    
    for _ in range(depth):
        layers += [
            nt_stax.Dense(hidden_size, W_std=W_std),
            nt_stax.Relu()
        ]
    
    # Final layer
    layers.append(nt_stax.Dense(1, W_std=W_std))

    # Initialize the network and return all three functions
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

def compute_empirical_ntk(apply_fn, params, x1, x2=None):
    """Compute empirical NTK using neural_tangents"""
    ntk_fn = nt.empirical_ntk_fn(
        apply_fn,
        vmap_axes=0,
        trace_axes=(-1,)
    )
    
    if x2 is None:
        return ntk_fn(x1, None, params)
    return ntk_fn(x1, x2, params)


# -------------------
# Training
# -------------------
def train_and_evaluate(init_fn: Any, apply_fn: Any, msp: MSPFunction,
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
        initial_ntk = compute_empirical_ntk(apply_fn, params, X_train)
        initial_ntk_test = compute_empirical_ntk(apply_fn, params, X_test, X_train)
        
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
        final_ntk = compute_empirical_ntk(apply_fn, params, X_train)
        final_ntk_test = compute_empirical_ntk(apply_fn, params, X_test, X_train)
        
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

def save_model(params: Any, path: str):
    """Save model parameters as NPZ"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        flattened = {}
        for idx, leaf in enumerate(jax.tree_util.tree_leaves(params)):
            flattened[f'param_{idx}'] = np.array(leaf)
        np.savez(path, **flattened)
    except Exception as e:
        print(f"Error saving model: {e}")

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
    experiment_name = "jax_test_lr001"
    P = 8
    d = 30
    master_size = 30000
    hidden_sizes = [100,400]
    depths = [4]
    n_test = 1000
    batch_size = 64
    epochs = 5000
    learning_rates = [0.001]
    weight_decay = 1e-4
    n_train_sizes = [10, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 20000]

    # MSP sets
    msp_sets = [{7}, {2,7}, {0,2,7}, {5,7,4}, {1}, {0,4}, {3,7}, 
                {0,1,2,3,4,6,7}]
    msp = MSPFunction(P, msp_sets)

    # Directories
    base_path = "/mnt/users/goringn/NNs_vs_Kernels"
    results_dir = os.path.join(base_path, "stair_function/results", experiment_name)
    data_dir = results_dir
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Generate timestamp
    if rank == 0:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        hyperparams = {
            'P': P,
            'd': d,
            'hidden_sizes': hidden_sizes,
            'depths': depths,
            'n_test': n_test,
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rates': learning_rates,
            'weight_decay': weight_decay,
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
        key = jax_random.PRNGKey(hash(str(params)) & 0xffffffff)
        indices = jax_random.permutation(key, master_size)[:params['n_train']]
        X_train = X_train_master[indices]
        y_train = y_train_master[indices]

        # Save training subset
        train_data_path = os.path.join(
            data_dir,
            f"train_data_h{params['hidden_size']}_d{params['depth']}"
            f"_n{params['n_train']}_lr{params['lr']}_{timestamp}_rank{rank}.npz"
        )
        if not save_dataset(X_train, y_train, train_data_path, rank):
            print(f"Failed to save training data for {params}")
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
                key
            )

            # Save model parameters
            model_path = os.path.join(
                results_dir,
                f"model_h{params['hidden_size']}_d{params['depth']}"
                f"_n{params['n_train']}_lr{params['lr']}_{timestamp}_rank{rank}.npz"
            )
            save_model(final_params, model_path)

            # Save NTK matrices
            init_ntk_path = os.path.join(
                results_dir,
                f"init_ntk_h{params['hidden_size']}_d{params['depth']}"
                f"_n{params['n_train']}_lr{params['lr']}_{timestamp}_rank{rank}.npy"
            )
            final_ntk_path = os.path.join(
                results_dir,
                f"final_ntk_h{params['hidden_size']}_d{params['depth']}"
                f"_n{params['n_train']}_lr{params['lr']}_{timestamp}_rank{rank}.npy"
            )
            np.save(init_ntk_path, np.array(init_ntk))
            np.save(final_ntk_path, np.array(final_ntk))

            # Store results
            result = {
                'hidden_size': params['hidden_size'],
                'depth': params['depth'],
                'n_train': params['n_train'],
                'learning_rate': params['lr'],
                'test_error': float(best_test_error),
                'init_ntk_test_error': float(init_ntk_error),
                'final_ntk_test_error': float(final_ntk_error),
                'ntk_cka': ntk_metrics['cka'],
                'ntk_norm_frob': ntk_metrics['norm_frob'],
                'training_history': training_history,
                'worker_rank': rank,
                'model_path': os.path.basename(model_path),
                'init_ntk_path': os.path.basename(init_ntk_path),
                'final_ntk_path': os.path.basename(final_ntk_path)
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

    # Gather results
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

        print("All workers completed. Results combined and saved.")

if __name__ == "__main__":
    main()