#!/usr/bin/env python3

"""
Example code using JAX + Neural Tangents (stax) to train an MLP on an MSP function,
but computing and saving the NNGP (instead of the NTK) and its layer-wise spectra 
at initial and final parameters.
"""

import os
import json
import numpy as np
from datetime import datetime
from functools import partial
from typing import List, Set, Any, Dict

# MPI
from mpi4py import MPI

# JAX / neural-tangents / optax
import jax
import jax.numpy as jnp
from jax import random as jax_random
import optax
import neural_tangents as nt
from neural_tangents import stax as nt_stax

import inspect
import neural_tangents as nt

print("Neural Tangents version:", nt.__version__)
print("Neural Tangents file:", nt.__file__)

# Let's see the function signature or source code:
print("Signature of empirical_nngp_fn:",
      inspect.signature(nt.empirical_nngp_fn))

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
    def __init__(self, P: int, sets: List[set]):
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


# -------------------
# NNGP Computation
# -------------------
def compute_empirical_nngp_batched(apply_fn, params, x1, x2=None, batch_size=512):
    """
    Compute the empirical NNGP (single (n1 x n2) matrix) in batches.
    If x2 is None, do the train–train matrix (symmetry).
    """
    # Create the empirical NNGP function
    nngp_fn = nt.empirical_nngp_fn(
        f=apply_fn,
        trace_axes=(-1,),  # trace over the output dimension
        diagonal_axes=()    # no diagonal axes
    )

    if x2 is None:
        x2 = x1

    n1 = x1.shape[0]
    n2 = x2.shape[0]
    K = jnp.zeros((n1, n2))

    i = 0
    while i < n1:
        i_end = min(i + batch_size, n1)
        x1_batch = x1[i:i_end]

        # If we're doing train–train, compute only upper-tri
        j_start = i if x2 is x1 else 0

        j = j_start
        while j < n2:
            j_end = min(j + batch_size, n2)
            x2_batch = x2[j:j_end]

            try:
                # shape (i_batch_size, j_batch_size)
                block = nngp_fn(x1_batch, x2_batch, params)
                # Ensure block is 2D
                if block.ndim == 0:
                    block = block * jnp.ones((i_end - i, j_end - j))
                elif block.ndim == 1:
                    block = block.reshape(-1, 1) * jnp.ones((1, j_end - j))

                K = K.at[i:i_end, j:j_end].set(block)

                # Mirror if symmetric
                if (x2 is x1) and (i != j):
                    K = K.at[j:j_end, i:i_end].set(block.T)

            except Exception as e:
                print(f"Error in batch computation: {e}")
                print(f"Shapes: x1_batch: {x1_batch.shape}, x2_batch: {x2_batch.shape}")
                raise

            j += batch_size
        i += batch_size

    return K


def compute_per_layer_nngp_batched(apply_fn, params, x1, x2=None, batch_size=512):
    """
    Compute a dictionary of per-layer empirical NNGP blocks (one matrix per layer),
    in batches, to avoid memory blow-ups.
    """
    # Create the empirical NNGP function
    nngp_fn = nt.empirical_nngp_fn(
        f=apply_fn,
        trace_axes=(-1,),  # trace over the output dimension
        diagonal_axes=()    # no diagonal axes
    )

    if x2 is None:
        x2 = x1

    n1 = x1.shape[0]
    n2 = x2.shape[0]

    # Dictionary: "layer_i" -> (n1 x n2) array
    nngps = {}

    i = 0
    while i < n1:
        i_end = min(i + batch_size, n1)
        x1_batch = x1[i:i_end]

        # If x1 == x2, only compute upper triangular
        j_start = i if (x2 is x1) else 0

        j = j_start
        while j < n2:
            j_end = min(j + batch_size, n2)
            x2_batch = x2[j:j_end]

            try:
                # nngp_batch is a tuple or list (one entry per layer)
                nngp_batch = nngp_fn(x1_batch, x2_batch, params)

                # Handle case where nngp_batch is a single value
                if not isinstance(nngp_batch, (tuple, list)):
                    nngp_batch = [nngp_batch]

                # Initialize dictionary with zeros if first time
                if not nngps:
                    for layer_idx, _ in enumerate(nngp_batch):
                        nngps[f"layer_{layer_idx}"] = jnp.zeros((n1, n2))

                # Place each block
                for layer_idx, layer_block in enumerate(nngp_batch):
                    # Ensure block is 2D
                    if layer_block.ndim == 0:
                        layer_block = layer_block * jnp.ones((i_end - i, j_end - j))
                    elif layer_block.ndim == 1:
                        layer_block = layer_block.reshape(-1, 1) * jnp.ones((1, j_end - j))

                    nngps[f"layer_{layer_idx}"] = nngps[f"layer_{layer_idx}"].at[i:i_end, j:j_end].set(layer_block)
                    
                    # Mirror if x1==x2 and off diagonal
                    if (x2 is x1) and (i != j):
                        nngps[f"layer_{layer_idx}"] = nngps[f"layer_{layer_idx}"].at[j:j_end, i:i_end].set(layer_block.T)

            except Exception as e:
                print(f"Error in batch computation: {e}")
                print(f"Shapes: x1_batch: {x1_batch.shape}, x2_batch: {x2_batch.shape}")
                raise

            j += batch_size
        i += batch_size

    return nngps


# -------------------
# Spectrum + Similarity
# -------------------
def compute_spectrum(kernel_mat, max_eigvals=None):
    """Compute eigenvalue spectrum of a kernel matrix with optional top-eigval limit."""
    n = kernel_mat.shape[0]
    eps = 1e-12

    if max_eigvals and n > max_eigvals:
        # Move to CPU for large matrices
        kernel_cpu = jax.device_put(kernel_mat, device=jax.devices('cpu')[0])
        try:
            # For very large n, you might want partial eigensolvers (not shown here)
            # but we'll do a full eigen-decomposition for demonstration
            eigenvals = jnp.linalg.eigvalsh(kernel_cpu)[-max_eigvals:]
        except Exception as e:
            print(f"Error computing spectrum on CPU: {e}")
            return None
    else:
        try:
            eigenvals = jnp.linalg.eigvalsh(kernel_mat)
        except Exception as e:
            print(f"Error computing spectrum on GPU: {e}")
            return None

    # Sort in descending order
    eigenvals = jnp.sort(eigenvals)[::-1]
    return eigenvals

def compute_layer_spectra(layer_kernels: Dict[str, jnp.ndarray], max_eigvals=2000):
    """Compute spectra for each layer's kernel matrix."""
    return {
        name: compute_spectrum(kmat, max_eigvals=max_eigvals)
        for name, kmat in layer_kernels.items()
    }

def save_spectra(spectra: Dict[str, jnp.ndarray], path: str, prefix: str) -> bool:
    """Save eigenvalue spectra to NPZ file."""
    try:
        data_dict = {
            f"{prefix}_{name}": spectrum
            for name, spectrum in spectra.items()
            if spectrum is not None
        }
        np.savez(path, **data_dict)
        print(f"Saved spectra to {path}")
        return True
    except Exception as e:
        print(f"Error saving spectra: {e}")
        return False

def compute_frob_norm_batched(K1, K2, batch_size=1024):
    """
    Compute an element-wise relative difference:
      mean(|K1[i,j] - K2[i,j]| / max(|K1[i,j]|, |K2[i,j]|)).
    """
    eps = 1e-12
    n = K1.shape[0]
    total_diff = 0.0
    count = 0

    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        K1_batch = K1[i:end]
        K2_batch = K2[i:end]

        diff = jnp.abs(K1_batch - K2_batch)
        max_vals = jnp.maximum(jnp.abs(K1_batch), jnp.abs(K2_batch)) + eps
        rel_diff = diff / max_vals

        total_diff += jnp.sum(rel_diff)
        count += rel_diff.size

    return total_diff / count

def compute_cka_batched(K1, K2, batch_size=1000):
    """
    Compute CKA between two kernel matrices in a memory-efficient manner.
    This does a linear-time CKA by centering and then computing HSIC / norms.
    """
    n = K1.shape[0]

    # Accumulators
    hsic_sum = 0.0
    norm1_sum = 0.0
    norm2_sum = 0.0

    # Row means
    K1_mean = jnp.mean(K1)
    K2_mean = jnp.mean(K2)
    K1_row_means = jnp.mean(K1, axis=1)
    K2_row_means = jnp.mean(K2, axis=1)

    for i in range(0, n, batch_size):
        end_idx = min(i + batch_size, n)
        K1_batch = K1[i:end_idx]
        K2_batch = K2[i:end_idx]

        # Center them
        K1_centered = (K1_batch
                       - K1_row_means[i:end_idx, None]
                       - K1_row_means[None, :]
                       + K1_mean)
        K2_centered = (K2_batch
                       - K2_row_means[i:end_idx, None]
                       - K2_row_means[None, :]
                       + K2_mean)

        hsic_sum += jnp.sum(K1_centered * K2_centered)
        norm1_sum += jnp.sum(K1_centered * K1_centered)
        norm2_sum += jnp.sum(K2_centered * K2_centered)

    return hsic_sum / (jnp.sqrt(norm1_sum) * jnp.sqrt(norm2_sum))

def compute_kernel_metrics_batched(initial_kernel, final_kernel, batch_size=1000, k_top=10):
    """
    Compute some similarity metrics (CKA, relative Frobenius difference,
    eigenvalue distance, top eigenvector alignment) between two kernel matrices.
    """
    if initial_kernel is None or final_kernel is None:
        return {
            'cka': float('nan'),
            'frob_norm': float('nan'),
            'eig_dist': float('nan'),
            'vec_align': float('nan')
        }

    try:
        # CKA
        cka = compute_cka_batched(initial_kernel, final_kernel, batch_size=batch_size)
        # Frobenius
        frob_norm = compute_frob_norm_batched(initial_kernel, final_kernel, batch_size=batch_size)

        # Eigenvalue distribution
        init_eigs = jnp.linalg.eigvalsh(initial_kernel)
        final_eigs = jnp.linalg.eigvalsh(final_kernel)

        init_eigs = jnp.sort(init_eigs)[::-1] / jnp.sum(jnp.abs(init_eigs))
        final_eigs = jnp.sort(final_eigs)[::-1] / jnp.sum(jnp.abs(final_eigs))
        eig_dist = jnp.mean(jnp.abs(init_eigs - final_eigs))

        # Eigenvector alignment (top k_top)
        _, init_vecs = jnp.linalg.eigh(initial_kernel)
        _, final_vecs = jnp.linalg.eigh(final_kernel)

        init_space = init_vecs[:, -k_top:]
        final_space = final_vecs[:, -k_top:]

        cross = init_space.T @ final_space
        vec_align = jnp.mean(jnp.linalg.svd(cross, compute_uv=False))

        return {
            'cka': float(cka),
            'frob_norm': float(frob_norm),
            'eig_dist': float(eig_dist),
            'vec_align': float(vec_align)
        }

    except Exception as e:
        print(f"Error computing kernel metrics: {e}")
        return {
            'cka': float('nan'),
            'frob_norm': float('nan'),
            'eig_dist': float('nan'),
            'vec_align': float('nan')
        }


# -------------------
# MLP Creation
# -------------------
def create_mlp(d: int, hidden_size: int, depth: int):
    """Create MLP with ReLU hidden layers using nt_stax."""
    layers = []
    W_std = 1.0 / (d ** 0.5)  # He initialization-like scaling

    for _ in range(depth):
        layers += [
            nt_stax.Dense(hidden_size, W_std=W_std),
            nt_stax.Relu()
        ]
    # Final layer
    layers.append(nt_stax.Dense(1, W_std=W_std))

    init_fn, apply_fn, kernel_fn = nt_stax.serial(*layers)
    return init_fn, apply_fn, kernel_fn


# -------------------
# Data Generation
# -------------------
def generate_master_dataset(P: int, d: int, master_size: int, n_test: int, 
                            msp: MSPFunction, seed: int = 42):
    """
    Generate a "master" training set plus a test set of ±1 Rademacher inputs.
    """
    np.random.seed(seed)
    X_train_master_np = 2.0 * np.random.binomial(1, 0.5, size=(master_size, d)) - 1.0
    X_test_np = 2.0 * np.random.binomial(1, 0.5, size=(n_test, d)) - 1.0

    X_train_master = jnp.array(X_train_master_np, dtype=jnp.float32)
    X_test = jnp.array(X_test_np, dtype=jnp.float32)
    y_train_master = msp.evaluate(X_train_master)
    y_test = msp.evaluate(X_test)
    return X_train_master, y_train_master, X_test, y_test


def shuffle_labels(y: jnp.ndarray, key: Any) -> jnp.ndarray:
    """Shuffle labels randomly using a JAX random key."""
    perm = jax_random.permutation(key, len(y))
    return y[perm]


def compute_mse(params: Any, apply_fn: Any, inputs: jnp.ndarray, 
                targets: jnp.ndarray, gamma: float = 1.0) -> float:
    """Compute MSE loss with an optional gamma scaling on predictions."""
    predictions = apply_fn(params, inputs).squeeze(-1)
    scaled_predictions = predictions / gamma
    return jnp.mean((scaled_predictions - targets) ** 2)


# -------------------
# Training + Evaluate
# -------------------
def train_and_evaluate(init_fn: Any, apply_fn: Any, msp: MSPFunction,
                       X_train: jnp.ndarray, y_train: jnp.ndarray, 
                       X_test: jnp.ndarray, y_test: jnp.ndarray,
                       batch_size: int, epochs: int, lr: float, 
                       weight_decay: float, gamma: float, key: Any,
                       save_dir: str, experiment_id: str):
    """
    Train the network (same as before) but compute + save the NNGP
    at initialization and at the end, plus its spectra.
    """

    # 1) Initialize model
    output_shape, params = init_fn(key, (-1,) + X_train.shape[1:])

    # 2) Compute initial NNGP, per-layer NNGP, and spectra
    print("Computing initial NNGP and spectra...")
    try:
        initial_nngp = compute_empirical_nngp_batched(apply_fn, params, X_train)
        initial_nngp_test = compute_empirical_nngp_batched(apply_fn, params, X_test, X_train)

        # Per-layer
        initial_layer_nngps = compute_per_layer_nngp_batched(apply_fn, params, X_train)
        initial_layer_spectra = compute_layer_spectra(initial_layer_nngps)

        # Save
        spectra_path = os.path.join(save_dir, f'nngp_spectra_{experiment_id}_initial.npz')
        save_spectra(initial_layer_spectra, spectra_path, 'initial')

        # Kernel regression with NNGP
        reg = 1e-6 * jnp.trace(initial_nngp) / len(X_train)
        reg_nngp = initial_nngp + reg * jnp.eye(len(X_train))
        alpha = jnp.linalg.solve(reg_nngp, y_train)
        y_pred_nngp = initial_nngp_test @ alpha
        initial_kernel_error = jnp.mean((y_pred_nngp - y_test) ** 2)
        print(f"Initial NNGP test error (kernel regression): {initial_kernel_error:.6f}")

    except Exception as e:
        print(f"Warning: Initial NNGP computation failed: {e}")
        initial_kernel_error = float('nan')
        initial_nngp = None

    # 3) Setup optimizer
    steps_per_epoch = max(1, len(X_train) // batch_size)
    total_steps = epochs * steps_per_epoch

    scheduler = optax.cosine_decay_schedule(
        init_value=lr,
        decay_steps=total_steps,
        alpha=0.0
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=scheduler, weight_decay=weight_decay)
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, batch_x, batch_y):
        def loss_fn(p):
            return compute_mse(p, apply_fn, batch_x, batch_y, gamma)
        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_val

    # 4) Training loop
    best_test_error = float('inf')
    training_history = {
        'train_errors': [],
        'test_errors': [],
        'epochs': []
    }
    print("Starting training...")
    for epoch in range(epochs):
        # Shuffle each epoch
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, len(X_train))
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]

        epoch_losses = []
        for i in range(0, len(X_train), batch_size):
            batch_x = X_train_shuffled[i:i+batch_size]
            batch_y = y_train_shuffled[i:i+batch_size]
            params, opt_state, loss = train_step(params, opt_state, batch_x, batch_y)
            epoch_losses.append(loss)

        # Periodic logging
        if epoch % 100 == 0 or epoch == epochs - 1:
            train_error = jnp.mean(jnp.array(epoch_losses))
            test_pred = apply_fn(params, X_test).squeeze(-1) / gamma
            test_error = jnp.mean((test_pred - y_test) ** 2)
            best_test_error = min(best_test_error, test_error)

            training_history['train_errors'].append(float(train_error))
            training_history['test_errors'].append(float(test_error))
            training_history['epochs'].append(epoch)

            print(f"Epoch {epoch}")
            print(f"Train Error: {train_error:.6f}")
            print(f"Test Error: {test_error:.6f}")
            print(f"Best Test Error: {best_test_error:.6f}")

    # 5) Compute final NNGP, spectra
        print("Computing initial NNGP and spectra...")
    try:
        initial_nngp = compute_empirical_nngp_batched(apply_fn, params, X_train)
        initial_nngp_test = compute_empirical_nngp_batched(apply_fn, params, X_test, X_train)

        # Per-layer
        initial_layer_nngps = compute_per_layer_nngp_batched(apply_fn, params, X_train)
        initial_layer_spectra = compute_layer_spectra(initial_layer_nngps)

        # Save
        spectra_path = os.path.join(save_dir, f'nngp_spectra_{experiment_id}_initial.npz')
        save_spectra(initial_layer_spectra, spectra_path, 'initial')

        # Kernel regression with NNGP
        reg = 1e-6 * jnp.trace(initial_nngp) / len(X_train)
        reg_nngp = initial_nngp + reg * jnp.eye(len(X_train))
        alpha = jnp.linalg.solve(reg_nngp, y_train)
        y_pred_nngp = initial_nngp_test @ alpha
        initial_kernel_error = jnp.mean((y_pred_nngp - y_test) ** 2)
        print(f"Initial NNGP test error (kernel regression): {initial_kernel_error:.6f}")

    except Exception as e:
        print(f"Warning: Initial NNGP computation failed: {e}")
        initial_kernel_error = float('nan')
        initial_nngp = None

    # 6) Similarity metrics: initial vs. final NNGP
    print("Computing NNGP similarity metrics...")
    nngp_metrics = compute_kernel_metrics_batched(initial_nngp, final_nngp, batch_size=1024)
    print(f"CKA between initial and final NNGP: {nngp_metrics['cka']:.6f}")
    print(f"Relative Frobenius diff: {nngp_metrics['frob_norm']:.6f}")
    print(f"Eigenvalue distribution distance: {nngp_metrics['eig_dist']:.6f}")
    print(f"Top eigenvector alignment: {nngp_metrics['vec_align']:.6f}")

    return (params, best_test_error, training_history,
            initial_kernel_error, final_kernel_error,
            initial_nngp, final_nngp,
            nngp_metrics)


# -------------------
# Save / Load Helpers
# -------------------
def save_results(results: List[dict], results_dir: str, filename_suffix: str):
    """Save results to JSON."""
    try:
        path = os.path.join(results_dir, f'results_{filename_suffix}.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print(f"Error saving results: {e}")


def save_dataset(X: jnp.ndarray, y: jnp.ndarray, path: str, rank: int):
    """Save dataset as NPZ."""
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
    # MPI init
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Example parameters
    base_experiment_name = "jax_msp_gpu_nngp_example"
    P = 8
    d = 30
    master_size = 35000
    hidden_sizes = [400, 1000]   # example
    depths = [4]
    n_test = 5000
    batch_size = 64
    epochs = 3000
    learning_rates = [0.005]
    weight_decay = 1e-4
    gammas = [1]
    shuffled = False
    n_train_sizes = [1000, 5000, 10000, 20000]  # example
    num_experiments = 1  # example

    # MSP sets
    msp_sets = [{7}, {2,7}, {0,2,7}, {5,7,4}, {1}, {0,4}, {3,7},
                {0,1,2,3,4,6,7}]
    msp = MSPFunction(P, msp_sets)

    for exp_num in range(1, num_experiments + 1):
        if rank == 0:
            print(f"\n{'='*50}")
            print(f"Starting experiment run {exp_num}/{num_experiments}")
            print(f"{'='*50}\n")

        experiment_name = f"{base_experiment_name}_{exp_num}"
        results_dir = os.path.join("/mnt/users/goringn/NNs_vs_Kernels/NTK/results", experiment_name)
        data_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

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

        # Master generates dataset
        if rank == 0:
            print("Master generating datasets...")
            key = jax_random.PRNGKey(exp_num)
            X_train_master, y_train_master, X_test, y_test = generate_master_dataset(
                P, d, master_size, n_test, msp
            )
            print("Data generation complete")
            print(f"Master set shape: {X_train_master.shape}")
            print(f"Test set shape: {X_test.shape}")

            master_data_path = os.path.join(data_dir, f'master_data_{timestamp}.npz')
            test_data_path = os.path.join(data_dir, f'test_data_{timestamp}.npz')

            if not save_dataset(X_train_master, y_train_master, master_data_path, rank):
                raise RuntimeError("Failed to save master dataset")
            if not save_dataset(X_test, y_test, test_data_path, rank):
                raise RuntimeError("Failed to save test dataset")

            print("Master saved all datasets successfully")
        else:
            X_train_master = None
            y_train_master = None
            X_test = None
            y_test = None

        # Broadcast datasets to all ranks
        X_train_master = comm.bcast(X_train_master, root=0)
        y_train_master = comm.bcast(y_train_master, root=0)
        X_test = comm.bcast(X_test, root=0)
        y_test = comm.bcast(y_test, root=0)

        # Parameter combos
        all_combinations = [
            {
                'hidden_size': h,
                'depth': dd,
                'n_train': n,
                'lr': lr,
                'gamma': g
            }
            for h in hidden_sizes
            for dd in depths
            for n in n_train_sizes
            for lr in learning_rates
            for g in gammas
        ]

        # Distribute among workers
        worker_combinations = [
            comb for i, comb in enumerate(all_combinations) if i % size == rank
        ]

        results = []

        for params_dict in worker_combinations:
            print(f"Worker {rank} processing: {params_dict}")

            # Subset training data
            key = jax_random.PRNGKey(
                hash(f"{exp_num}_{str(params_dict)}") & 0xffffffff
            )
            indices = jax_random.permutation(key, master_size)[:params_dict['n_train']]
            X_train = X_train_master[indices]
            y_train = y_train_master[indices]

            # Optionally shuffle
            if shuffled:
                shuffle_key = jax_random.PRNGKey(
                    hash(f"shuffle_{params_dict['n_train']}_{timestamp}_{rank}_{exp_num}") & 0xffffffff
                )
                y_train = shuffle_labels(y_train, shuffle_key)
                params_dict['shuffled'] = True
                params_dict['shuffle_seed'] = int(shuffle_key[0])
            else:
                params_dict['shuffled'] = False
                params_dict['shuffle_seed'] = None

            # Save subset
            subset_path = os.path.join(
                data_dir,
                f"train_subset_h{params_dict['hidden_size']}_d{params_dict['depth']}_"
                f"n{params_dict['n_train']}_lr{params_dict['lr']}_g{params_dict['gamma']}_"
                f"{timestamp}_rank{rank}.npz"
            )
            save_dataset(X_train, y_train, subset_path, rank)

            # Create network
            init_fn, apply_fn, kernel_fn = create_mlp(
                d, params_dict['hidden_size'], params_dict['depth']
            )

            # Unique ID
            experiment_id = (f"h{params_dict['hidden_size']}_d{params_dict['depth']}_"
                             f"n{params_dict['n_train']}_lr{params_dict['lr']}_"
                             f"g{params_dict['gamma']}_{timestamp}_rank{rank}")

            try:
                (final_params, best_test_error, training_history,
                 init_kernel_error, final_kernel_error,
                 init_nngp, final_nngp,
                 nngp_metrics) = train_and_evaluate(
                    init_fn, apply_fn, msp,
                    X_train, y_train,
                    X_test, y_test,
                    batch_size, epochs,
                    params_dict['lr'], weight_decay,
                    params_dict['gamma'], key,
                    results_dir, experiment_id
                )

                # Store
                result = {
                    'experiment_run': exp_num,
                    'hidden_size': params_dict['hidden_size'],
                    'depth': params_dict['depth'],
                    'n_train': params_dict['n_train'],
                    'learning_rate': params_dict['lr'],
                    'gamma': params_dict['gamma'],
                    'shuffled': params_dict['shuffled'],
                    'shuffle_seed': params_dict['shuffle_seed'],
                    'test_error': float(best_test_error),
                    'init_nngp_test_error': float(init_kernel_error),
                    'final_nngp_test_error': float(final_kernel_error),
                    'nngp_cka': nngp_metrics['cka'],
                    'nngp_frob_norm': nngp_metrics['frob_norm'],
                    'nngp_eig_dist': nngp_metrics['eig_dist'],
                    'nngp_vec_align': nngp_metrics['vec_align'],
                    'training_history': training_history,
                    'worker_rank': rank
                }
                results.append(result)

                # Intermediate save
                save_results(results, results_dir, f'{timestamp}_rank{rank}')

            except Exception as e:
                print(f"Error processing {params_dict}: {e}")
                continue

            print(f"Worker {rank} completed {params_dict}")

        comm.Barrier()
        all_results = comm.gather(results, root=0)

        if rank == 0:
            combined_results = []
            for worker_results in all_results:
                combined_results.extend(worker_results)
            final_results_path = os.path.join(results_dir, f'final_results_{timestamp}.json')
            with open(final_results_path, 'w') as f:
                json.dump(combined_results, f, indent=4)
            print(f"Experiment {exp_num} completed. Results saved.")

    if rank == 0:
        print("\nAll experiments completed successfully.")


if __name__ == "__main__":
    main()
