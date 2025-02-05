#!/usr/bin/env python3

"""
Example code using JAX + Neural Tangents (stax) to train a LeNet on MNIST,
computing empirical NTKs at initial/final parameters, plus the infinite-width NTK.
We record *accuracies* from each kernel and measure CKA/Frobenius norms.

MPI is used for multi-worker distribution of different hyperparameter settings.
"""

import os
import json
import struct
import shutil
import gzip
import urllib.request
import urllib.error
import numpy as np
from datetime import datetime
from functools import partial
from typing import List, Any

# -------------------
# MPI
# -------------------
from mpi4py import MPI

# -------------------
# JAX / neural-tangents / optax
# -------------------
import jax
import jax.numpy as jnp
from jax import random as jax_random
import optax
import neural_tangents as nt
from neural_tangents import stax

# -------------------
# Flush-print
# -------------------
print = partial(print, flush=True)


# =============================================================================
# 1) MNIST Download + Local Loader (with graceful fallback)
# =============================================================================
def maybe_download_mnist(data_dir: str):
    """
    Ensure the 4 raw MNIST files exist locally (unzipped).
    If not, attempt download from Yann LeCun's site and unzip them.
    """
    os.makedirs(data_dir, exist_ok=True)

    base_url = "http://yann.lecun.com/exdb/mnist/"
    files_gz = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    for fgz in files_gz:
        gz_path = os.path.join(data_dir, fgz)
        raw_file = fgz.replace(".gz", "")
        raw_path = os.path.join(data_dir, raw_file)

        if os.path.exists(raw_path):
            continue
        if os.path.exists(gz_path):
            continue

        url = base_url + fgz
        print(f"Attempting download of {fgz} from {url} -> {gz_path}")
        try:
            urllib.request.urlretrieve(url, gz_path)
            print(f"Downloaded: {gz_path}")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            print(f"[Warning] Download for {fgz} failed due to: {e}")
            if not os.path.exists(gz_path) and not os.path.exists(raw_path):
                msg = (f"Failed to download {fgz} and local file not found.\n"
                       f"Please place {fgz} or {raw_file} manually into {data_dir}.")
                raise RuntimeError(msg)

    for fgz in files_gz:
        gz_path = os.path.join(data_dir, fgz)
        raw_file = fgz.replace(".gz", "")
        raw_path = os.path.join(data_dir, raw_file)

        if not os.path.exists(raw_path):
            if not os.path.exists(gz_path):
                msg = (f"Missing both {raw_file} and {fgz}. "
                       f"Please place them manually in {data_dir}.")
                raise RuntimeError(msg)

            print(f"Extracting {gz_path} to {raw_path}...")
            with gzip.open(gz_path, 'rb') as f_in, open(raw_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            print(f"Extracted: {raw_path}")


def load_mnist_local(data_dir: str):
    """Load MNIST from local raw files in `data_dir`."""
    def read_idx_images(filename):
        with open(filename, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
            return images

    def read_idx_labels(filename):
        with open(filename, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            labels = np.fromfile(f, dtype=np.uint8)
            return labels

    train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte')
    train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte')
    test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte')
    test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte')

    X_train_np = read_idx_images(train_images_path)
    y_train_np = read_idx_labels(train_labels_path)
    X_test_np = read_idx_images(test_images_path)
    y_test_np = read_idx_labels(test_labels_path)

    X_train_np = X_train_np.astype(np.float32) / 255.0
    X_test_np = X_test_np.astype(np.float32) / 255.0

    X_train_np = X_train_np[..., np.newaxis]
    X_test_np = X_test_np[..., np.newaxis]

    X_train = jnp.array(X_train_np)
    y_train = jnp.array(y_train_np, dtype=jnp.int32)
    X_test = jnp.array(X_test_np)
    y_test = jnp.array(y_test_np, dtype=jnp.int32)

    return X_train, y_train, X_test, y_test


def load_mnist_dataset():
    """Wrapper that loads MNIST from local raw files."""
    data_dir = "/path/to/your/mnist/data"  # Change this to your data directory
    return load_mnist_local(data_dir)


# =============================================================================
# 2) LeNet Implementation
# =============================================================================
def build_lenet(num_classes=10):
    """
    Build a LeNet-style CNN using neural-tangents (stax).
    Architecture:
    - Conv 6 filters (5x5)
    - MaxPool (2x2)
    - Conv 16 filters (5x5)
    - MaxPool (2x2)
    - Dense 120
    - Dense 84
    - Dense num_classes
    """
    return stax.serial(
        # First conv block
        stax.Conv(6, (5, 5), padding='SAME'),
        stax.Relu(),
        stax.AvgPool((2, 2), strides=(2, 2)),
        
        # Second conv block
        stax.Conv(16, (5, 5), padding='SAME'),
        stax.Relu(),
        stax.AvgPool((2, 2), strides=(2, 2)),
        
        # Flatten and dense layers
        stax.Flatten(),
        stax.Dense(120),
        stax.Relu(),
        stax.Dense(84),
        stax.Relu(),
        stax.Dense(num_classes)
    )


# =============================================================================
# 3) Kernel / Metric Utilities
# =============================================================================
def center_matrix(K):
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
    K1c = center_matrix(K1)
    K2c = center_matrix(K2)
    hsic = jnp.sum(K1c * K2c)
    norm1 = jnp.sqrt(jnp.sum(K1c * K1c))
    norm2 = jnp.sqrt(jnp.sum(K2c * K2c))
    return hsic / (norm1 * norm2)

def compute_ntk_metrics(KA, KB):
    if KA is None or KB is None:
        return {'cka': float('nan'), 'norm_frob': float('nan')}
    try:
        cka_val = float(compute_cka(KA, KB))
        frob_val = float(normalized_frobenius_norm(KA, KB))
        return {'cka': cka_val, 'norm_frob': frob_val}
    except Exception as e:
        print(f"Error computing NTK metrics: {e}")
        return {'cka': float('nan'), 'norm_frob': float('nan')}


# =============================================================================
# 4) Loss and NTK
# =============================================================================
def cross_entropy_loss(params, apply_fn, inputs, targets):
    logits = apply_fn(params, inputs)
    one_hot = jax.nn.one_hot(targets, 10)
    return optax.softmax_cross_entropy(logits, one_hot).mean()

def compute_empirical_ntk(apply_fn, params, x1, x2=None):
    ntk_fn = nt.empirical_ntk_fn(
        apply_fn,
        vmap_axes=0,
        trace_axes=(-1,)
    )
    if x2 is None:
        return ntk_fn(x1, None, params)
    else:
        return ntk_fn(x1, x2, params)


# =============================================================================
# 5) Training + Evaluation
# =============================================================================


def train_and_evaluate(init_fn, apply_fn, infinite_kernel_fn,
                       X_train, y_train,
                       X_test, y_test,
                       batch_size, epochs, lr, weight_decay,
                       key):
    """Train LeNet and compute NTK metrics using larger subset for H100."""
    
    # Initialize network
    output_shape, params = init_fn(key, X_train.shape)

    # Use 15k examples for training NTK and 5k for test NTK
    max_train_ntk = 15000
    max_test_ntk = 5000
    X_train_sub = X_train[:max_train_ntk]
    y_train_sub = y_train[:max_train_ntk]
    X_test_sub = X_test[:max_test_ntk]
    y_test_sub = y_test[:max_test_ntk]
    one_hot_train_sub = jax.nn.one_hot(y_train_sub, 10)

    # Initial Empirical NTK
    try:
        print("Computing initial empirical NTK...")
        init_ntk = compute_empirical_ntk(apply_fn, params, X_train_sub)
        init_ntk_test = compute_empirical_ntk(apply_fn, params, X_test_sub, X_train_sub)

        reg_val = 1e-6 * jnp.trace(init_ntk) / len(X_train_sub)
        K_reg_init = init_ntk + reg_val * jnp.eye(len(X_train_sub))
        alpha_init = jnp.linalg.solve(K_reg_init, one_hot_train_sub)
        logits_init = init_ntk_test @ alpha_init
        init_pred = jnp.argmax(logits_init, axis=1)
        init_ntk_accuracy = 1.0 - jnp.mean(init_pred != y_test_sub)
        print(f"Initial Empirical NTK Test Accuracy (15k subset): {init_ntk_accuracy:.6f}")
    except Exception as e:
        print(f"[Warning] Initial empirical NTK computation failed: {e}")
        init_ntk = None
        init_ntk_accuracy = float('nan')

    # Infinite-Width NTK
    try:
        print("Computing infinite-width NTK...")
        inf_train_train_ntk = infinite_kernel_fn(X_train_sub, X_train_sub, get='ntk')
        inf_test_train_ntk = infinite_kernel_fn(X_test_sub, X_train_sub, get='ntk')

        reg_val_inf = 1e-6 * jnp.trace(inf_train_train_ntk) / len(X_train_sub)
        K_reg_inf = inf_train_train_ntk + reg_val_inf * jnp.eye(len(X_train_sub))
        alpha_inf = jnp.linalg.solve(K_reg_inf, one_hot_train_sub)
        logits_inf = inf_test_train_ntk @ alpha_inf
        inf_pred = jnp.argmax(logits_inf, axis=1)
        inf_ntk_accuracy = 1.0 - jnp.mean(inf_pred != y_test_sub)
        print(f"Infinite-Width NTK Test Accuracy (15k subset): {inf_ntk_accuracy:.6f}")
    except Exception as e:
        print(f"[Warning] Infinite-width NTK computation failed: {e}")
        inf_train_train_ntk = None
        inf_ntk_accuracy = float('nan')

    # Training Setup
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
    def train_step(params, opt_state, bx, by):
        def loss_fn(p):
            return cross_entropy_loss(p, apply_fn, bx, by)
        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_val

    # Training Loop
    best_test_accuracy = 0.0
    training_history = {
        'train_losses': [],
        'test_accuracies': [],
        'epochs': []
    }

    print("\nStarting training...")
    key_train = key
    for epoch in range(epochs):
        key_train, subkey = jax.random.split(key_train)
        perm = jax.random.permutation(subkey, len(X_train))
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]

        epoch_losses = []
        for i in range(0, len(X_train), batch_size):
            bx = X_train_shuffled[i:i+batch_size]
            by = y_train_shuffled[i:i+batch_size]
            params, opt_state, loss_val = train_step(params, opt_state, bx, by)
            epoch_losses.append(loss_val)

        if epoch % 5 == 0 or epoch == epochs - 1:
            train_loss = jnp.mean(jnp.array(epoch_losses))
            logits_test = apply_fn(params, X_test)
            test_pred = jnp.argmax(logits_test, axis=1)
            test_accuracy = 1.0 - jnp.mean(test_pred != y_test)
            best_test_accuracy = max(best_test_accuracy, test_accuracy)

            training_history['train_losses'].append(float(train_loss))
            training_history['test_accuracies'].append(float(test_accuracy))
            training_history['epochs'].append(epoch)
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | "
                  f"Test Accuracy: {test_accuracy:.6f} | Best: {best_test_accuracy:.6f}")

    # Final Empirical NTK
    try:
        print("\nComputing final empirical NTK...")
        final_ntk = compute_empirical_ntk(apply_fn, params, X_train_sub)
        final_ntk_test = compute_empirical_ntk(apply_fn, params, X_test_sub, X_train_sub)

        reg_val_final = 1e-6 * jnp.trace(final_ntk) / len(X_train_sub)
        K_reg_final = final_ntk + reg_val_final * jnp.eye(len(X_train_sub))
        alpha_final = jnp.linalg.solve(K_reg_final, one_hot_train_sub)
        logits_final = final_ntk_test @ alpha_final
        final_pred = jnp.argmax(logits_final, axis=1)
        final_ntk_accuracy = 1.0 - jnp.mean(final_pred != y_test_sub)
        print(f"Final Empirical NTK Test Accuracy (15k subset): {final_ntk_accuracy:.6f}")
    except Exception as e:
        print(f"[Warning] Final empirical NTK computation failed: {e}")
        final_ntk = None
        final_ntk_accuracy = float('nan')

    # Compute Similarity Metrics
    print("\nComputing NTK similarity metrics...")
    init_vs_final = compute_ntk_metrics(init_ntk, final_ntk)
    inf_vs_final = compute_ntk_metrics(inf_train_train_ntk, final_ntk)

    print(f"CKA (init vs final): {init_vs_final['cka']:.6f}")
    print(f"Frob (init vs final): {init_vs_final['norm_frob']:.6f}")
    print(f"CKA (inf vs final): {inf_vs_final['cka']:.6f}")
    print(f"Frob (inf vs final): {inf_vs_final['norm_frob']:.6f}")

    return {
        'params': params,
        'best_test_accuracy': float(best_test_accuracy),
        'training_history': training_history,
        'init_ntk_accuracy': float(init_ntk_accuracy),
        'final_ntk_accuracy': float(final_ntk_accuracy),
        'inf_ntk_accuracy': float(inf_ntk_accuracy),
        'init_vs_final': init_vs_final,
        'inf_vs_final': inf_vs_final,
        'init_ntk': init_ntk,
        'final_ntk': final_ntk,
        'inf_ntk': inf_train_train_ntk
    }

# =============================================================================
# 6) Save Functions
# =============================================================================
def save_results(results: List[dict], results_dir: str, filename_suffix: str):
    try:
        path = os.path.join(results_dir, f'results_{filename_suffix}.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print(f"Error saving results: {e}")

def save_model(params: Any, path: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        flattened = {}
        for idx, leaf in enumerate(jax.tree_util.tree_leaves(params)):
            flattened[f'param_{idx}'] = np.array(leaf)
        np.savez(path, **flattened)
    except Exception as e:
        print(f"Error saving model: {e}")

def save_dataset(X: jnp.ndarray, y: jnp.ndarray, path: str, rank: int):
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


# =============================================================================
# 7) Main
# =============================================================================
def main():
    # MPI init
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Experiment name / hyperparams
    experiment_name = "mnist_lenet"
    n_train_sizes = [10, 100, 1000]
    batch_size = 64
    epochs = 100  # Reduced from 2500 for faster iteration
    learning_rates = [0.01]  # Increased from 0.001
    weight_decay = 1e-4

    # Directories
    base_path = "./"
    results_dir = os.path.join(base_path, experiment_name, "results")
    data_dir = results_dir
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Timestamp
    if rank == 0:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        hyperparams = {
            'n_train_sizes': n_train_sizes,
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rates': learning_rates,
            'weight_decay': weight_decay,
            'num_workers': size
        }
        hyperparams_path = os.path.join(results_dir, f'hyperparameters_{timestamp}.json')
        with open(hyperparams_path, 'w') as f:
            json.dump(hyperparams, f, indent=4)
    else:
        timestamp = None

    timestamp = comm.bcast(timestamp, root=0)

    # Rank 0 loads MNIST
    if rank == 0:
        print("Loading MNIST from local raw files...")
        X_train_master, y_train_master, X_test, y_test = load_mnist_dataset()
        print("Data loaded. Shapes:", X_train_master.shape, X_test.shape)

        master_data_path = os.path.join(data_dir, f'master_data_{timestamp}.npz')
        if not save_dataset(X_train_master, y_train_master, master_data_path, rank):
            raise RuntimeError("Failed to save master dataset")

        test_data_path = os.path.join(data_dir, f'test_data_{timestamp}.npz')
        if not save_dataset(X_test, y_test, test_data_path, rank):
            raise RuntimeError("Failed to save test dataset")

        print("Master saved datasets.")
    else:
        X_train_master, y_train_master, X_test, y_test = None, None, None, None

    # Broadcast
    X_train_master = comm.bcast(X_train_master, root=0)
    y_train_master = comm.bcast(y_train_master, root=0)
    X_test = comm.bcast(X_test, root=0)
    y_test = comm.bcast(y_test, root=0)

    # Generate all combinations
    all_combinations = []
    for n_tr in n_train_sizes:
        for lr in learning_rates:
            combo = {
                'n_train': n_tr,
                'lr': lr
            }
            all_combinations.append(combo)

    # Distribute combos among ranks
    worker_combos = [c for i, c in enumerate(all_combinations) if i % size == rank]

    # Store results
    results_list = []

    for combo in worker_combos:
        print(f"Rank {rank} working on combo: {combo}")
        n_train = combo['n_train']
        lr = combo['lr']

        # Subsample if needed
        X_train = X_train_master[:n_train]
        y_train = y_train_master[:n_train]

        # Save training subset
        train_data_path = os.path.join(
            data_dir,
            f"train_data_n{n_train}_lr{lr}_{timestamp}_rank{rank}.npz"
        )
        if not save_dataset(X_train, y_train, train_data_path, rank):
            print(f"[Warning] Failed to save training subset for combo {combo}")

        # Create and train LeNet
        try:
            key = jax_random.PRNGKey(hash(str(combo)) & 0xffffffff)
            init_fn, apply_fn, inf_kernel_fn = build_lenet(num_classes=10)

            train_out = train_and_evaluate(
                init_fn=init_fn,
                apply_fn=apply_fn,
                infinite_kernel_fn=inf_kernel_fn,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                batch_size=batch_size,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                key=key
            )

            final_params = train_out['params']

            # Save model
            model_path = os.path.join(
                results_dir,
                f"model_n{n_train}_lr{lr}_{timestamp}_rank{rank}.npz"
            )
            save_model(final_params, model_path)

            # Save NTKs
            init_ntk_path = os.path.join(
                results_dir,
                f"init_ntk_n{n_train}_lr{lr}_{timestamp}_rank{rank}.npy"
            )
            final_ntk_path = os.path.join(
                results_dir,
                f"final_ntk_n{n_train}_lr{lr}_{timestamp}_rank{rank}.npy"
            )
            inf_ntk_path = os.path.join(
                results_dir,
                f"infinite_ntk_n{n_train}_lr{lr}_{timestamp}_rank{rank}.npy"
            )

            if train_out['init_ntk'] is not None:
                np.save(init_ntk_path, np.array(train_out['init_ntk']))
            if train_out['final_ntk'] is not None:
                np.save(final_ntk_path, np.array(train_out['final_ntk']))
            if train_out['inf_ntk'] is not None:
                np.save(inf_ntk_path, np.array(train_out['inf_ntk']))

            # Gather results
            result_dict = {
                'n_train': n_train,
                'lr': lr,
                'best_test_accuracy': train_out['best_test_accuracy'],
                'init_ntk_accuracy': train_out['init_ntk_accuracy'],
                'final_ntk_accuracy': train_out['final_ntk_accuracy'],
                'infinite_ntk_accuracy': train_out['inf_ntk_accuracy'],
                'ntk_cka_init_vs_final': train_out['init_vs_final']['cka'],
                'ntk_frob_init_vs_final': train_out['init_vs_final']['norm_frob'],
                'ntk_cka_inf_vs_final': train_out['inf_vs_final']['cka'],
                'ntk_frob_inf_vs_final': train_out['inf_vs_final']['norm_frob'],
                'training_history': train_out['training_history'],
                'worker_rank': rank,
                'model_path': os.path.basename(model_path),
                'init_ntk_path': os.path.basename(init_ntk_path),
                'final_ntk_path': os.path.basename(final_ntk_path),
                'inf_ntk_path': os.path.basename(inf_ntk_path)
            }
            results_list.append(result_dict)

            # Save partial results
            save_results(results_list, results_dir, f'{timestamp}_rank{rank}')

        except Exception as e:
            print(f"[Error] Exception for combo {combo}: {e}")
            continue

        print(f"Rank {rank} done with combo {combo}.")

    # MPI barrier
    comm.Barrier()

    # Gather results
    all_worker_results = comm.gather(results_list, root=0)

    if rank == 0:
        combined_results = []
        for wr in all_worker_results:
            combined_results.extend(wr)

        final_results_path = os.path.join(results_dir, f'final_results_{timestamp}.json')
        with open(final_results_path, 'w') as f:
            json.dump(combined_results, f, indent=4)

        print("All workers done. Final results saved.")


if __name__ == "__main__":
    main()