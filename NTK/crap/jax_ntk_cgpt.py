#!/usr/bin/env python3

"""
Example code using JAX + Neural Tangents (stax) to train an MLP on an MSP function.
We compute and save empirical NTKs at initial and final parameters,
as well as the test MSE from the initial/final NTK.
"""

import os
import json
import random
import numpy as np
from datetime import datetime
from functools import partial
from typing import List, Set, Tuple

# MPI
from mpi4py import MPI

# JAX / neural-tangents
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


# -------------------
# Data Generation
# -------------------
def generate_master_dataset(P, d, master_size, n_test, msp: MSPFunction, seed=42):
    """
    Generate master training set + test set with a fixed seed, returning JAX arrays.
    """
    np.random.seed(seed)
    X_train_master_np = 2.0 * np.random.binomial(1, 0.5, size=(master_size, d)) - 1.0
    X_test_np = 2.0 * np.random.binomial(1, 0.5, size=(n_test, d)) - 1.0

    X_train_master = jnp.array(X_train_master_np, dtype=jnp.float32)
    X_test = jnp.array(X_test_np, dtype=jnp.float32)
    y_train_master = msp.evaluate(X_train_master)
    y_test = msp.evaluate(X_test)
    return X_train_master, y_train_master, X_test, y_test


# -------------------
# Create MLP (no `use_bias=True`).
# -------------------
def create_mlp(d: int, hidden_size: int, depth: int):
    """
    Create an MLP with ReLU hidden layers using nt_stax (Neural Tangents).
    
    We use parameterization='standard' to ensure each Dense layer has exactly (W, b).
    We also explicitly set W_init/b_init. For instance, b_init=nt_stax.zeros will create
    a bias parameter array but initialize it to 0.

    Because older versions of Neural Tangents do not accept 'use_bias=True',
    we remove that argument. Instead, we rely on the presence of b_init to include a bias.
    """

    layers = []
    for _ in range(depth):
        layers += [
            nt_stax.Dense(
                out_dim=hidden_size,
                parameterization='standard',
                W_init=nt_stax.randn(std=0.1),
                b_init=nt_stax.zeros
            ),
            nt_stax.Relu()
        ]
    # Final layer
    layers.append(
        nt_stax.Dense(
            out_dim=1,
            parameterization='standard',
            W_init=nt_stax.randn(std=0.1),
            b_init=nt_stax.zeros
        )
    )

    init_fn, apply_fn, kernel_fn = nt_stax.serial(*layers)

    # Wrap apply_fn to force mask=None if your version of neural_tangents tries to do masking.
    def apply_fn_no_mask(params, x):
        return apply_fn(params, x, mask=None)

    return init_fn, apply_fn_no_mask, kernel_fn


# -------------------
# Empirical NTK
# -------------------
def compute_empirical_ntk(apply_fn, params, x1, x2=None):
    """
    Compute empirical NTK using neural_tangents.empirical_ntk_fn, which expects
    ntk_fn(params, x1, x2).
    """
    if x2 is None:
        x2 = x1
    ntk_fn = nt.empirical_ntk_fn(apply_fn, vmap_axes=0)
    return ntk_fn(params, x1, x2)


# -------------------
# MSE Loss
# -------------------
def mse_loss(params, apply_fn, X, y):
    """
    Mean-squared error between apply_fn(params, X) and y.
    """
    preds = apply_fn(params, X).reshape(-1)
    return jnp.mean((preds - y) ** 2)


# -------------------
# Training + Evaluate
# -------------------
def train_and_evaluate(
    init_params,
    apply_fn,
    msp: MSPFunction,
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    X_test: jnp.ndarray,
    y_test: jnp.ndarray,
    batch_size: int,
    epochs: int,
    base_lr: float,
    weight_decay: float,
):
    """
    Train MLP with AdamW + Cosine Annealing schedule. 
    Also compute + return NTKs at init & final, plus test MSE from the NTK reg. solution.
    """
    # Cosine decay for the entire training
    scheduler = optax.cosine_decay_schedule(init_value=base_lr, decay_steps=epochs)
    optimizer = optax.adamw(learning_rate=scheduler, weight_decay=weight_decay)
    opt_state = optimizer.init(init_params)

    # Evaluate initial MSE
    initial_train_error = mse_loss(init_params, apply_fn, X_train, y_train).item()
    initial_test_error = mse_loss(init_params, apply_fn, X_test, y_test).item()
    print(f"Initial predictions stats:")
    print(f"Train MSE: {initial_train_error:.6f}")
    print(f"Test  MSE: {initial_test_error:.6f}")

    # -----------------------------
    # Empirical NTK at init
    # -----------------------------
    init_ntk = compute_empirical_ntk(apply_fn, init_params, X_train)
    init_ntk_cross = compute_empirical_ntk(apply_fn, init_params, X_test, X_train)

    lam = 1e-5
    n_train = X_train.shape[0]
    reg_init_ntk = init_ntk + lam * jnp.eye(n_train)
    alpha_init = jnp.linalg.solve(reg_init_ntk, y_train)
    y_pred_init_ntk_test = init_ntk_cross @ alpha_init
    init_ntk_test_mse = jnp.mean((y_pred_init_ntk_test - y_test)**2).item()

    error_history = {
        'train_errors': [],
        'test_errors': [],
        'epochs': []
    }
    best_test_error = float('inf')
    params = init_params

    train_size = X_train.shape[0]

    @jax.jit
    def update_step(params, opt_state, Xb, yb):
        lfun = lambda prms: mse_loss(prms, apply_fn, Xb, yb)
        grads = jax.grad(lfun)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params=params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    rng = jax_random.PRNGKey(1234)

    # Training loop
    for epoch in range(epochs):
        rng, shuffle_key = jax_random.split(rng)
        perm = jax_random.permutation(shuffle_key, train_size)
        X_train = X_train[perm]
        y_train = y_train[perm]

        for i in range(0, train_size, batch_size):
            Xb = X_train[i:i+batch_size]
            yb = y_train[i:i+batch_size]
            params, opt_state = update_step(params, opt_state, Xb, yb)

        # Evaluate
        train_error = mse_loss(params, apply_fn, X_train, y_train).item()
        test_error = mse_loss(params, apply_fn, X_test, y_test).item()
        best_test_error = min(best_test_error, test_error)

        error_history['train_errors'].append(train_error)
        error_history['test_errors'].append(test_error)
        error_history['epochs'].append(epoch)

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Current Test MSE: {test_error:.6f}, Best Test MSE: {best_test_error:.6f}")
            print(f"Training MSE: {train_error:.6f}")

    # Final training error
    final_train_error = mse_loss(params, apply_fn, X_train, y_train).item()

    # -----------------------------
    # Empirical NTK at final
    # -----------------------------
    final_ntk = compute_empirical_ntk(apply_fn, params, X_train)
    final_ntk_cross = compute_empirical_ntk(apply_fn, params, X_test, X_train)
    reg_final_ntk = final_ntk + lam * jnp.eye(n_train)
    alpha_final = jnp.linalg.solve(reg_final_ntk, y_train)
    y_pred_final_ntk_test = final_ntk_cross @ alpha_final
    final_ntk_test_mse = jnp.mean((y_pred_final_ntk_test - y_test)**2).item()

    return (best_test_error,
            initial_train_error,
            final_train_error,
            error_history,
            params,
            init_ntk,
            final_ntk,
            init_ntk_test_mse,
            final_ntk_test_mse)


# -------------------
# Saving + Loading
# -------------------
def save_results(results: List[dict], results_dir: str, filename_suffix: str):
    """ Save list of result dicts to JSON. """
    try:
        path = os.path.join(results_dir, f'results_{filename_suffix}.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print(f"Error saving results: {e}")


def save_model(params, path: str):
    """
    Save model parameters (pytree of arrays) as a .npz.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        flattened = {}
        idx = 0
        # Flatten leaves in the param pytree
        for leaf in jax.tree_util.tree_leaves(params):
            flattened[f'param_{idx}'] = np.array(leaf)
            idx += 1
        np.savez(path, **flattened)
    except Exception as e:
        print(f"Error saving model: {e}")


def save_dataset(X: jnp.ndarray, y: jnp.ndarray, path: str, rank: int, min_size_bytes: int = 1000):
    """
    Save dataset as .npz with verification.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        X_np = np.array(X)
        y_np = np.array(y)
        np.savez(path,
                 X=X_np,
                 y=y_np,
                 shape_X=X_np.shape,
                 shape_y=y_np.shape,
                 saved_by_rank=rank,
                 timestamp=datetime.now().isoformat())
        # Verify
        if not os.path.exists(path):
            raise RuntimeError(f"File does not exist after save: {path}")
        if os.path.getsize(path) < min_size_bytes:
            raise RuntimeError(f"File too small after save: {path} ({os.path.getsize(path)} bytes)")
        print(f"Rank {rank}: Successfully saved and verified dataset at {path}")
        return True
    except Exception as e:
        print(f"Rank {rank}: Error saving dataset: {e}")
        return False


# -------------------
# Main
# -------------------
def main():
    # MPI init
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print(f"Rank {rank} started. Using JAX default device: {jax.default_backend()}")

    random.seed(42)
    np.random.seed(42)

    # Example parameters
    experiment_name = "jax_test1"
    P = 8 
    d = 30
    master_size = 60000
    hidden_sizes = [400]  # test a single hidden size
    depths = [4]
    n_test = 1000
    batch_size = 64
    epochs = 500
    learning_rates = [0.0001]  # test a single LR
    weight_decay = 1e-4
    n_train_sizes = [10]

    # MSP sets
    msp_sets = [{7}, {2,7}, {0,2,7}, {5,7,4}, {1}, {0,4}, {3,7}, {0,1,2,3,4,6,7}]
    msp = MSPFunction(P, msp_sets)

    base_path = "./NNs_vs_Kernels"
    results_dir = os.path.join(base_path, "stair_function/results", experiment_name)
    data_dir = os.path.join(base_path, "stair_function/data", experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Timestamp
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
            'device': "jax_default",
            'num_workers': size
        }
        hyperparams_path = os.path.join(results_dir, f'hyperparameters_{timestamp}.json')
        with open(hyperparams_path, 'w') as f:
            json.dump(hyperparams, f, indent=4)
    else:
        timestamp = None

    timestamp = comm.bcast(timestamp, root=0)

    # Rank 0: generate and save data
    if rank == 0:
        print("Master generating datasets...")
        X_train_master, y_train_master, X_test, y_test = generate_master_dataset(
            P, d, master_size, n_test, msp, seed=42
        )
        print("Data generation complete")
        print(f"Master set shape: {X_train_master.shape}")
        print(f"Test set shape: {X_test.shape}")

        # Save master
        master_data_path = os.path.join(data_dir, f'master_data_{timestamp}.npz')
        print(f"Attempting to save master data to: {os.path.abspath(master_data_path)}")
        if not save_dataset(X_train_master, y_train_master, master_data_path, rank):
            raise RuntimeError("Failed to save master dataset")

        # Save test
        test_data_path = os.path.join(data_dir, f'test_data_{timestamp}.npz')
        print(f"Attempting to save test data to: {os.path.abspath(test_data_path)}")
        if not save_dataset(X_test, y_test, test_data_path, rank):
            raise RuntimeError("Failed to save test dataset")

        print("Master saved all datasets successfully")
    else:
        X_train_master = None
        y_train_master = None
        X_test = None
        y_test = None

    # Broadcast data
    X_train_master = comm.bcast(X_train_master, root=0)
    y_train_master = comm.bcast(y_train_master, root=0)
    X_test = comm.bcast(X_test, root=0)
    y_test = comm.bcast(y_test, root=0)

    # Parameter grid
    def get_parameter_combinations(hidden_sizes, depths, n_train_sizes, learning_rates):
        combos = []
        for hs in hidden_sizes:
            for dep in depths:
                for n_tr in n_train_sizes:
                    for lr in learning_rates:
                        combos.append({
                            'hidden_size': hs,
                            'depth': dep,
                            'n_train': n_tr,
                            'lr': lr
                        })
        return combos

    all_combinations = get_parameter_combinations(hidden_sizes, depths, n_train_sizes, learning_rates)
    
    # Distribute work among ranks
    worker_combinations = []
    for i in range(len(all_combinations)):
        if i % size == rank:
            worker_combinations.append(all_combinations[i])

    results = []

    for params in worker_combinations:
        print(f"Worker {rank} processing: {params}")

        # Subsample from master
        sample_seed = (hash(f"sample_{params['n_train']}") & 0xffffffff)
        rnd = np.random.RandomState(sample_seed)
        indices_np = rnd.permutation(master_size)[:params['n_train']]
        X_train = X_train_master[indices_np]
        y_train = y_train_master[indices_np]

        train_data_filename = (
            f"train_data_h{params['hidden_size']}"
            f"_d{params['depth']}"
            f"_n{params['n_train']}"
            f"_lr{params['lr']}"
            f"_{timestamp}_rank{rank}.npz"
        )
        train_data_path = os.path.join(results_dir, train_data_filename)
        if not save_dataset(X_train, y_train, train_data_path, rank):
            print(f"Rank {rank}: Failed to save training data, skipping {params}")
            continue

        print(f"Sampled training set shape: {X_train.shape}")
        print(f"Sample seed used: {sample_seed}")

        # Initialize MLP
        init_fn, apply_fn_no_mask, kernel_fn = create_mlp(d, params['hidden_size'], params['depth'])
        key_init = jax_random.PRNGKey(123 + rank)
        input_shape = (1, d)
        out_shape, init_params = init_fn(key_init, input_shape)

        # Save initial model
        init_model_prefix = (
            f"initial_model_h{params['hidden_size']}"
            f"_d{params['depth']}"
            f"_n{params['n_train']}"
            f"_lr{params['lr']}"
        )
        init_model_path = os.path.join(
            results_dir,
            f'{init_model_prefix}_{timestamp}_rank{rank}.npz'
        )
        save_model(init_params, init_model_path)

        # Train
        try:
            (best_test_error,
             initial_train_error,
             final_train_error,
             error_history,
             final_params,
             init_ntk,
             final_ntk,
             init_ntk_test_mse,
             final_ntk_test_mse) = train_and_evaluate(
                 init_params,
                 apply_fn_no_mask,  # pass the no_mask version
                 msp,
                 X_train,
                 y_train,
                 X_test,
                 y_test,
                 batch_size,
                 epochs,
                 params['lr'],
                 weight_decay
             )

            # Save final model
            final_model_prefix = (
                f"final_model_h{params['hidden_size']}"
                f"_d{params['depth']}"
                f"_n{params['n_train']}"
                f"_lr{params['lr']}"
            )
            final_model_path = os.path.join(
                results_dir,
                f'{final_model_prefix}_{timestamp}_rank{rank}.npz'
            )
            save_model(final_params, final_model_path)

            # Save NTKs
            init_ntk_path = os.path.join(
                results_dir,
                f'init_ntk_h{params["hidden_size"]}_d{params["depth"]}_n{params["n_train"]}_lr{params["lr"]}_{timestamp}_rank{rank}.npy'
            )
            final_ntk_path = os.path.join(
                results_dir,
                f'final_ntk_h{params["hidden_size"]}_d{params["depth"]}_n{params["n_train"]}_lr{params["lr"]}_{timestamp}_rank{rank}.npy'
            )
            np.save(init_ntk_path, np.array(init_ntk))
            np.save(final_ntk_path, np.array(final_ntk))

            # Results record
            rdict = {
                'hidden_size': params['hidden_size'],
                'depth': params['depth'],
                'n_train': params['n_train'],
                'learning_rate': params['lr'],
                'test_error': best_test_error,
                'initial_train_error': initial_train_error,
                'final_train_error': final_train_error,
                'init_ntk_test_mse': init_ntk_test_mse,
                'final_ntk_test_mse': final_ntk_test_mse,
                'error_history': error_history,
                'worker_rank': rank,
                'sample_seed': sample_seed,
                'init_model_path': os.path.basename(init_model_path),
                'final_model_path': os.path.basename(final_model_path),
                'init_ntk_path': os.path.basename(init_ntk_path),
                'final_ntk_path': os.path.basename(final_ntk_path),
            }
            results.append(rdict)
            save_results(results, results_dir, f'{timestamp}_rank{rank}')

        except RuntimeError as e:
            print(f"Rank {rank}: Runtime error for params {params}. Error: {e}")
            continue

        print(f"Worker {rank} completed {params}")

    # Gather + finalize
    comm.Barrier()
    all_results = comm.gather(results, root=0)
    if rank == 0:
        combined_results = []
        for rlist in all_results:
            combined_results.extend(rlist)
        final_results_path = os.path.join(results_dir, f'final_results_{timestamp}.json')
        with open(final_results_path, 'w') as f:
            json.dump(combined_results, f, indent=4)
        print("All workers completed. Results combined and saved.")


if __name__ == "__main__":
    main()
