#!/usr/bin/env python3
import os
import numpy as np
from mpi4py import MPI
from itertools import product
from functools import partial
import neural_tangents as nt
from jax import random, jit, vmap
import jax.numpy as jnp
from sklearn.model_selection import train_test_split
from datetime import datetime
import json
from math import sqrt, ceil
import pathlib
import jax
import multiprocessing

# Define print globally
print = partial(print, flush=True)

def detect_system_resources():
    """Detect available CPU cores and memory"""
    total_cores = len(os.sched_getaffinity(0))
    mpi_size = MPI.COMM_WORLD.Get_size()
    
    # Calculate optimal distribution of resources
    cores_per_process = max(1, total_cores // mpi_size)
    
    # Get memory info (in GB)
    try:
        with open('/proc/meminfo') as f:
            mem_total = int([l for l in f if 'MemTotal' in l][0].split()[1]) / 1024 / 1024
    except:
        mem_total = 16  # Default to 16GB if can't detect
    
    return total_cores, cores_per_process, mem_total

def set_environment_variables():
    """Dynamically configure environment based on available resources"""
    total_cores, cores_per_process, mem_total = detect_system_resources()
    mpi_size = MPI.COMM_WORLD.Get_size()
    
    # Set thread-related environment variables
    os.environ['OMP_NUM_THREADS'] = str(cores_per_process)
    os.environ['MKL_NUM_THREADS'] = str(cores_per_process)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cores_per_process)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cores_per_process)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cores_per_process)
    
    # JAX-specific settings
    os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={cores_per_process}'
    os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    
    return total_cores, cores_per_process, mem_total

# Call this before any other imports
TOTAL_CORES, CORES_PER_PROCESS, MEMORY_GB = set_environment_variables()

def configure_parallel_environment():
    """Configure parallel processing environment"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"System Configuration:")
        print(f"Total CPU cores: {TOTAL_CORES}")
        print(f"MPI processes: {size}")
        print(f"Cores per process: {CORES_PER_PROCESS}")
        print(f"Total memory: {MEMORY_GB:.1f} GB")
    
    return comm, rank, size

def initialize_globals():
    global COMM, RANK, SIZE
    COMM, RANK, SIZE = configure_parallel_environment()

# Initialize globals
initialize_globals()

def compute_optimal_chunk_distribution(n_samples, n_ranks, min_chunk_size=1000):
    """Dynamically compute optimal chunk distribution"""
    # Calculate chunk size based on available cores and memory
    memory_based_chunk = int(MEMORY_GB * 1e9 / (8 * n_ranks * 4))  # Assume 4 bytes per float32
    core_based_chunk = n_samples // (n_ranks * CORES_PER_PROCESS)
    optimal_chunk = max(min_chunk_size, min(memory_based_chunk, core_based_chunk))
    
    chunks = []
    for i in range(0, n_samples, optimal_chunk):
        end = min(i + optimal_chunk, n_samples)
        chunks.append((i, end))
    return chunks


def generate_polynomials(r, d):
    """Generate all multi-indices where sum(alpha) <= d"""
    indices = [alpha for alpha in product(range(d + 1), repeat=r) if sum(alpha) <= d]
    return indices
def generate_latent_poly_data(n_samples, ambient_dim, latent_dim, degree, noise_std=0.1, random_state=None):
    """Generate synthetic data with improved memory efficiency"""
    if random_state is not None:
        np.random.seed(random_state)
    
    chunk_size = min(10000, n_samples)
    X = np.empty((n_samples, ambient_dim), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)
    
    U, _ = np.linalg.qr(np.random.randn(ambient_dim, latent_dim).astype(np.float32))
    terms = generate_polynomials(latent_dim, degree)
    # Fixed line - don't use astype for single floats
    coeff_vec = [float(np.random.randn()) for term in terms if sum(term) > 0]
    
    for i in range(0, n_samples, chunk_size):
        end = min(i + chunk_size, n_samples)
        X[i:end] = np.random.randn(end - i, ambient_dim).astype(np.float32)
        X_latent = X[i:end] @ U
        
        for term_idx, term in enumerate(terms):
            if sum(term) > 0:
                term_value = np.ones(end - i, dtype=np.float32)
                for dim, power in enumerate(term):
                    if power > 0:
                        term_value *= X_latent[:, dim] ** power
                y[i:end] += coeff_vec[term_idx - 1] * term_value
    
    y = y / np.std(y).astype(np.float32)
    if noise_std > 0:
        y += (noise_std * np.random.choice([-1, 1], size=n_samples)).astype(np.float32)
    
    return X, y, U, coeff_vec

def create_network(input_dim, hidden_dim=1000):
    """Create neural network using neural_tangents"""
    return nt.stax.serial(
        nt.stax.Dense(hidden_dim), nt.stax.Relu(),
        nt.stax.Dense(hidden_dim), nt.stax.Relu(),
        nt.stax.Dense(hidden_dim), nt.stax.Relu(),
        nt.stax.Dense(hidden_dim), nt.stax.Relu(),
        nt.stax.Dense(hidden_dim), nt.stax.Relu(),
        nt.stax.Dense(1)
    )

@partial(jit, static_argnums=(0, 3))
def compute_kernel_chunk_parallel(kernel_fn, x1_chunk, x2_chunk, kernel_type='nngp'):
    """Compute kernel for a chunk of data with JIT compilation"""
    return kernel_fn(x1_chunk, x2_chunk, get=kernel_type)

def matrix_free_mvp(kernel_fn, X, v, chunk_size, kernel_type='nngp'):
    """Matrix-free matrix-vector product implementation"""
    n_samples = len(X)
    result = np.zeros(n_samples, dtype=np.float32)
    
    for i in range(0, n_samples, chunk_size):
        end_i = min(i + chunk_size, n_samples)
        K_chunk = compute_kernel_chunk_parallel(
            kernel_fn,
            jnp.array(X[i:end_i], dtype=jnp.float32),
            jnp.array(X, dtype=jnp.float32),
            kernel_type
        )
        result[i:end_i] = (K_chunk @ v).astype(np.float32)
    
    return result

def improved_conjugate_gradients(kernel_fn, X, y, kernel_type='nngp', chunk_size=1000, max_iter=100, tol=1e-6):
    """Improved conjugate gradients with matrix-free operations"""
    n_samples = len(X)
    
    # Initialize vectors with float32
    x = np.zeros(n_samples, dtype=np.float32)
    r = y.astype(np.float32) - matrix_free_mvp(kernel_fn, X, x, chunk_size, kernel_type)
    p = r.copy()
    
    rr = np.dot(r, r)
    initial_rr = rr
    
    for iter in range(max_iter):
        Ap = matrix_free_mvp(kernel_fn, X, p, chunk_size, kernel_type)
        
        pAp = np.dot(p, Ap)
        alpha = (rr / pAp).astype(np.float32)
        x += alpha * p
        r -= alpha * Ap
        
        rr_new = np.dot(r, r)
        beta = (rr_new / rr).astype(np.float32)
        
        if sqrt(rr_new) < tol * sqrt(initial_rr):
            if RANK == 0:
                print(f"Converged at iteration {iter}")
            break
            
        p = r + beta * p
        rr = rr_new
        
        if RANK == 0 and iter % 10 == 0:
            print(f"Iteration {iter}: residual = {sqrt(rr_new)}")
    
    return x

def compute_predictions(kernel_fn, X_train, alpha, X_eval, chunk_size, kernel_type='nngp'):
    """Compute predictions using chunked matrix multiplication"""
    n_eval = len(X_eval)
    predictions = np.zeros(n_eval, dtype=np.float32)
    
    for i in range(0, n_eval, chunk_size):
        end_i = min(i + chunk_size, n_eval)
        K_chunk = compute_kernel_chunk_parallel(
            kernel_fn,
            jnp.array(X_eval[i:end_i], dtype=jnp.float32),
            jnp.array(X_train, dtype=jnp.float32),
            kernel_type
        )
        predictions[i:end_i] = (K_chunk @ alpha).astype(np.float32)
    
    return predictions

def setup_results_directory():
    """Create results directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = pathlib.Path("results") / f"experiment_{timestamp}"
    if RANK == 0:
        results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir

def save_hyperparameters(config, results_dir):
    """Save hyperparameters to a file"""
    if RANK == 0:
        config_path = results_dir / "hyperparameters.txt"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

def save_results(results, results_dir):
    """Save results to numpy files"""
    if RANK == 0:
        np.save(results_dir / "train_sizes.npy", np.array(results["train_sizes"]))
        np.save(results_dir / "train_errors.npy", np.array(results["train_errors"]))
        np.save(results_dir / "test_errors.npy", np.array(results["test_errors"]))



def main():
    # Create results directory
    results_dir = setup_results_directory()
    
    # Dynamic configuration based on system resources
    optimal_chunk_size = int(MEMORY_GB * 1e9 / (8 * SIZE * CORES_PER_PROCESS * 4))
    optimal_chunk_size = min(optimal_chunk_size, 5000)  # Cap at 5000 to ensure good parallelization
    
    # Hyperparameters with dynamic chunk sizes
    config = {
        "train_sizes": [10, 500, 1000, 5000, 10000, 20000, 50000, 100000],
        "test_size": 20000,
        "ambient_dim": 20,
        "latent_dim": 3,
        "degree": 5,
        "noise_std": 0.0,
        "hidden_dim": 400,
        "chunk_size": optimal_chunk_size,
        "kernel_type": "ntk",
        "max_iter": 200,
        "tol": 1e-6
    }
    
    if RANK == 0:
        print(f"\nChosen configuration:")
        print(f"Chunk size: {config['chunk_size']}")
        print(f"Processes: {SIZE}")
        print(f"Cores per process: {CORES_PER_PROCESS}")
    
    # Save hyperparameters
    save_hyperparameters(config, results_dir)
    
    # Initialize results storage
    results = {
        "train_sizes": config["train_sizes"],
        "train_errors": [],
        "test_errors": []
    }
    
    # Initialize network
    init_fn, apply_fn, kernel_fn = create_network(config["ambient_dim"], config["hidden_dim"])
    
    if RANK == 0:
        print("\nSize    | Train MSE   | Test MSE")
        print("-" * 40)
    
    # Training loop
    for train_size in config["train_sizes"]:
        if RANK == 0:
            print(f"\nTraining with {train_size} samples")
        
        # Adjust chunk size based on training size
        local_chunk_size = min(config["chunk_size"], train_size // (SIZE * 2))
        
        # Generate dataset
        total_size = train_size + config["test_size"]
        X, y, U, coeff_vec = generate_latent_poly_data(
            n_samples=total_size,
            ambient_dim=config["ambient_dim"],
            latent_dim=config["latent_dim"],
            degree=config["degree"],
            noise_std=config["noise_std"],
            random_state=42
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config["test_size"], random_state=42
        )
        
        # Train using improved conjugate gradients with dynamic chunk size
        alpha = improved_conjugate_gradients(
            kernel_fn,
            X_train,
            y_train,
            kernel_type=config["kernel_type"],
            chunk_size=local_chunk_size,
            max_iter=config["max_iter"],
            tol=config["tol"]
        )
        
        # Compute predictions
        train_pred = compute_predictions(
            kernel_fn, X_train, alpha, X_train,
            local_chunk_size, config["kernel_type"]
        )
        test_pred = compute_predictions(
            kernel_fn, X_train, alpha, X_test,
            local_chunk_size, config["kernel_type"]
        )
        
        if RANK == 0:
            train_mse = np.mean((y_train - train_pred) ** 2)
            test_mse = np.mean((y_test - test_pred) ** 2)
            
            results["train_errors"].append(train_mse)
            results["test_errors"].append(test_mse)
            
            print(f"{train_size:5d} | {train_mse:.6f} | {test_mse:.6f}")
    
    # Save results
    save_results(results, results_dir)

if __name__ == "__main__":
    main()