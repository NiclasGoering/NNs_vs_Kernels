#!/mnt/users/goringn/miniconda3/envs/gpu_a100_ntk/bin/python
import os
import numpy as np
from datetime import datetime
import jax 
import neural_tangents as nt
from jax import random
import jax.numpy as jnp
from functools import partial
from itertools import product

# Configure immediate flushing of print statements
print = partial(print, flush=True)

def generate_polynomials(r, d):
    """Generate all multi-indices where sum(alpha) <= d"""
    return [alpha for alpha in product(range(d + 1), repeat=r) if sum(alpha) <= d]

def generate_latent_poly_data(n_samples, ambient_dim, latent_dim, degree, noise_std=0.1, random_state=None):
    """Generate polynomial data with latent structure"""
    if random_state is not None:
        np.random.seed(random_state)
        
    U, _ = np.linalg.qr(np.random.randn(ambient_dim, latent_dim))
    X = np.random.randn(n_samples, ambient_dim)
    X_latent = X @ U
    
    terms = generate_polynomials(latent_dim, degree)
    y = np.zeros(n_samples)
    coeff_vec = []
    
    for term in terms:
        if sum(term) > 0:
            coef = np.random.randn()
            coeff_vec.append(coef)
            term_value = np.ones(n_samples)
            for dim, power in enumerate(term):
                if power > 0:
                    term_value *= X_latent[:, dim] ** power
            y += coef * term_value
    
    y = y / np.std(y)
    noise = noise_std * np.random.choice([-1, 1], size=n_samples)
    y = y + noise
    
    return X, y, U, coeff_vec

def compute_kernel_in_batches(kernel_fn, X1, X2=None, batch_size=1000, kernel_type='ntk'):
    """Compute kernel in batches to avoid OOM errors"""
    n1 = X1.shape[0]
    n2 = n1 if X2 is None else X2.shape[0]
    X2 = X1 if X2 is None else X2
    
    n_batches1 = (n1 + batch_size - 1) // batch_size
    n_batches2 = (n2 + batch_size - 1) // batch_size
    
    kernel_blocks = []
    print(f"Computing kernel in {n_batches1 * n_batches2} blocks...")
    
    for i in range(n_batches1):
        start_i = i * batch_size
        end_i = min((i + 1) * batch_size, n1)
        X_i = X1[start_i:end_i]
        
        row_blocks = []
        for j in range(n_batches2):
            start_j = j * batch_size
            end_j = min((j + 1) * batch_size, n2)
            X_j = X2[start_j:end_j]
            
            k_ij = kernel_fn(X_i, X_j, kernel_type)
            row_blocks.append(k_ij)
            
            if ((i * n_batches2 + j + 1) % 5 == 0) or ((i + 1) * (j + 1) == n_batches1 * n_batches2):
                print(f"Processed block {i * n_batches2 + j + 1}/{n_batches1 * n_batches2}")
        
        kernel_blocks.append(jnp.concatenate(row_blocks, axis=1))
    
    return jnp.concatenate(kernel_blocks, axis=0)

def compute_mse_prediction_batched(kernel_fn, k_train_train, k_test_train, y_train):
    """Compute MSE predictions using precomputed kernels"""
    n_train = y_train.shape[0]
    solve = jax.scipy.linalg.solve(
        k_train_train + 1e-6 * jnp.eye(n_train),
        y_train,
        assume_a='pos')
    return jnp.dot(k_test_train, solve)

def main(results_dir='results', experiment_name='nt_prediction'):
    # Configuration
    train_sizes = [10,100,1000,5000,20000,50000]
    test_size = 20000
    ambient_dim = 20
    latent_dim = 3
    degree = 5
    noise_std = 0.0
    hidden_dim = 400
    kernel_type = 'ntk'
    batch_size = 5000

    # Create network using Neural Tangents
    init_fn, apply_fn, kernel_fn = nt.stax.serial(
        nt.stax.Dense(hidden_dim), nt.stax.Relu(),
        nt.stax.Dense(hidden_dim), nt.stax.Relu(),
        nt.stax.Dense(hidden_dim), nt.stax.Relu(),
        nt.stax.Dense(hidden_dim), nt.stax.Relu(),
        nt.stax.Dense(hidden_dim), nt.stax.Relu(),
        nt.stax.Dense(1)
    )

    # Create results directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.join(results_dir, f'{experiment_name}_{timestamp}')
    os.makedirs(base_dir, exist_ok=True)

    # Initialize results array
    results = np.zeros((2, len(train_sizes)))  # [train_error, test_error]

    # Training loop for different dataset sizes
    for size_idx, train_size in enumerate(train_sizes):
        print(f"\nPredicting with {train_size} samples")
        
        # Generate dataset
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
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Convert to JAX arrays
        X_train, y_train = jnp.array(X_train), jnp.array(y_train).reshape(-1, 1)
        X_test, y_test = jnp.array(X_test), jnp.array(y_test).reshape(-1, 1)
        
        # Compute kernels and predictions
        print("Computing training kernel...")
        k_train_train = compute_kernel_in_batches(kernel_fn, X_train, batch_size=batch_size, kernel_type=kernel_type)
        
        print("Computing test-train kernel...")
        k_test_train = compute_kernel_in_batches(kernel_fn, X_test, X_train, batch_size=batch_size, kernel_type=kernel_type)
        
        print("Computing predictions...")
        y_pred_train = compute_mse_prediction_batched(kernel_fn, k_train_train, k_train_train, y_train)
        y_pred_test = compute_mse_prediction_batched(kernel_fn, k_train_train, k_test_train, y_train)
        
        # Calculate and store errors
        train_error = np.mean((y_pred_train - y_train) ** 2)
        test_error = np.mean((y_pred_test - y_test) ** 2)
        
        results[0, size_idx] = train_error
        results[1, size_idx] = test_error
        
        print(f'Train Error: {train_error:.6f}')
        print(f'Test Error: {test_error:.6f}')
        
        # Save intermediate results
        np.save(os.path.join(base_dir, f'results_size_{train_size}.npy'), 
                {'train_error': train_error, 'test_error': test_error})

    # Save final results
    np.save(os.path.join(base_dir, 'prediction_results.npy'), results)

    # Print final results
    print("\nFinal Results:")
    print("Train Size | Train Error | Test Error")
    print("-" * 40)
    for i, size in enumerate(train_sizes):
        print(f"{size:9d} | {results[0,i]:10.6f} | {results[1,i]:9.6f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='NTK Prediction with configurable results directory')
    parser.add_argument('--results-dir', type=str, default='results',
                      help='Directory to store results (default: results)')
    parser.add_argument('--experiment-name', type=str, default='nt_prediction',
                      help='Name of the experiment (default: nt_prediction)')
    
    args = parser.parse_args()
    main(args.results_dir, args.experiment_name)