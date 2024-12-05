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
        
    U, _ = np.linalg.qr(np.random.randn(ambient_dim, latent_dim).astype(np.float32))
    X = np.random.randn(n_samples, ambient_dim).astype(np.float32)
    X_latent = X @ U
    
    terms = generate_polynomials(latent_dim, degree)
    y = np.zeros(n_samples, dtype=np.float32)
    coeff_vec = []
    
    for term in terms:
        if sum(term) > 0:
            coef = np.float32(np.random.randn())
            coeff_vec.append(coef)
            term_value = np.ones(n_samples, dtype=np.float32)
            for dim, power in enumerate(term):
                if power > 0:
                    term_value *= X_latent[:, dim] ** power
            y += coef * term_value
    
    y = y / np.std(y)
    noise = (noise_std * np.random.choice([-1, 1], size=n_samples)).astype(np.float32)
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

def compute_diagonal_regularization(k_train_train, batch_size=1000):
    n = k_train_train.shape[0]
    n_batches = (n + batch_size - 1) // batch_size
    diag = jnp.zeros(n, dtype=jnp.float32)
    
    for i in range(n_batches):
        start_i = i * batch_size
        end_i = min((i + 1) * batch_size, n)
        diag = diag.at[start_i:end_i].set(jnp.diag(k_train_train[start_i:end_i, start_i:end_i]))
    
    mean_diag = jnp.mean(diag)
    return jnp.maximum(jnp.float32(0), mean_diag * jnp.float32(1e-6) * jnp.ones_like(diag))

def matrix_vector_product_with_reg(k_train_train, v, reg_lambda=1e-6, diag_reg=None, batch_size=1000):
    n = k_train_train.shape[0]
    n_batches = (n + batch_size - 1) // batch_size
    result = jnp.zeros_like(v, dtype=jnp.float32)
    
    result = result + jnp.float32(reg_lambda) * v
    if diag_reg is not None:
        result = result + diag_reg.reshape(-1, 1) * v
    
    for i in range(n_batches):
        start_i = i * batch_size
        end_i = min((i + 1) * batch_size, n)
        row_batch = k_train_train[start_i:end_i]
        result = result.at[start_i:end_i].set(
            result[start_i:end_i] + jnp.dot(row_batch, v)
        )
    
    return result

def conjugate_gradient_solve_multi(k_train_train, y_train_batch, batch_size=1000, max_iter=200, tol=1e-8):
    n = k_train_train.shape[0]
    print("Computing diagonal regularization...")
    diag_reg = compute_diagonal_regularization(k_train_train, batch_size)
    x = jnp.zeros_like(y_train_batch, dtype=jnp.float32)
    r = y_train_batch - matrix_vector_product_with_reg(k_train_train, x, 
                                                      reg_lambda=jnp.float32(1e-6), 
                                                      diag_reg=diag_reg,
                                                      batch_size=batch_size)
    p = r
    r_norm_sq = jnp.sum(r * r, axis=0)
    initial_residual = jnp.sqrt(r_norm_sq)
    print(f"Initial residual: {jnp.mean(initial_residual):.6e}")
    
    for i in range(max_iter):
        Ap = matrix_vector_product_with_reg(k_train_train, p,
                                          reg_lambda=jnp.float32(1e-6),
                                          diag_reg=diag_reg,
                                          batch_size=batch_size)
        
        alpha = r_norm_sq / jnp.sum(p * Ap, axis=0)
        x = x + alpha[None, :] * p
        r_new = r - alpha[None, :] * Ap
        r_norm_sq_new = jnp.sum(r_new * r_new, axis=0)
        beta = r_norm_sq_new / r_norm_sq
        r = r_new
        r_norm_sq = r_norm_sq_new
        p = r + beta[None, :] * p
        
        residual = jnp.sqrt(r_norm_sq)
        relative_residual = residual / initial_residual
        
        if (i + 1) % 10 == 0:
            print(f"CG iteration {i+1}, mean residual: {jnp.mean(residual):.6e}, "
                  f"mean relative residual: {jnp.mean(relative_residual):.6e}")
        
        if jnp.all(relative_residual < tol):
            print(f"CG converged in {i+1} iterations")
            break
            
        if jnp.any(jnp.isnan(residual)) or jnp.any(jnp.isinf(residual)):
            print("Warning: Numerical instability detected!")
            break
    
    return x

def compute_predictions_in_batches(k_train_train, k_test_train, y_train, 
                                 batch_size=1000, output_batch_size=10):
    print("Computing solve using conjugate gradient...")
    
    n_train = k_train_train.shape[0]
    n_test = k_test_train.shape[0]
    n_outputs = y_train.shape[1]
    
    n_output_batches = (n_outputs + output_batch_size - 1) // output_batch_size
    predictions = []
    
    for i in range(n_output_batches):
        start_out = i * output_batch_size
        end_out = min((i + 1) * output_batch_size, n_outputs)
        print(f"\nProcessing output batch {i+1}/{n_output_batches} "
              f"(dimensions {start_out+1} to {end_out})")
        
        y_batch = y_train[:, start_out:end_out]
        solution_batch = conjugate_gradient_solve_multi(
            k_train_train,
            y_batch,
            batch_size=batch_size
        )
        
        batch_predictions = []
        n_pred_batches = (n_test + batch_size - 1) // batch_size
        print(f"Computing predictions in {n_pred_batches} batches...")
        
        for j in range(n_pred_batches):
            start_idx = j * batch_size
            end_idx = min((j + 1) * batch_size, n_test)
            k_batch = k_test_train[start_idx:end_idx]
            pred_batch = jnp.dot(k_batch, solution_batch)
            batch_predictions.append(pred_batch)
            
            if (j + 1) % 5 == 0 or (j + 1) == n_pred_batches:
                print(f"Processed prediction batch {j + 1}/{n_pred_batches}")
        
        output_preds = jnp.concatenate(batch_predictions, axis=0)
        predictions.append(output_preds)
    
    final_predictions = jnp.hstack(predictions)
    print(f"Final predictions shape: {final_predictions.shape}")
    return final_predictions

def main():
    # Set default dtypes for JAX and NumPy
    jax.config.update("jax_enable_x64", False)  # Force float32
    
    # Configuration
    results_dir = '/mnt/users/goringn/NNs_vs_Kernels/low_dim_poly/results'
    experiment_name = 'ntk_400'
    train_sizes = [10, 500, 1000, 5000, 10000, 20000, 50000,60000]
    test_size = 20000
    ambient_dim = 20
    latent_dim = 3
    degree = 5
    noise_std = 0.0
    hidden_dim = 400
    kernel_type = 'ntk'
    batch_size = 2000
    output_batch_size = 2
    reg_lambda = 1e-6
    
    # Create network using Neural Tangents
    init_fn, apply_fn, kernel_fn = nt.stax.serial(
        nt.stax.Dense(hidden_dim), nt.stax.Relu(),
        nt.stax.Dense(hidden_dim), nt.stax.Relu(),
        nt.stax.Dense(hidden_dim), nt.stax.Relu(),
        nt.stax.Dense(hidden_dim), nt.stax.Relu(),
        nt.stax.Dense(hidden_dim), nt.stax.Relu(),
        nt.stax.Dense(1)
    )

    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    print(f"Created/verified results directory at: {results_dir}")

    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(results_dir, f'{experiment_name}_{timestamp}')
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Created experiment directory at: {experiment_dir}")

    # Initialize results array
    results = np.zeros((1, len(train_sizes)), dtype=np.float32)

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
        X_train = jnp.array(X_train, dtype=jnp.float32)
        y_train = jnp.array(y_train, dtype=jnp.float32).reshape(-1, 1)
        X_test = jnp.array(X_test, dtype=jnp.float32)
        y_test = jnp.array(y_test, dtype=jnp.float32).reshape(-1, 1)
        
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")
        
        # Compute kernels and predictions
        print("Computing training kernel...")
        k_train_train = compute_kernel_in_batches(kernel_fn, X_train, batch_size=batch_size, kernel_type=kernel_type)
        print(f"k_train_train shape: {k_train_train.shape}")
        
        print("Computing test-train kernel...")
        k_test_train = compute_kernel_in_batches(kernel_fn, X_test, X_train, batch_size=batch_size, kernel_type=kernel_type)
        print(f"k_test_train shape: {k_test_train.shape}")
        
        print("Computing predictions...")
        y_pred_test = compute_predictions_in_batches(
            k_train_train, k_test_train, y_train,
            batch_size=batch_size,
            output_batch_size=output_batch_size
        )
        
        # Calculate and store test error
        # Calculate and store test error
        test_error = jnp.mean((y_pred_test - y_test) ** 2)
        results[0, size_idx] = float(test_error)  # Convert to float for numpy array
        
        print(f'Test Error: {test_error:.6f}')
        
        # Save intermediate results with explicit file closing
        result_dict = {
            'test_error': float(test_error),
            'train_size': train_size,
            'timestamp': timestamp
        }
        intermediate_path = os.path.join(experiment_dir, f'results_size_{train_size}.npy')
        try:
            np.save(intermediate_path, result_dict)
            print(f"Saved intermediate results to: {intermediate_path}")
        except Exception as e:
            print(f"Error saving intermediate results: {e}")

    # Save final results with explicit file closing
    final_results = {
        'results_array': results,
        'train_sizes': train_sizes,
        'timestamp': timestamp,
        'experiment_name': experiment_name
    }
    final_path = os.path.join(experiment_dir, 'prediction_results.npy')
    try:
        np.save(final_path, final_results)
        print(f"Saved final results to: {final_path}")
    except Exception as e:
        print(f"Error saving final results: {e}")

    # Print final results
    print("\nFinal Results:")
    print("Train Size | Test Error")
    print("-" * 25)
    for i, size in enumerate(train_sizes):
        print(f"{size:9d} | {results[0,i]:9.6f}")

    return experiment_dir  # Return the directory where results were saved

if __name__ == "__main__":
    main()