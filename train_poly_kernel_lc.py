#!/usr/bin/env python3
import os
import numpy as np
from itertools import product
from functools import partial
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
import json
from math import sqrt
from datetime import datetime
import neural_tangents as nt
import jax.numpy as jnp

print = partial(print, flush=True)

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
    Generate synthetic data for learning polynomials with low-dimensional structure.
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
            print(f"Term {i}: {term} (powers for each dimension)")
    
    # Print statistics
    total_terms = len(terms)
    non_constant_terms = len([t for t in terms if sum(t) > 0])
    print(f"\nStatistics:")
    print(f"Total terms (including constant): {total_terms}")
    print(f"Non-constant terms: {non_constant_terms}")
    
    # Normalize y to have unit variance
    y = y / np.std(y)
    
    # Add symmetric noise
    noise = noise_std * np.random.choice([-1, 1], size=n_samples)
    y = y + noise
    
    return X, y, U, coeff_vec

# Training configurations
train_sizes = [10, 500, 1000, 5000, 10000, 20000, 50000, 100000]
test_size = 20000
n_samples = 2 * max(train_sizes)
ambient_dim = 20
latent_dim = 3
degree = 5
noise_std = 0.0
hidden_dim = 400

# Create results directory structure
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
    }
}

with open(os.path.join(base_dir, 'hyperparameters.json'), 'w') as f:
    json.dump(hyperparams, f, indent=4)

# Generate full dataset
X, y, U, coeff_vec = generate_latent_poly_data(
    n_samples=n_samples,
    ambient_dim=ambient_dim, 
    latent_dim=latent_dim,
    degree=degree,
    noise_std=noise_std,
    random_state=42
)

# Define the neural network architecture using neural_tangents
init_fn, apply_fn, kernel_fn = nt.stax.serial(
    nt.stax.Dense(hidden_dim), nt.stax.Relu(),
    nt.stax.Dense(hidden_dim), nt.stax.Relu(),
    nt.stax.Dense(hidden_dim), nt.stax.Relu(),
    nt.stax.Dense(hidden_dim), nt.stax.Relu(),
    nt.stax.Dense(hidden_dim), nt.stax.Relu(),
    nt.stax.Dense(1)
)

# Convert data to jax arrays
X = jnp.array(X)
y = jnp.array(y)

# Create fixed test set
X_remaining, X_test, y_remaining, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# Initialize results array: [nngp_train_error, nngp_test_error, ntk_train_error, ntk_test_error]
results = np.zeros((4, len(train_sizes)))

# Training loop for different dataset sizes
for size_idx, train_size in enumerate(train_sizes):
    print(f"\nTraining with {train_size} samples")
    
    # Sample training data
    indices = np.random.choice(len(X_remaining), train_size, replace=False)
    X_train = X_remaining[indices]
    y_train = y_remaining[indices]
    
    # Compute kernels
    kernel = kernel_fn(X_train, X_train, get=('nngp', 'ntk'))
    
    # Get test predictions for both NNGP and NTK
    predict_fn = nt.predict.gradient_descent_mse_ensemble(
        kernel_fn=kernel_fn,
        x_train=X_train,
        y_train=y_train.reshape(-1, 1),
        diag_reg=1e-4
    )
    
    # Get both NNGP and NTK predictions
    predictions = predict_fn(x_test=X_test, get=('nngp', 'ntk'))
    
    # Compute MSE for both methods
    nngp_train_preds = predict_fn(x_test=X_train, get='nngp')
    nngp_test_preds = predictions.nngp
    ntk_train_preds = predict_fn(x_test=X_train, get='ntk')
    ntk_test_preds = predictions.ntk
    
    # Calculate errors
    results[0, size_idx] = np.mean((nngp_train_preds - y_train.reshape(-1, 1))**2)
    results[1, size_idx] = np.mean((nngp_test_preds - y_test.reshape(-1, 1))**2)
    results[2, size_idx] = np.mean((ntk_train_preds - y_train.reshape(-1, 1))**2)
    results[3, size_idx] = np.mean((ntk_test_preds - y_test.reshape(-1, 1))**2)
    
    print(f"NNGP - Train Error: {results[0, size_idx]:.6f}, Test Error: {results[1, size_idx]:.6f}")
    print(f"NTK  - Train Error: {results[2, size_idx]:.6f}, Test Error: {results[3, size_idx]:.6f}")

# Save results
np.save(os.path.join(base_dir, 'kernel_results.npy'), results)

# Print final results
print("\nFinal Results:")
print("Train Size | NNGP Train | NNGP Test  | NTK Train  | NTK Test")
print("-" * 65)
for i, size in enumerate(train_sizes):
    print(f"{size:9d} | {results[0,i]:10.6f} | {results[1,i]:10.6f} | "
          f"{results[2,i]:10.6f} | {results[3,i]:9.6f}")