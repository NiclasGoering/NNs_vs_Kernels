#!/usr/bin/env python3
import numpy as np
from functools import partial
print = partial(print, flush=True)

import jax
print("JAX devices:", jax.devices())
print("JAX CUDA available:", jax.default_backend() == "gpu")
print("Device type:", jax.devices()[0].device_kind)

import jax.numpy as jnp
from jax import random
import neural_tangents as nt
from neural_tangents import stax
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



# Generate synthetic data (keeping the same data generation from original code)
def generate_polynomials(r, d):
    """Generate all multi-indices where sum(alpha) <= d"""
    from itertools import product
    return [alpha for alpha in product(range(d + 1), repeat=r) if sum(alpha) <= d]

def generate_latent_poly_data(n_samples, ambient_dim, latent_dim, degree, noise_std=0.1, random_state=None):
    """Generate synthetic data for learning polynomials with low-dimensional structure."""
    if random_state is not None:
        np.random.seed(random_state)
        
    U, _ = np.linalg.qr(np.random.randn(ambient_dim, latent_dim))
    X = np.random.randn(n_samples, ambient_dim)
    X_latent = X @ U
    terms = generate_polynomials(latent_dim, degree)
    y = np.zeros(n_samples)
    
    coeff_vec = []
    for i, term in enumerate(terms):
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

# Generate data
n_samples = 100000
ambient_dim = 30
latent_dim = 3
degree = 5
noise_std = 0.0

X, y, U, coeff_vec = generate_latent_poly_data(
    n_samples=n_samples,
    ambient_dim=ambient_dim, 
    latent_dim=latent_dim,
    degree=degree,
    noise_std=noise_std,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to correct shape for Neural Tangents
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Define the neural network architecture using Neural Tangents
init_fn, apply_fn, kernel_fn = stax.serial(
    stax.Dense(800), stax.Relu(),
    stax.Dense(800), stax.Relu(),
    stax.Dense(800), stax.Relu(),
    stax.Dense(800), stax.Relu(),
    stax.Dense(1)
)

# Create prediction function using NNGP
predict_fn = nt.predict.gradient_descent_mse_ensemble(
    kernel_fn=kernel_fn,
    x_train=X_train,
    y_train=y_train,
    diag_reg=1e-4  # Small regularization for numerical stability
)

# Make predictions
predictions = predict_fn(x_test=X_test, get='nngp')

# Calculate MSE
mse = np.mean((predictions - y_test) ** 2)
print(f'\nFinal Results:')
print(f'Test MSE: {mse:.6f}')

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('NNGP Predictions vs True Values')
plt.grid(True)
plt.show()