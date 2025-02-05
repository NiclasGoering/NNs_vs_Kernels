#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from jax import random
import jax.numpy as jnp
import neural_tangents as nt
from typing import List, Set

class MSPFunction:
    def __init__(self, P: int, sets: List[Set[int]]):
        self.P = P
        self.sets = sets
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        batch_size = x.shape[0]
        result = np.zeros(batch_size, dtype=np.float64)
        
        for S in self.sets:
            term = np.ones(batch_size, dtype=np.float64)
            for idx in S:
                term = term * x[:, idx]
            result = result + term
            
        return result.reshape(-1, 1)

def create_neural_tangent_model(d: int, hidden_size: int, depth: int):
    """Create a neural tangent kernel model"""
    layers = []
    for _ in range(depth):
        layers.extend([
            nt.stax.Dense(hidden_size), 
            nt.stax.Relu()
        ])
    layers.append(nt.stax.Dense(1))
    return nt.stax.serial(*layers)

def calculate_cumulative_power(kernel_matrix: np.ndarray, target_values: np.ndarray, n_eigenvals: int = None):
    """Calculate cumulative power distribution"""
    
    # Compute eigendecomposition
    eigenvals, eigenvecs = np.linalg.eigh(kernel_matrix)
    
    # Sort in descending order
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    # Handle negative eigenvalues (numerical issues)
    eigenvals = np.maximum(eigenvals, 0)
    
    # Calculate coefficients (weights)
    coeffs = np.dot(eigenvecs.T, target_values)
    weight_sq = np.square(coeffs).flatten()
    
    # Normalize eigenvalues
    eigenvals = eigenvals / np.sum(eigenvals)
    
    # Calculate cumulative power
    if n_eigenvals is None:
        n_eigenvals = len(eigenvals)
    else:
        n_eigenvals = min(n_eigenvals, len(eigenvals))
        
    eigenvals = eigenvals[:n_eigenvals]
    weight_sq = weight_sq[:n_eigenvals]
    
    power_vals = eigenvals * weight_sq
    cum_power = np.cumsum(power_vals) / np.sum(power_vals)
    
    return eigenvals, weight_sq, cum_power

def analyze_msp_cumpower(d: int, hidden_size: int, depth: int, n_samples: int, n_eigenvals: int, save_dir: str = None):
    """Analyze cumulative power for MSP function"""
    
    # Set random seed
    key = random.PRNGKey(0)
    
    # Define MSP sets
    msp_sets = [{7}, {2, 7}, {0, 2, 7}, {5, 7, 4}, {1}, {0, 4}, {3, 7}, {0, 1, 2, 3, 4, 6, 7}]
    msp = MSPFunction(P=8, sets=msp_sets)
    
    # Generate random input data
    key, subkey = random.split(key)
    X = 2 * random.bernoulli(subkey, shape=(n_samples, d), p=0.5).astype(np.float32) - 1
    y = msp.evaluate(X)
    
    # Create NTK model
    init_fn, apply_fn, kernel_fn = create_neural_tangent_model(d, hidden_size, depth)
    
    # Compute kernel matrix
    kernel = kernel_fn(X, None, 'ntk')
    
    # Calculate cumulative power
    eigenvals, weight_sq, cum_power = calculate_cumulative_power(kernel, y, n_eigenvals)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot eigenspectrum and weights
    ax1.loglog(np.arange(1, len(eigenvals) + 1), eigenvals, 'b-', label='Eigenvalues')
    ax1.loglog(np.arange(1, len(weight_sq) + 1), weight_sq, 'r-', label='Weights²')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Magnitude')
    ax1.legend()
    ax1.set_title('Eigenspectrum and Weights')
    ax1.grid(True)
    
    # Plot cumulative power
    ax2.plot(np.arange(1, len(cum_power) + 1), cum_power, 'g-')
    ax2.set_xlabel('Number of Modes')
    ax2.set_ylabel('Cumulative Power')
    ax2.set_title('Cumulative Power Distribution')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Save results if save_dir is provided
    if save_dir is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate filename with parameters
        filename = f"msp_cumpower_d{d}_h{hidden_size}_depth{depth}_n{n_samples}"
        
        # Save individual arrays
        np.save(os.path.join(save_dir, f"{filename}_eigenvals.npy"), eigenvals)
        np.save(os.path.join(save_dir, f"{filename}_weights.npy"), weight_sq)
        np.save(os.path.join(save_dir, f"{filename}_cumpower.npy"), cum_power)
        
        # Save all data in a single npz file
        np.savez(os.path.join(save_dir, f"{filename}_all.npz"),
                eigenvals=eigenvals,
                weights=weight_sq,
                cumpower=cum_power,
                params={"d": d, 
                       "hidden_size": hidden_size,
                       "depth": depth,
                       "n_samples": n_samples,
                       "n_eigenvals": n_eigenvals})
        
        print(f"Saved results to {save_dir}")
    
    return eigenvals, weight_sq, cum_power

# Example usage:
if __name__ == "__main__":
    # Parameters
    d = 30  # input dimension
    hidden_size = 400  # width of hidden layers
    depth = 4  # number of hidden layers
    n_samples = 4096  # number of data points
    n_eigenvals = 4000  # number of eigenvalues to analyze
    
    # Create a directory for saving results
    save_dir = "/mnt/users/goringn/NNs_vs_Kernels/alignment"
    
    eigenvals, weight_sq, cum_power = analyze_msp_cumpower(
        d=d,
        hidden_size=hidden_size,
        depth=depth,
        n_samples=n_samples,
        n_eigenvals=n_eigenvals,
        save_dir=save_dir
    )