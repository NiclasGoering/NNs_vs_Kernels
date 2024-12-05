#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
import matplotlib.pyplot as plt
import random
import jax
import jax.numpy as jnp
import neural_tangents as nt
from jax import random as jrandom
from functools import partial
import json
from datetime import datetime
import os

# Define print globally first
print = partial(print, flush=True)

class SpikedWishart:
    def __init__(self, d: int, beta: float):
        self.d = d
        self.beta = beta
        # Generate spike vector with norm sqrt(d)
        self.u = np.random.choice([-1, 1], size=d)  # Already has norm sqrt(d)
        
    def generate_data(self, n: int, class_label: int = 1):
        if class_label == -1:
            # Standard Gaussian
            X = torch.randn(n, self.d)
        else:
            # Spiked Wishart model
            g = torch.randn(n)
            z = torch.randn(n, self.d)
            u_tensor = torch.from_numpy(self.u).float()
            X = torch.sqrt(torch.tensor(self.beta/self.d)) * torch.outer(g, u_tensor) + z
        return X
    


def verify_data_properties(self):
    n = 10000
    X_null = self.generate_data(n, -1)
    X_spiked = self.generate_data(n, 1)
    
    # Check covariances
    cov_null = torch.cov(X_null.T)
    cov_spiked = torch.cov(X_spiked.T)
    
    print("Null hypothesis covariance should be identity:")
    print(torch.mean((cov_null - torch.eye(self.d))**2).item())
    
    print("\nSpiked covariance should be I + beta*uu^T:")
    u_tensor = torch.from_numpy(self.u).float()
    expected_cov = torch.eye(self.d) + self.beta * torch.outer(u_tensor, u_tensor) / self.d
    print(torch.mean((cov_spiked - expected_cov)**2).item())

class DeepNN(nn.Module):
    def __init__(self, d: int, hidden_dims: List[int]):
        super().__init__()
        
        layers = []
        prev_dim = d
        for hidden_dim in hidden_dims:
            linear = nn.Linear(prev_dim, hidden_dim)
            gain = nn.init.calculate_gain('relu')
            std = gain / np.sqrt(prev_dim)
            nn.init.normal_(linear.weight, mean=0.0, std=std)
            nn.init.zeros_(linear.bias)
            
            layers.extend([
                linear,
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        final_layer = nn.Linear(prev_dim, 1)
        nn.init.normal_(final_layer.weight, std=0.01)
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze()
def train_and_evaluate(model: nn.Module,
                      X_train: torch.Tensor,
                      y_train: torch.Tensor,
                      X_test: torch.Tensor,
                      y_test: torch.Tensor,
                      batch_size: int,
                      epochs: int,
                      lr: float,
                      weight_decay: float = 0.0) -> float:
    """Train the neural network and return best test error"""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.BCEWithLogitsLoss()
    
    best_test_error = float('inf')
    
    for epoch in range(epochs):
        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        
        # Calculate test error
        with torch.no_grad():
            test_pred = torch.sigmoid(model(X_test))
            train_pred = torch.sigmoid(model(X_train))
            
            # Fix: Convert to float before computing mean
            train_error = ((train_pred > 0.5).float() != y_train.float()).float().mean().item()
            test_error = ((test_pred > 0.5).float() != y_test.float()).float().mean().item()
            best_test_error = min(best_test_error, test_error)
            
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Train Error: {train_error:.6f}, Test Error: {test_error:.6f}")
    
    return best_test_error

def build_neural_tangents_model(hidden_dims: List[int]):
    """Build a neural-tangents model with the same architecture."""
    layers = []
    for dim in hidden_dims:
        layers.extend([
            nt.stax.Dense(dim), 
            nt.stax.Relu()
        ])
    layers.append(nt.stax.Dense(1))
    return nt.stax.serial(*layers)


def main():
    # Parameters
    d = 20  # input dimension
    beta = 5.0  # signal-to-noise ratio
    hidden_dims = [800, 400, 200,100]
    n_test = 2000  # number of test samples
    batch_size = 64
    epochs = 1500
    lr = 0.05
    weight_decay = 0.0002
    n_train_sizes = [20, 40, 60, 80, 100, 500,800, 1000, 2000,5000,10000]
    
    # Save hyperparameters
    hyperparams = {
        'd': d,
        'beta': beta,
        'hidden_dims': hidden_dims,
        'n_test': n_test,
        'batch_size': batch_size,
        'epochs': epochs,
        'lr': lr,
        'weight_decay': weight_decay,
        'n_train_sizes': n_train_sizes
    }
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'hyperparameters_{timestamp}.txt', 'w') as f:
        json.dump(hyperparams, f, indent=4)
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    key = jrandom.PRNGKey(42)
    
    # Initialize spiked Wishart model
    spiked_wishart = SpikedWishart(d, beta)
    
    # Generate test data (fixed)
    X_test_pos = spiked_wishart.generate_data(n_test//2, 1)
    X_test_neg = spiked_wishart.generate_data(n_test//2, -1)
    X_test = torch.cat([X_test_pos, X_test_neg])
    y_test = torch.cat([torch.ones(n_test//2), torch.zeros(n_test//2)])
    
    # Convert test data to JAX arrays
    X_test_jax = jnp.array(X_test.numpy())
    y_test_jax = jnp.array(y_test.numpy()[:, None])
    
    # Initialize neural-tangents models
    init_fn, apply_fn, kernel_fn = build_neural_tangents_model(hidden_dims)
    init_fn_random, apply_fn_random, kernel_fn_random = build_neural_tangents_model(hidden_dims)
    
    # Results storage
    results = []
    
    # For each training size
    for n_train in n_train_sizes:
        print(f"\nProcessing n_train = {n_train}")
        
        # Generate training data
        X_train_pos = spiked_wishart.generate_data(n_train//2, 1)
        X_train_neg = spiked_wishart.generate_data(n_train//2, -1)
        X_train = torch.cat([X_train_pos, X_train_neg])
        y_train = torch.cat([torch.ones(n_train//2), torch.zeros(n_train//2)])
        
        # Convert training data to JAX arrays
        X_train_jax = jnp.array(X_train.numpy())
        y_train_jax = jnp.array(y_train.numpy()[:, None])
        
        # Get NNGP and NTK predictions (matched initialization)
        predictor_matched = nt.predict.gradient_descent_mse_ensemble(
            kernel_fn, X_train_jax, y_train_jax)
        y_nngp_matched, y_ntk_matched = predictor_matched(
            x_test=X_test_jax, get=('nngp', 'ntk'))
        
        # Get NNGP and NTK predictions (random initialization)
        predictor_random = nt.predict.gradient_descent_mse_ensemble(
            kernel_fn_random, X_train_jax, y_train_jax)
        y_nngp_random, y_ntk_random = predictor_random(
            x_test=X_test_jax, get=('nngp', 'ntk'))
        
        # Train SGD model
        model = DeepNN(d, hidden_dims)
        sgd_test_error = train_and_evaluate(
            model, X_train, y_train, X_test, y_test,
            batch_size, epochs, lr, weight_decay
        )
        
        # Calculate test errors for kernel predictions
        nngp_matched_error = float(((y_nngp_matched.squeeze() > 0.5) != y_test_jax.squeeze()).mean())
        ntk_matched_error = float(((y_ntk_matched.squeeze() > 0.5) != y_test_jax.squeeze()).mean())
        nngp_random_error = float(((y_nngp_random.squeeze() > 0.5) != y_test_jax.squeeze()).mean())
        ntk_random_error = float(((y_ntk_random.squeeze() > 0.5) != y_test_jax.squeeze()).mean())
        
        # Store results
        result = {
            'n_train': n_train,
            'sgd_test_error': sgd_test_error,
            'nngp_matched_error': nngp_matched_error,
            'ntk_matched_error': ntk_matched_error,
            'nngp_random_error': nngp_random_error,
            'ntk_random_error': ntk_random_error
        }
        results.append(result)
        
        # Save results after each iteration
        with open(f'results_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Results saved for n_train = {n_train}")
        print(f"SGD Test Error: {sgd_test_error:.6f}")
        print(f"NNGP (matched) Error: {nngp_matched_error:.6f}")
        print(f"NTK (matched) Error: {ntk_matched_error:.6f}")
        print(f"NNGP (random) Error: {nngp_random_error:.6f}")
        print(f"NTK (random) Error: {ntk_random_error:.6f}")

if __name__ == "__main__":
    main()