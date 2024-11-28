#!/usr/bin/env python3
import os
import numpy as np
from itertools import product
from functools import partial
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn.init as init
from datetime import datetime
from copy import deepcopy
import json
from math import sqrt

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

class DeepNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=1000):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.network(x)

def initialize_network(model):
    """Initialize network with small random weights"""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            init.normal_(m.weight, mean=0.0, std=0.1 / sqrt(m.in_features))
            init.zeros_(m.bias)

# Training configurations
train_sizes = [10, 500, 1000, 5000, 10000, 20000, 50000, 100000]  # Different training set sizes
test_size = 20000  # Fixed test set size
ambient_dim = 20
latent_dim = 3
degree = 5
noise_std = 0.0
hidden_dim = 400
initial_lr = 0.0005
num_epochs = 2000
weight_decay = 1e-4
batch_size = 64
criterion = nn.MSELoss()

# Create results directory structure
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
base_dir = os.path.join('results', f'training_run_{timestamp}')
os.makedirs(base_dir, exist_ok=True)

# Save hyperparameters
hyperparams = {
    'train_sizes': train_sizes,
    'test_size': test_size,
    'ambient_dim': ambient_dim,
    'latent_dim': latent_dim,
    'degree': degree,
    'noise_std': noise_std,
    'model_architecture': {
        'input_dim': ambient_dim,
        'hidden_dim': hidden_dim,
        'num_layers': 5
    },
    'training_params': {
        'initial_lr': initial_lr,
        'num_epochs': num_epochs,
        'weight_decay': weight_decay,
        'batch_size': batch_size,
        'scheduler': {
            'type': 'StepLR',
            'step_size': 500,
            'gamma': 0.5
        }
    }
}

with open(os.path.join(base_dir, 'hyperparameters.json'), 'w') as f:
    json.dump(hyperparams, f, indent=4)

# Initialize results array: [initial_train_error, initial_test_error, best_train_error, best_test_error]
results = np.zeros((4, len(train_sizes)))

# Training loop for different dataset sizes
for size_idx, train_size in enumerate(train_sizes):
    print(f"\nTraining with {train_size} samples")
    
    # Generate new dataset for this experiment
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).reshape(-1, 1)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = DeepNet(input_dim=ambient_dim, hidden_dim=hidden_dim)
    initialize_network(model)
    
    # Calculate initial errors
    model.eval()
    with torch.no_grad():
        initial_train_error = criterion(model(X_train), y_train).item()
        initial_test_error = criterion(model(X_test), y_test).item()
        
    results[0, size_idx] = initial_train_error
    results[1, size_idx] = initial_test_error
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    
    best_test_error = float('inf')
    best_model = None
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                train_error = criterion(model(X_train), y_train).item()
                test_error = criterion(model(X_test), y_test).item()
                
                if test_error < best_test_error:
                    best_test_error = test_error
                    best_model = deepcopy(model.state_dict())
                    best_train_error = train_error
                    
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Train Error: {train_error:.6f}, '
                      f'Test Error: {test_error:.6f}')
    
    # Save best results
    results[2, size_idx] = best_train_error
    results[3, size_idx] = best_test_error
    
    # Save best model
    torch.save(best_model, os.path.join(base_dir, f'best_model_size_{train_size}.pt'))

# Save results
np.save(os.path.join(base_dir, 'training_results.npy'), results)

# Print final results
print("\nFinal Results:")
print("Train Size | Initial Train | Initial Test | Best Train | Best Test")
print("-" * 65)
for i, size in enumerate(train_sizes):
    print(f"{size:9d} | {results[0,i]:11.6f} | {results[1,i]:11.6f} | "
          f"{results[2,i]:10.6f} | {results[3,i]:9.6f}")