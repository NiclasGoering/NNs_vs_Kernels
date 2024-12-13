#!/usr/bin/env python3
import os
import numpy as np
from itertools import product
from functools import partial
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.init as init
from datetime import datetime
from copy import deepcopy
import json
from math import sqrt
import torch.optim as optim

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
    Generate synthetic data without normalization
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
    
    # Add symmetric noise
    noise = noise_std * np.random.choice([-1, 1], size=n_samples)
    y = y + noise
    
    return X, y, U, coeff_vec

def generate_and_normalize_data(n_train, n_test, ambient_dim, latent_dim, degree, noise_std=0.1, random_state=None):
    """
    Generate training and test data with proper normalization based on training set
    """
    # Generate training data
    X_train, y_train, U, coeff_vec = generate_latent_poly_data(
        n_samples=n_train,
        ambient_dim=ambient_dim,
        latent_dim=latent_dim,
        degree=degree,
        noise_std=noise_std,
        random_state=random_state
    )
    
    # Generate separate test data with same U and coefficients
    if random_state is not None:
        np.random.seed(random_state + 1)  # Different seed for test set
    
    X_test = np.random.randn(n_test, ambient_dim)
    X_test_latent = X_test @ U
    
    y_test = np.zeros(n_test)
    terms = generate_polynomials(latent_dim, degree)
    
    # Use same coefficients for test set
    coeff_idx = 0
    for term in terms:
        if sum(term) > 0:
            term_value = np.ones(n_test)
            for dim, power in enumerate(term):
                if power > 0:
                    term_value *= X_test_latent[:, dim] ** power
            y_test += coeff_vec[coeff_idx] * term_value
            coeff_idx += 1
    
    # Add noise to test set
    noise_test = noise_std * np.random.choice([-1, 1], size=n_test)
    y_test = y_test + noise_test
    
    # Calculate normalization parameters from training data
    y_train_mean = np.mean(y_train)
    y_train_std = np.std(y_train)
    
    # Normalize both sets using training statistics
    y_train_normalized = (y_train - y_train_mean) / y_train_std
    y_test_normalized = (y_test - y_train_mean) / y_train_std
    
    return X_train, y_train_normalized, X_test, y_test_normalized

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
train_sizes = [10,50,100,250,400, 500,750, 1000, 5000, 10000, 20000, 50000, 60000]
train_fractions = [0.8] * len(train_sizes)  # 80-20 split for all experiments
ambient_dim = 20
latent_dim = 3
degree = 5
noise_std = 0.0
hidden_dims = [25, 60, 100,  200 , 300, 500, 800, 1000]
initial_lr = 0.001 #0.0005
num_epochs = 2000
weight_decay = 1e-4
batch_size = 64
criterion = nn.MSELoss()
experiment_name = "sgd_25-100"

# Create results directory structure
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
base_dir = os.path.join('/mnt/users/goringn/NNs_vs_Kernels/low_dim_poly/results', f'{experiment_name}_{timestamp}')
os.makedirs(base_dir, exist_ok=True)

# Save hyperparameters
hyperparams = {
    'experiment_name': experiment_name,
    'train_sizes': train_sizes,
    'train_fractions': train_fractions,
    'ambient_dim': ambient_dim,
    'latent_dim': latent_dim,
    'degree': degree,
    'noise_std': noise_std,
    'hidden_dims': hidden_dims,
    'model_architecture': {
        'input_dim': ambient_dim,
        'num_layers': 5
    },
    'training_params': {
        'initial_lr': initial_lr,
        'num_epochs': num_epochs,
        'weight_decay': weight_decay,
        'batch_size': batch_size,
        'scheduler': {
            'type': 'CosineAnnealingLR',
            'num_epochs': num_epochs
        }
    }
}

with open(os.path.join(base_dir, 'hyperparameters.json'), 'w') as f:
    json.dump(hyperparams, f, indent=4)

# Initialize results array

results = np.zeros((5, len(train_sizes), len(hidden_dims)))  # Changed from 4 to 5

# Training loop
for hidden_idx, hidden_dim in enumerate(hidden_dims):
    print(f"\nTraining with hidden dimension {hidden_dim}")
    
    for size_idx, desired_train_size in enumerate(train_sizes):
        print(f"\nGenerating dataset for {desired_train_size} training samples")
        
        # Generate normalized train and test sets
        X_train, y_train, X_test, y_test = generate_and_normalize_data(
            n_train=desired_train_size,
            n_test=20000,  # Fixed test size of 20k
            ambient_dim=ambient_dim,
            latent_dim=latent_dim,
            degree=degree,
            noise_std=noise_std,
            random_state=42
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
        
        results[0, size_idx, hidden_idx] = initial_train_error
        results[1, size_idx, hidden_idx] = initial_test_error
        
        # Save initial model
        initial_model_path = os.path.join(base_dir, f'initial_model_size_{desired_train_size}_hidden_{hidden_dim}.pt')
        torch.save(model.state_dict(), initial_model_path)
        
        # Training setup
        optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

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
        results[2, size_idx, hidden_idx] = best_train_error
        results[3, size_idx, hidden_idx] = best_test_error

        # Compute final training error
        model.eval()
        with torch.no_grad():
            final_train_error = criterion(model(X_train), y_train).item()
        results[4, size_idx, hidden_idx] = final_train_error
        
        # Save best model
        best_model_path = os.path.join(base_dir, f'best_model_size_{desired_train_size}_hidden_{hidden_dim}.pt')
        torch.save(best_model, best_model_path)

# Save final results
# Compute final training error


np.save(os.path.join(base_dir, 'training_results.npy'), results)


# Print final results
print("\nFinal Results:")
for hidden_idx, hidden_dim in enumerate(hidden_dims):
    print(f"\nResults for hidden dimension {hidden_dim}:")
    print("Train Size | Initial Train | Initial Test | Best Train | Best Test | Final Train")
    print("-" * 80)
    for size_idx, size in enumerate(train_sizes):
        print(f"{size:9d} | {results[0,size_idx,hidden_idx]:11.6f} | {results[1,size_idx,hidden_idx]:11.6f} | "
              f"{results[2,size_idx,hidden_idx]:10.6f} | {results[3,size_idx,hidden_idx]:9.6f} | {results[4,size_idx,hidden_idx]:10.6f}")