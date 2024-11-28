#!/usr/bin/env python3
import numpy as np
from itertools import product
from functools import partial
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print = partial(print, flush=True)

def generate_polynomials(r, d):
    """Generate all multi-indices where sum(alpha) <= d"""
    indices = [alpha for alpha in product(range(d + 1), repeat=r) if sum(alpha) <= d]
    return indices

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

class SpectralLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Compute initialization scale according to Parametrization 1
        self.sigma = min(1, np.sqrt(out_features/in_features)) / np.sqrt(in_features)
        
        # Initialize weights with correct scaling
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * self.sigma)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Store dimensions for learning rate scaling
        self.in_features = in_features
        self.out_features = out_features
        
    def forward(self, x):
        return nn.functional.linear(x, self.weight, self.bias)

class SpectralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            SpectralLinear(input_dim, hidden_dim),
            nn.ReLU(),
            SpectralLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            SpectralLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            SpectralLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            SpectralLinear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.network(x)

def get_layer_lr(layer):
    """Compute learning rate for a layer according to Parametrization 1"""
    if isinstance(layer, SpectralLinear):
        return layer.out_features / layer.in_features
    return 1.0

# Generate data

n_samples = 30000
ambient_dim = 30
latent_dim = 3
degree = 5
noise_std=0.0

X, y, U, coeff_vec = generate_latent_poly_data(
    n_samples=n_samples,
    ambient_dim=ambient_dim, 
    latent_dim=latent_dim,
    degree=degree,
    noise_std=noise_std,
    random_state=42
)

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y).reshape(-1, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_tensor, test_size=0.2, random_state=42
)

# Create data loaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

# Initialize model
model = SpectralNet(input_dim=ambient_dim, hidden_dim=64)

# Set up layer-specific learning rates
base_lr = 0.01
param_groups = []
for name, layer in model.named_modules():
    if isinstance(layer, SpectralLinear):
        param_groups.append({
            'params': layer.parameters(),
            'lr': base_lr * get_layer_lr(layer)
        })

# Training parameters
num_epochs = 5000
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(param_groups)

# Lists to store losses
train_losses = []
test_losses = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_X, test_y = next(iter(test_loader))
        test_outputs = model(test_X)
        test_loss = criterion(test_outputs, test_y).item()
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.6f}, '
              f'Test Loss: {test_loss:.6f}')

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Test Loss Over Time')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.show()

# Final evaluation
model.eval()
with torch.no_grad():
    train_predictions = model(X_train)
    test_predictions = model(X_test)
    final_train_loss = criterion(train_predictions, y_train).item()
    final_test_loss = criterion(test_predictions, y_test).item()

print(f'\nFinal Results:')
print(f'Training MSE: {final_train_loss:.6f}')
print(f'Test MSE: {final_test_loss:.6f}')