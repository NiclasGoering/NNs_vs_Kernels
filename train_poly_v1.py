#!/usr/bin/env python3
import numpy as np
from itertools import product
from functools import partial
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.init as init
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from math import sqrt
print = partial(print, flush=True)

def generate_polynomials(r, d):
    """
    Generate all multi-indices where sum(alpha) <= d
    r: number of variables
    d: maximum degree
    """
    # Generate all multi-indices where sum(alpha) <= d
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
    
    coeff_vec=[]
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
            #nn.Linear(hidden_dim, hidden_dim),
            #nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.network(x)

def initialize_network(model):
    """Initialize network with small random weights"""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            # Slightly more conservative initialization
            init.normal_(m.weight, mean=0.0, std=0.1 / sqrt(m.in_features))
            init.zeros_(m.bias)

# Generate data
n_samples = 100000
ambient_dim = 20
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

# Convert data to PyTorch tensors
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
model = DeepNet(input_dim=ambient_dim, hidden_dim=400)
initialize_network(model)

# Training parameters
initial_lr = 0.0005
num_epochs = 2000
weight_decay = 1e-4  # Added weight decay
criterion = nn.MSELoss()

# Initialize optimizer with weight decay
optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)

# Learning rate scheduler
# Reduce learning rate by factor of 0.5 every 1000 epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

# Lists to store losses
train_losses = []
test_losses = []
learning_rates = []

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
    
    # Step the scheduler
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    learning_rates.append(current_lr)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'LR: {current_lr:.6f}, '
              f'Train Loss: {train_loss:.6f}, '
              f'Test Loss: {test_loss:.6f}')

# Plotting training curves
plt.figure(figsize=(15, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Test Loss Over Time')
plt.legend()
plt.yscale('log')
plt.grid(True)

# Learning rate plot
plt.subplot(1, 2, 2)
plt.plot(learning_rates, label='Learning Rate')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.yscale('log')
plt.grid(True)
plt.legend()

plt.tight_layout()
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