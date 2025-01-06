#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Set, Tuple
import random
from functools import partial
import json
from datetime import datetime
import os
import argparse
import pickle

parser = argparse.ArgumentParser(description="Train a neural network on the MSP function.")
parser.add_argument('--n_train', type=int, help='Number of training samples')
args = parser.parse_args()
n_train = args.n_train

# Define print globally first
print = partial(print, flush=True)

class MSPFunction:
    def __init__(self, P: int, sets: List[Set[int]]):
        self.P = P
        self.sets = sets

    def evaluate(self, z: torch.Tensor) -> torch.Tensor:
        device = z.device
        batch_size = z.shape[0]
        result = torch.zeros(batch_size, dtype=torch.float32, device=device)

        for S in self.sets:
            term = torch.ones(batch_size, dtype=torch.float32, device=device)
            for idx in S:
                term = term * z[:, idx]
            result = result + term

        return result

class DeepNN(nn.Module):
    def __init__(self, d: int, hidden_size: int, depth: int, mode: str = 'special'):
        super().__init__()

        torch.set_default_dtype(torch.float32)

        layers = []
        prev_dim = d
        for _ in range(depth):
            linear = nn.Linear(prev_dim, hidden_size)

            if mode == 'special':
                # Special initialization as in original code
                gain = nn.init.calculate_gain('relu')
                std = gain / np.sqrt(prev_dim)
                nn.init.normal_(linear.weight, mean=0.0, std=std)
                nn.init.zeros_(linear.bias)
            else:
                # Standard PyTorch initialization
                nn.init.xavier_uniform_(linear.weight)
                nn.init.zeros_(linear.bias)

            layers.extend([
                linear,
                nn.ReLU()
            ])
            prev_dim = hidden_size

        final_layer = nn.Linear(prev_dim, 1)
        if mode == 'special':
            nn.init.normal_(final_layer.weight, std=0.01)
        else:
            nn.init.xavier_uniform_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze()

def save_results(results: List[dict], results_dir: str, model_prefix: str):
    """Helper function to save results with error handling"""
    try:
        results_path = os.path.join(results_dir, f'results_{model_prefix}.json')
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print(f"Error saving results: {e}")

def save_model(model: nn.Module, path: str):
    """Helper function to save model with error handling"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_cpu = model.cpu()
        torch.save(model_cpu.state_dict(), path)
        model.to(next(model.parameters()).device)
    except Exception as e:
        print(f"Error saving model: {e}")

def calculate_error(X, y, batch_size, model, device):
    with torch.no_grad():
        losses = []
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size].to(device)
            batch_y = y[i:i+batch_size].to(device)

            output = model(batch_X)
            loss = (output - batch_y) ** 2
            losses.append(loss)

        return torch.mean(torch.cat(losses)).item()


def train_and_evaluate(model: nn.Module,
                      msp: MSPFunction,
                      X_train: torch.Tensor,
                      y_train: torch.Tensor,
                      X_test: torch.Tensor,
                      y_test: torch.Tensor,
                      batch_size: int,
                      epochs: int,
                      lr: float,
                      weight_decay: float,
                      mode: str,
                      device) -> Tuple[float, float, float, float]:

    """Train the neural network and return best test error and training errors"""
    if mode == 'special':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_test_error = float('inf')

    # Get initial training error
    with torch.no_grad():
        initial_train_error = calculate_error(X_train, y_train, batch_size, model, device)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    for epoch in range(epochs):
        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size].to(device)
            batch_y = y_train[i:i+batch_size].to(device)

            optimizer.zero_grad()
            output = model(batch_X)
            loss = torch.mean((output - batch_y) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # Calculate test error

        if epoch % 100 == 0:
            trainmm_error = calculate_error(X_train, y_train, batch_size, model, device)
            test_error = calculate_error(X_test, y_test, batch_size, model, device)
            best_test_error = min(best_test_error, test_error)
            print(f"Epoch {epoch}, Current Test Error: {test_error:.6f}, Best Test Error: {best_test_error:.6f}")
            print(f"Training Error: {trainmm_error:.6f}")

    # Get final training error
    with torch.no_grad():
        final_train_error = calculate_error(X_train, y_train, batch_size, model, device)
        final_test_error = calculate_error(X_test, y_test, batch_size, model, device)

    return best_test_error, final_test_error, initial_train_error, final_train_error

#!/usr/bin/env python3
# [Previous imports remain the same...]

# [Previous class definitions (MSPFunction, DeepNN) and helper functions remain the same...]

def main():
    # Set deterministic behavior
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.set_default_dtype(torch.float32)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Parameters
    experiment_name = "msp_NN_gpu_test"
    P = 8
    d = 30
    hidden_size = 800 # Made iteratable again
    depth = 4  # Made iteratable again
    n_test = 1000
    batch_size = 64
    epochs = 5000
    lr = 0.001
    weight_decay = 1e-4
    mode = 'standard'  # or 'standard'

    # Define the specific MSP sets
    msp_sets = [{7}, {2, 7}, {0, 2, 7}, {4, 5, 7}, {1}, {0, 4}, {3, 7}, {0, 1, 2, 3, 4, 6, 7}]

    # Create results directory if it doesn't exist
    results_dir = f"stair_function/results/{experiment_name}"
    os.makedirs(results_dir, exist_ok=True)

    # Save hyperparameters
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    hyperparams = {
        'P': P,
        'd': d,
        'hidden_size': hidden_size,
        'depth': depth,
        'n_test': n_test,
        'batch_size': batch_size,
        'epochs': epochs,
        'lr': lr,
        'weight_decay': weight_decay,
        'mode': mode,
        'n_train': n_train,
        'msp_sets': [list(s) for s in msp_sets],
        'device': str(device)
    }

    hyperparams_path = os.path.join(results_dir, f'hyperparameters_{timestamp}.json')
    with open(hyperparams_path, 'w') as f:
        json.dump(hyperparams, f, indent=4)

    # Initialize MSP function
    msp = MSPFunction(P, msp_sets)

    # Generate test data
    test_data_path = f'{results_dir}/test_data.pkl'
    if os.path.exists(test_data_path):
        with open(test_data_path, 'rb') as f:
            X_test, y_test = pickle.load(f)
    else:
        X_test = (2 * torch.bernoulli(0.5 * torch.ones((n_test, d), dtype=torch.float32)) - 1)
        y_test = msp.evaluate(X_test)
        with open(test_data_path, 'wb') as f:
            pickle.dump((X_test, y_test), f)

    # Results storage
    results = []

    # Iterate over architectures
    print(f"\nProcessing architecture: hidden_size={hidden_size}, depth={depth}")

    # Iterate over training sizes
    print(f"\nProcessing n_train = {n_train}")

    # Generate training data for this size
    X_train = (2 * torch.bernoulli(0.5 * torch.ones((n_train, d), dtype=torch.float32)) - 1)
    y_train = msp.evaluate(X_train)

    with open(f'{results_dir}/train_data_{n_train}_{timestamp}.pkl', 'wb') as f:
        pickle.dump((X_train, y_train), f)

    # Save results at the start of each n_train iteration
    save_results(results, results_dir, timestamp)

    # Train for different learning rates
    print(f"\nTraining with learning rate = {lr}")

    # Initialize and save initial model
    initial_model = DeepNN(d, hidden_size, depth, mode=mode)
    model_prefix = f'h{hidden_size}_d{depth}_n{n_train}_lr{lr}_{mode}'
    initial_model_path = os.path.join(results_dir, f'initial_model_{model_prefix}_{timestamp}.pt')
    save_model(initial_model, initial_model_path)

    # Train model
    model = DeepNN(d, hidden_size, depth, mode=mode)
    model.load_state_dict(initial_model.state_dict())
    model = model.to(device)
    best_test_error, final_test_error, initial_train_error, final_train_error = train_and_evaluate(
        model, msp, X_train, y_train, X_test, y_test,
        batch_size, epochs, lr, weight_decay, mode, device
    )

    # Save final model
    final_model_path = os.path.join(results_dir, f'final_model_{model_prefix}_{timestamp}.pt')
    save_model(model, final_model_path)

    # Store and save results
    result = {
        'hidden_size': hidden_size,
        'depth': depth,
        'n_train': n_train,
        'learning_rate': lr,
        'mode': mode,
        'best_test_error': best_test_error,
        'final_test_error': final_test_error,
        'initial_train_error': initial_train_error,
        'final_train_error': final_train_error
    }
    results.append(result)
    save_results(results, results_dir, model_prefix)

    print(f"Results saved for h{hidden_size}_d{depth}, n_train = {n_train}, lr = {lr}, mode = {mode}")
    print(f"Best Test Error: {best_test_error:.6f}")
    print(f"Final Test Error: {final_test_error:.6f}")
    print(f"Initial Train Error: {initial_train_error:.6f}")
    print(f"Final Train Error: {final_train_error:.6f}")

if __name__ == "__main__":
    main()