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

# Define print globally first
print = partial(print, flush=True)

class MSPFunction:
    def __init__(self, P: int, sets: List[Set[int]]):
        self.P = P
        self.sets = sets

        # Verify MSP property
        # for i in range(1, len(sets)):
        #     prev_union = set().union(*sets[:i])
        #     diff = sets[i] - prev_union
        #     if len(diff) > 1:
        #         raise ValueError(f"Not an MSP: Set {sets[i]} adds {len(diff)} new elements: {diff}")

    def evaluate(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[0]
        result = torch.zeros(batch_size, device=z.device)

        for S in self.sets:
            term = torch.ones(batch_size, device=z.device)
            for idx in S:
                term = term * z[:, idx]
            result = result + term

        return result

class DeepNN(nn.Module):
    def __init__(self, d: int, hidden_size: int, depth: int):
        super().__init__()

        layers = []
        prev_dim = d
        for _ in range(depth):
            linear = nn.Linear(prev_dim, hidden_size)
            gain = nn.init.calculate_gain('relu')
            std = gain / np.sqrt(prev_dim)
            nn.init.normal_(linear.weight, mean=0.0, std=std)
            nn.init.zeros_(linear.bias)

            layers.extend([
                linear,
                nn.ReLU()
            ])
            prev_dim = hidden_size

        final_layer = nn.Linear(prev_dim, 1)
        nn.init.normal_(final_layer.weight, std=0.01)
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze()

def train_and_evaluate(model: nn.Module,
                      msp: MSPFunction,
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

    best_test_error = float('inf')

    for epoch in range(epochs):
        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]

            optimizer.zero_grad()
            output = model(batch_X)
            loss = torch.mean((output - batch_y) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # Calculate test error
        with torch.no_grad():
            test_pred = model(X_test)
            trainmm = model(X_train)
            trainmm_error = torch.mean((trainmm - y_train) ** 2).item()
            test_error = torch.mean((test_pred - y_test) ** 2).item()
            best_test_error = min(best_test_error, test_error)

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Current Test Error: {test_error:.6f}, Best Test Error: {best_test_error:.6f}")
            print(f"Training Error: {trainmm_error:.6f}")

    return best_test_error

def main():

    # Parameters
    experiment_name = "msp_learning_test"
    P = 8
    d = 30
    hidden_sizes = [800]
    depths = [4]
    n_test = 1000
    batch_size = 64
    epochs = 3000
    lr = 0.001
    weight_decay = 1e-4
    n_train_sizes = [10, 50, 100, 200, 300, 400, 500, 800, 1000, 5000, 10000, 20000]


    # Define the specific MSP sets
    msp_sets = [{7}, {2, 7}, {0, 2, 7}, {4, 5, 7}, {1}, {0, 4}, {3, 7}, {0, 1, 2, 3, 4, 6, 7}]

    # Create results directory if it doesn't exist
    results_dir = f"stair_function/results/{experiment_name}"
    os.makedirs(results_dir, exist_ok=True)

    # Save hyperparameters
    hyperparams = {
        'P': P,
        'd': d,
        'hidden_sizes': hidden_sizes,
        'depths': depths,
        'n_test': n_test,
        'batch_size': batch_size,
        'epochs': epochs,
        'lr': lr,
        'weight_decay': weight_decay,
        'n_train_sizes': n_train_sizes,
        'msp_sets': [list(s) for s in msp_sets]  # Convert sets to lists for JSON serialization
    }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'{results_dir}/hyperparameters_{timestamp}.json', 'w') as f:
        json.dump(hyperparams, f, indent=4)

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(44)

    # Initialize MSP function
    msp = MSPFunction(P, msp_sets)

    # Generate test data (fixed)
    X_test = 2 * torch.bernoulli(0.5 * torch.ones((n_test, d))) - 1
    y_test = msp.evaluate(X_test)



    # Results storage
    results = []

    # Iterate over architectures and training sizes
    for hidden_size in hidden_sizes:
        for depth in depths:
            print(f"\nProcessing architecture: hidden_size={hidden_size}, depth={depth}")

            for n_train in n_train_sizes:
                print(f"\nProcessing n_train = {n_train}")

                # Generate training data
                X_train = 2 * torch.bernoulli(0.5 * torch.ones((n_train, d))) - 1
                y_train = msp.evaluate(X_train)

                # Initialize and save initial model
                initial_model = DeepNN(d, hidden_size, depth)
                model_prefix = f'h{hidden_size}_d{depth}_n{n_train}'
                torch.save(initial_model.state_dict(),
                         f'{results_dir}/initial_model_{model_prefix}_{timestamp}.pt')

                # Train model
                model = DeepNN(d, hidden_size, depth)
                model.load_state_dict(initial_model.state_dict())  # Start from same initialization
                sgd_test_error = train_and_evaluate(
                    model, msp, X_train, y_train, X_test, y_test,
                    batch_size, epochs, lr, weight_decay
                )

                # Save final model
                torch.save(model.state_dict(),
                         f'{results_dir}/final_model_{model_prefix}_{timestamp}.pt')

                # Store results
                result = {
                    'hidden_size': hidden_size,
                    'depth': depth,
                    'n_train': n_train,
                    'sgd_test_error': sgd_test_error
                }
                results.append(result)

                # Save results after each iteration
                with open(f'{results_dir}/results_{timestamp}.json', 'w') as f:
                    json.dump(results, f, indent=4)

                print(f"Results saved for architecture h{hidden_size}_d{depth}, n_train = {n_train}")
                print(f"SGD Test Error: {sgd_test_error:.6f}")

if __name__ == "__main__":
    main()