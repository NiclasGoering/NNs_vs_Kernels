#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Set, Tuple
import matplotlib.pyplot as plt
import random
import jax
import jax.numpy as jnp
import neural_tangents as nt
from jax import random as jrandom


from functools import partial

# Define print globally first
print = partial(print, flush=True)


class MSPFunction:
    def __init__(self, P: int, sets: List[Set[int]]):
        self.P = P
        self.sets = sets
            
        # Verify MSP property
        for i in range(1, len(sets)):
            prev_union = set().union(*sets[:i])
            diff = sets[i] - prev_union
            if len(diff) > 1:
                raise ValueError(f"Not an MSP: Set {sets[i]} adds {len(diff)} new elements: {diff}")
    
    def evaluate(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[0]
        result = torch.zeros(batch_size, device=z.device)
        
        for S in self.sets:
            term = torch.ones(batch_size, device=z.device)
            for idx in S:
                term = term * z[:, idx]
            result = result + term
            
        return result

def generate_random_msp(P: int) -> List[Set[int]]:
    sets = []
    num_sets = random.randint(1, P)
    
    size = random.randint(1, min(3, P))
    sets.append(set(random.sample(range(P), size)))
    
    for _ in range(num_sets - 1):
        prev_union = set().union(*sets)
        remaining = set(range(P)) - prev_union
        
        if remaining and random.random() < 0.7:
            new_elem = random.choice(list(remaining))
            base_elems = random.sample(list(prev_union), random.randint(0, len(prev_union)))
            new_set = set(base_elems + [new_elem])
        else:
            size = random.randint(1, len(prev_union))
            new_set = set(random.sample(list(prev_union), size))
        
        sets.append(new_set)
    
    return sets

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
def train(model: nn.Module, 
          msp: MSPFunction,
          d: int,
          n: int,
          batch_size: int,
          epochs: int,
          lr: float,
          weight_decay: float = 0.0) -> Tuple[List[float], List[float]]:
    """Train the neural network"""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    train_losses = []
    test_errors = []
    
    for epoch in range(epochs):
        # Generate fresh samples 
        X = 2 * torch.bernoulli(0.5 * torch.ones((n, d))) - 1
        y = msp.evaluate(X)
        
        epoch_losses = []
        # Mini-batch training
        for i in range(0, n, batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            
            optimizer.zero_grad()
            output = model(batch_X)
            loss = torch.mean((output - batch_y) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        scheduler.step()
        
        if epoch % 10 == 0:
            avg_loss = np.mean(epoch_losses)
            train_losses.append(avg_loss)
            test_X = 2 * torch.bernoulli(0.5 * torch.ones((1000, d))) - 1
            test_y = msp.evaluate(test_X)
            with torch.no_grad():
                test_pred = model(test_X)
                test_error = torch.mean((test_pred - test_y) ** 2).item()
            test_errors.append(test_error)
            print(f"Epoch {epoch}, Train Loss: {avg_loss:.6f}, Test Error: {test_error:.6f}")
    
    return train_losses, test_errors

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
            trainmm=model(X_train)
            trainmm_error = torch.mean((trainmm - y_train) ** 2).item()
            test_error = torch.mean((test_pred - y_test) ** 2).item()
            best_test_error = min(best_test_error, test_error)
            
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Current Test Error: {test_error:.6f}, Best Test Error: {best_test_error:.6f}")
            print(trainmm_error)
    
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

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Set, Tuple
import random
import jax
import jax.numpy as jnp
import neural_tangents as nt
from jax import random as jrandom
import json
from datetime import datetime
import os



def main():
    # Parameters
    P = 9
    d = 54
    hidden_dims = [1000, 500, 250, 100]
    n_test = 1000
    batch_size = 64
    epochs = 5000
    lr = 0.001
    weight_decay = 1e-4
    n_train_sizes = [10, 50, 100, 200, 300, 400, 500, 800, 1000, 5000, 10000, 20000]
    
    # Save hyperparameters
    hyperparams = {
        'P': P,
        'd': d,
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
    random.seed(41)
    key = jrandom.PRNGKey(42)
    
    # Generate random MSP function
    sets = generate_random_msp(P)
    msp = MSPFunction(P, sets)
    
    # Generate test data (fixed)
    X_test = 2 * torch.bernoulli(0.5 * torch.ones((n_test, d))) - 1
    y_test = msp.evaluate(X_test)
    
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
        X_train = 2 * torch.bernoulli(0.5 * torch.ones((n_train, d))) - 1
        y_train = msp.evaluate(X_train)
        
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
            model, msp, X_train, y_train, X_test, y_test,
            batch_size, epochs, lr, weight_decay
        )
        
        # Calculate test errors
        nngp_matched_error = float(((y_nngp_matched.squeeze() - y_test_jax.squeeze())**2).mean())
        ntk_matched_error = float(((y_ntk_matched.squeeze() - y_test_jax.squeeze())**2).mean())
        nngp_random_error = float(((y_nngp_random.squeeze() - y_test_jax.squeeze())**2).mean())
        ntk_random_error = float(((y_ntk_random.squeeze() - y_test_jax.squeeze())**2).mean())
        
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
        with open(f'stair_function/results_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Results saved for n_train = {n_train}")
        print(f"SGD Test Error: {sgd_test_error:.6f}")
        print(f"NNGP (matched) Error: {nngp_matched_error:.6f}")
        print(f"NTK (matched) Error: {ntk_matched_error:.6f}")
        print(f"NNGP (random) Error: {nngp_random_error:.6f}")
        print(f"NTK (random) Error: {ntk_random_error:.6f}")

if __name__ == "__main__":
    main()