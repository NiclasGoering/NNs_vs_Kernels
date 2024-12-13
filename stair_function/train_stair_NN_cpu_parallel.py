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
from mpi4py import MPI

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

def save_results(results: List[dict], results_dir: str, timestamp: str):
    """Helper function to save results with error handling"""
    try:
        results_path = os.path.join(results_dir, f'results_{timestamp}.json')
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
                      mode: str) -> Tuple[float, float, float]:
    
    """Train the neural network and return best test error and training errors"""
    if mode == 'special':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_test_error = float('inf')
    
    # Get initial training error
    with torch.no_grad():
        initial_train_pred = model(X_train)
        initial_train_error = torch.mean((initial_train_pred - y_train) ** 2).item()
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
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
    
    # Get final training error
    with torch.no_grad():
        final_train_pred = model(X_train)
        final_train_error = torch.mean((final_train_pred - y_train) ** 2).item()
    
    return best_test_error, initial_train_error, final_train_error

def get_parameter_combinations(hidden_sizes, depths, n_train_sizes, learning_rates):
    """Generate all possible hyperparameter combinations"""
    combinations = []
    for hidden_size in hidden_sizes:
        for depth in depths:
            for n_train in n_train_sizes:
                for lr in learning_rates:
                    combinations.append({
                        'hidden_size': hidden_size,
                        'depth': depth,
                        'n_train': n_train,
                        'lr': lr
                    })
    return combinations

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Set deterministic behavior
    torch.manual_seed(42 + rank)  # Different seed for each worker
    np.random.seed(42 + rank)
    random.seed(42 + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    torch.set_default_dtype(torch.float32)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if rank == 0:
        print(f"Master process using device: {device}")
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42 + rank)
    
    # Parameters (same as before)
    experiment_name = "msp_NN_grid_1012_big grid"
    P = 8 
    d = 30
    hidden_sizes = [10,20,30,40,50,75,85,100,120,150,200,300,400,500,600,800,1000,1500,2000,3000,4000]
    depths = [1,2,4,6,8]
    n_test = 1000
    batch_size = 64
    epochs = 5000
    learning_rates = [0.00005,0.0001,0.0003,0.0005,0.0008,0.001,0.003,0.005,0.008,0.01]
    weight_decay = 1e-4
    mode = 'standard'
    n_train_sizes = [10,50,100,200,300,400,500,800,1000,2500,5000,8000,10000,15000,20000]
    
    # Define MSP sets
    msp_sets = [{7},{2,7},{0,2,7},{5,7,4},{1},{0,4},{3,7},{0,1,2,3,4,6,7}]
    
    # Create results directory if doesn't exist (only master process)
    results_dir = f"stair_function/results/{experiment_name}"
    if rank == 0:
        os.makedirs(results_dir, exist_ok=True)
        
        # Save hyperparameters
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        hyperparams = {
            'P': P,
            'd': d,
            'hidden_sizes': hidden_sizes,
            'depths': depths,
            'n_test': n_test,
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rates': learning_rates,
            'weight_decay': weight_decay,
            'mode': mode,
            'n_train_sizes': n_train_sizes,
            'msp_sets': [list(s) for s in msp_sets],
            'device': str(device),
            'num_workers': size
        }
        
        hyperparams_path = os.path.join(results_dir, f'hyperparameters_{timestamp}.json')
        with open(hyperparams_path, 'w') as f:
            json.dump(hyperparams, f, indent=4)
    
    # Broadcast timestamp to all workers
    if rank == 0:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    else:
        timestamp = None
    timestamp = comm.bcast(timestamp, root=0)
    
    # Initialize MSP function
    msp = MSPFunction(P, msp_sets)
    
    # Generate test data (same for all workers)
    X_test = (2 * torch.bernoulli(0.5 * torch.ones((n_test, d), dtype=torch.float32)) - 1).to(device)
    y_test = msp.evaluate(X_test)
    
    # Generate all parameter combinations
    all_combinations = get_parameter_combinations(hidden_sizes, depths, n_train_sizes, learning_rates)
    
    # Distribute work among workers
    worker_combinations = []
    for i in range(len(all_combinations)):
        if i % size == rank:
            worker_combinations.append(all_combinations[i])
    
    # Process combinations assigned to this worker
    results = []
    for params in worker_combinations:
        print(f"Worker {rank} processing: {params}")
        
        # Generate training data
        X_train = (2 * torch.bernoulli(0.5 * torch.ones((params['n_train'], d), dtype=torch.float32)) - 1).to(device)
        y_train = msp.evaluate(X_train)
        
        # Initialize and save initial model
        initial_model = DeepNN(d, params['hidden_size'], params['depth'], mode=mode).to(device)
        model_prefix = f'h{params["hidden_size"]}_d{params["depth"]}_n{params["n_train"]}_lr{params["lr"]}_{mode}'
        initial_model_path = os.path.join(results_dir, f'initial_model_{model_prefix}_{timestamp}_rank{rank}.pt')
        save_model(initial_model, initial_model_path)
        
        # Train model
        model = DeepNN(d, params['hidden_size'], params['depth'], mode=mode).to(device)
        model.load_state_dict(initial_model.state_dict())
        test_error, initial_train_error, final_train_error = train_and_evaluate(
            model, msp, X_train, y_train, X_test, y_test,
            batch_size, epochs, params['lr'], weight_decay, mode
        )
        
        # Save final model
        final_model_path = os.path.join(results_dir, f'final_model_{model_prefix}_{timestamp}_rank{rank}.pt')
        save_model(model, final_model_path)
        
        # Store results
        result = {
            'hidden_size': params['hidden_size'],
            'depth': params['depth'],
            'n_train': params['n_train'],
            'learning_rate': params['lr'],
            'mode': mode,
            'test_error': test_error,
            'initial_train_error': initial_train_error,
            'final_train_error': final_train_error,
            'worker_rank': rank
        }
        results.append(result)
        
        # Save intermediate results for this worker
        worker_results_path = os.path.join(results_dir, f'results_{timestamp}_rank{rank}.json')
        save_results(results, results_dir, f'{timestamp}_rank{rank}')
        
        print(f"Worker {rank} completed {params}")
    
    # Wait for all workers to complete
    comm.Barrier()
    
    # Gather all results to master process
    all_results = comm.gather(results, root=0)
    
    # Master process combines and saves all results
    if rank == 0:
        combined_results = []
        for worker_results in all_results:
            combined_results.extend(worker_results)
        
        final_results_path = os.path.join(results_dir, f'final_results_{timestamp}.json')
        with open(final_results_path, 'w') as f:
            json.dump(combined_results, f, indent=4)
        
        print("All workers completed. Results combined and saved.")

if __name__ == "__main__":
    main()