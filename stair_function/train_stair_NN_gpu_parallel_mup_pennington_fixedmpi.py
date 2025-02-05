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
    def __init__(self, P: int, sets: List[Set[int]], device=None):
        self.P = P
        self.sets = sets
        self.device = device
    
    def to(self, device):
        """Allow moving MSP function to a device"""
        self.device = device
        return self
    
    def evaluate(self, z: torch.Tensor) -> torch.Tensor:
        device = self.device if self.device is not None else z.device
        batch_size = z.shape[0]
        result = torch.zeros(batch_size, dtype=torch.float32, device=device)
        
        for S in self.sets:
            term = torch.ones(batch_size, dtype=torch.float32, device=device)
            for idx in S:
                term = term * z[:, idx]
            result = result + term
            
        return result

def generate_master_dataset(P, d, master_size, n_test, msp, seed=42, device=None):
    """Generate master training set and test set with fixed seed"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate master training set
    X_train_master = (2 * torch.bernoulli(0.5 * torch.ones((master_size, d), dtype=torch.float32)) - 1).to(device)
    y_train_master = msp.evaluate(X_train_master)
    
    # Generate test set
    X_test = (2 * torch.bernoulli(0.5 * torch.ones((n_test, d), dtype=torch.float32)) - 1).to(device)
    y_test = msp.evaluate(X_test)
    
    return X_train_master, y_train_master, X_test, y_test

class DeepNN(nn.Module):
    def __init__(self, d: int, hidden_size: int, depth: int, mode: str = 'special'):
        super().__init__()
        
        torch.set_default_dtype(torch.float32)
        
        self.mode = mode
        self.depth = depth
        self.hidden_size = hidden_size
        self.input_dim = d
        
        layers = []
        prev_dim = d
        self.layer_lrs = []  # Store layerwise learning rates
        
        for layer_idx in range(depth):
            linear = nn.Linear(prev_dim, hidden_size)
            
            if mode == 'special':
                # Special initialization as in original code
                gain = nn.init.calculate_gain('relu')
                std = gain / np.sqrt(prev_dim)
                nn.init.normal_(linear.weight, mean=0.0, std=std)
                nn.init.zeros_(linear.bias)
                self.layer_lrs.append(1.0)
                
            elif mode == 'spectral':
                # Implement spectral initialization
                fan_in = prev_dim
                fan_out = hidden_size
                std = (1.0 / np.sqrt(fan_in)) * min(1.0, np.sqrt(fan_out / fan_in))
                nn.init.normal_(linear.weight, mean=0.0, std=std)
                nn.init.zeros_(linear.bias)
                self.layer_lrs.append(float(fan_out) / fan_in)
                
            elif mode == 'mup_pennington':
                # muP initialization and learning rates from the paper
                if layer_idx == 0:  # Embedding layer
                    std = 1.0 / np.sqrt(prev_dim)
                    lr_scale = 1.0  # O(1) learning rate for embedding
                else:  # Hidden layers
                    std = 1.0 / np.sqrt(prev_dim)
                    lr_scale = 1.0 / prev_dim  # O(1/n) learning rate for hidden
                nn.init.normal_(linear.weight, mean=0.0, std=std)
                nn.init.zeros_(linear.bias)
                self.layer_lrs.append(lr_scale)
                
            else:  # standard
                nn.init.xavier_uniform_(linear.weight)
                nn.init.zeros_(linear.bias)
                self.layer_lrs.append(1.0)
            
            layers.extend([
                linear,
                nn.ReLU()
            ])
            prev_dim = hidden_size
        
        # Final layer
        final_layer = nn.Linear(prev_dim, 1)
        if mode == 'special':
            nn.init.normal_(final_layer.weight, std=0.01)
            self.layer_lrs.append(1.0)
        elif mode == 'spectral':
            fan_in = prev_dim
            fan_out = 1
            std = (1.0 / np.sqrt(fan_in)) * min(1.0, np.sqrt(fan_out / fan_in))
            nn.init.normal_(final_layer.weight, std=std)
            self.layer_lrs.append(float(fan_out) / fan_in)
        elif mode == 'mup_pennington':
            std = 1.0 / np.sqrt(prev_dim)
            lr_scale = 1.0 / prev_dim  # O(1/n) learning rate for readout
            nn.init.normal_(final_layer.weight, std=std)
            self.layer_lrs.append(lr_scale)
        else:
            nn.init.xavier_uniform_(final_layer.weight)
            self.layer_lrs.append(1.0)
            
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze()
    
    def get_layer_learning_rates(self, base_lr: float) -> List[float]:
        """Return list of learning rates for each layer"""
        return [base_lr * lr for lr in self.layer_lrs]

def create_layer_specific_optimizer(model: DeepNN, base_lr: float, weight_decay: float):
    """Create optimizer with layer-specific learning rates"""
    if model.mode not in ['spectral', 'mup_pennington']:
        if model.mode == 'special':
            return optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
        else:
            return optim.SGD(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    
    # For spectral and muP modes, create parameter groups with different learning rates
    layer_lrs = model.get_layer_learning_rates(base_lr)
    param_groups = []
    
    linear_layer_idx = 0
    for name, param in model.named_parameters():
        if 'weight' in name or 'bias' in name:
            param_groups.append({
                'params': [param],
                'lr': layer_lrs[linear_layer_idx // 2],  # Integer division because we have weight+bias for each layer
                'weight_decay': weight_decay
            })
            if 'bias' in name:
                linear_layer_idx += 1
    
    # Use Adam for muP mode, SGD for spectral mode
    if model.mode == 'mup_pennington':
        return optim.Adam(param_groups, lr=base_lr)
    else:
        return optim.SGD(param_groups, lr=base_lr)

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
                      mode: str,
                      accumulation_steps: int = 4) -> Tuple[float, float, float, dict]:
    """Train with gradient accumulation to manage memory"""
    optimizer = create_layer_specific_optimizer(model, lr, weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # Get initial training error
    with torch.no_grad():
        initial_train_pred = model(X_train)
        initial_test_pred = model(X_test)
        initial_train_error = torch.mean((initial_train_pred - y_train) ** 2).item()
        initial_test_error = torch.mean((initial_test_pred - y_test) ** 2).item()
        
        print(f"Initial predictions stats:")
        print(f"Train - mean: {torch.mean(initial_train_pred):.6f}, std: {torch.std(initial_train_pred):.6f}")
        print(f"Test  - mean: {torch.mean(initial_test_pred):.6f}, std: {torch.std(initial_test_pred):.6f}")
    
    # Initialize error history
    error_history = {
        'train_errors': [],
        'test_errors': [],
        'epochs': []
    }
    
    best_test_error = float('inf')
    
    for epoch in range(epochs):
        optimizer.zero_grad()  # Zero gradients at start of epoch
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            output = model(batch_X)
            loss = torch.mean((output - batch_y) ** 2) / accumulation_steps
            loss.backward()
            
            if (i // batch_size + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()  # Clear GPU memory
        
        # Handle remaining gradients
        if (len(X_train) // batch_size) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
            
        scheduler.step()
        
        # Calculate errors
        with torch.no_grad():
            test_pred = model(X_test)
            train_pred = model(X_train)
            train_error = torch.mean((train_pred - y_train) ** 2).item()
            test_error = torch.mean((test_pred - y_test) ** 2).item()
            best_test_error = min(best_test_error, test_error)
            
            error_history['train_errors'].append(train_error)
            error_history['test_errors'].append(test_error)
            error_history['epochs'].append(epoch)
            
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Current Test Error: {test_error:.6f}, Best Test Error: {best_test_error:.6f}")
            print(f"Training Error: {train_error:.6f}")
            
        torch.cuda.empty_cache()  # Clear GPU memory after each epoch
    
    # Get final training error
    with torch.no_grad():
        final_train_pred = model(X_train)
        final_train_error = torch.mean((final_train_pred - y_train) ** 2).item()
    
    return best_test_error, initial_train_error, final_train_error, error_history

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

def save_dataset(X: torch.Tensor, y: torch.Tensor, path: str, rank: int, min_size_bytes: int = 1000):
    """Helper function to save dataset with built-in verification"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {
            'X': X.cpu(),
            'y': y.cpu(),
            'shape_X': X.shape,
            'shape_y': y.shape,
            'saved_by_rank': rank,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(save_dict, path)
        
        # Verify the save
        if not os.path.exists(path):
            raise RuntimeError(f"File does not exist after save: {path}")
        if os.path.getsize(path) < min_size_bytes:
            raise RuntimeError(f"File too small after save: {path} ({os.path.getsize(path)} bytes)")
            
        print(f"Rank {rank}: Successfully saved and verified dataset at {path}")
        return True
    except Exception as e:
        print(f"Rank {rank}: Error saving dataset: {e}")
        return False

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Set up GPU device for this process
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        device = torch.device('cpu')
    else:
        gpu_id = rank % n_gpus
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)  # Important: this sets the default CUDA device
    
    if rank == 0:
        print(f"Number of available GPUs: {n_gpus}")
        print(f"Master process using device: {device}")
    
    print(f"Rank {rank} using device: {device}")
    
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Parameters 
    experiment_name = "msp_NN_grid_1201_mup_lr_grid"
    P = 8 
    d = 30
    master_size = 60000  # Size of master training set
    #hidden_sizes =  [10,200,400,500,1000,3000] #5000
    
    hidden_sizes=[50,75,100,2000]
    hidden_sizes.reverse()
    depths = [4,1]
    n_test = 1000
    batch_size = 64
    epochs = 5000
    learning_rates = [1,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001,0.00001]
    weight_decay = 1e-4
    mode = 'mup_pennington'
    n_train_sizes = [10,50,100,200,300,400,500,800,1000,2500,5000,8000,10000,15000,20000,30000,40000]
    n_train_sizes.reverse()
    
    # Define MSP sets
    msp_sets = [{7},{2,7},{0,2,7},{5,7,4},{1},{0,4},{3,7},{0,1,2,3,4,6,7}]
    
    # Create results and data directories for all ranks
    base_path = "/mnt/users/goringn/NNs_vs_Kernels"
    results_dir = os.path.join(base_path, "stair_function/results", experiment_name)
    data_dir = os.path.join(base_path, "stair_function/data", experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate timestamp (rank 0)
    if rank == 0:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save hyperparameters
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
            'num_workers': size,
            'num_gpus': n_gpus
        }
        
        hyperparams_path = os.path.join(results_dir, f'hyperparameters_{timestamp}.json')
        with open(hyperparams_path, 'w') as f:
            json.dump(hyperparams, f, indent=4)
    else:
        timestamp = None
    
    # Broadcast timestamp to all workers
    timestamp = comm.bcast(timestamp, root=0)
    
    # Initialize MSP function and move to correct device
    msp = MSPFunction(P, msp_sets).to(device)
    
    # Master process generates datasets with fixed seed
    if rank == 0:
        print("Master process generating datasets...")
        X_train_master, y_train_master, X_test, y_test = generate_master_dataset(
            P, d, master_size, n_test, msp, seed=42, device=device
        )
        print("Data generation complete")
        print(f"Master set shape: {X_train_master.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        # Save master dataset
        master_data_path = os.path.join(data_dir, f'master_data_{timestamp}.pt')
        print(f"Attempting to save master data to: {os.path.abspath(master_data_path)}")
        if not save_dataset(X_train_master, y_train_master, master_data_path, rank):
            raise RuntimeError("Failed to save master dataset")
        
        # Save test dataset
        test_data_path = os.path.join(data_dir, f'test_data_{timestamp}.pt')
        print(f"Attempting to save test data to: {os.path.abspath(test_data_path)}")
        if not save_dataset(X_test, y_test, test_data_path, rank):
            raise RuntimeError("Failed to save test dataset")
        
        print("Master process saved all datasets successfully")
    else:
        X_train_master = None
        y_train_master = None
        X_test = None
        y_test = None
    
    # Broadcast datasets to all workers
    X_train_master = comm.bcast(X_train_master, root=0)
    y_train_master = comm.bcast(y_train_master, root=0)
    X_test = comm.bcast(X_test, root=0)
    y_test = comm.bcast(y_test, root=0)
    
    # Move broadcast data to correct device for this process
    X_train_master = X_train_master.to(device)
    y_train_master = y_train_master.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
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
        print(f"Worker {rank} processing on device {device}: {params}")
        torch.cuda.empty_cache()  # Clear GPU memory before starting new model
        
        # Sample from master training set using fixed seed for this configuration
        sample_seed = hash(f"sample_{params['n_train']}")
        torch.manual_seed(sample_seed)
        indices = torch.randperm(master_size, device='cpu')[:params['n_train']]  # Create on CPU first
        X_train = X_train_master[indices].to(device)
        y_train = y_train_master[indices].to(device)
        
        # Save training data before proceeding with training
        train_data_path = os.path.join(results_dir, 
            f'train_data_h{params["hidden_size"]}_d{params["depth"]}_n{params["n_train"]}_lr{params["lr"]}_{timestamp}_rank{rank}.pt')
        
        print(f"Rank {rank}: Saving training data before training...")
        if not save_dataset(X_train, y_train, train_data_path, rank):
            print(f"Rank {rank}: Failed to save training data, skipping this configuration")
            continue
            
        print(f"Rank {rank}: Successfully saved training data, proceeding with training")
        print(f"Sampled training set shape: {X_train.shape}")
        print(f"Sample seed used: {sample_seed}")
        
        # Initialize model with new seed
        model_seed = hash(f"model_{datetime.now()}_{rank}")
        torch.manual_seed(model_seed)
        print(f"Model initialization seed: {model_seed}")
        
        # Initialize and save initial model
        initial_model = DeepNN(d, params['hidden_size'], params['depth'], mode=mode).to(device)
        model_prefix = f'h{params["hidden_size"]}_d{params["depth"]}_n{params["n_train"]}_lr{params["lr"]}_{mode}'
        initial_model_path = os.path.join(results_dir, f'initial_model_{model_prefix}_{timestamp}_rank{rank}.pt')
        print(f"Attempting to save model to: {os.path.abspath(initial_model_path)}")
        save_model(initial_model, initial_model_path)
        
        # Train model
        model = DeepNN(d, params['hidden_size'], params['depth'], mode=mode).to(device)
        model.load_state_dict(initial_model.state_dict())
        
        try:
            test_error, initial_train_error, final_train_error, error_history = train_and_evaluate(
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
                'error_history': error_history,
                'worker_rank': rank,
                'device': str(device),
                'sample_seed': sample_seed,
                'model_seed': model_seed,
                'train_data_path': os.path.basename(train_data_path),
                'test_data_path': f'test_data_{timestamp}.pt',
                'initial_model_path': os.path.basename(initial_model_path),
                'final_model_path': os.path.basename(final_model_path)
            }
            results.append(result)
            
            # Save intermediate results
            save_results(results, results_dir, f'{timestamp}_rank{rank}')
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Rank {rank}: Out of memory error for params {params}. Skipping this configuration.")
                torch.cuda.empty_cache()
                continue
            else:
                raise e
        
        print(f"Worker {rank} completed {params}")
        torch.cuda.empty_cache()
    
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