#!/usr/bin/env python3
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Set, Tuple
import random
from functools import partial
import json
from datetime import datetime
from itertools import product
from torch.utils.data import TensorDataset, DataLoader
from mpi4py import MPI
import sys
from datetime import datetime
import os
import torch

# Define print globally first  
print = partial(print, flush=True)

def get_gpu_id(rank, world_size):
    """Map MPI rank to GPU ID"""
    return rank % torch.cuda.device_count()

def distribute_to_gpus(comm, rank, size):
    """
    Distribute work among available GPUs
    Returns device and GPU ID for this process
    """
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        return torch.device('cpu'), None
    
    gpu_id = get_gpu_id(rank, size)
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)
    
    if rank == 0:
        print(f"Found {n_gpus} GPUs")
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f}GB memory")
    
    return device, gpu_id

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
    
    # Add symmetric noise
    noise = noise_std * np.random.choice([-1, 1], size=n_samples)
    y = y + noise
    
    return X, y, U, coeff_vec

def generate_fixed_test_data(ambient_dim, latent_dim, degree, n_test=10000, noise_std=0.1, U=None, coeff_vec=None, random_state=None):
    """
    Generate test set using same U matrix and coefficients as training data
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    X_test = np.random.randn(n_test, ambient_dim)
    
    # Project onto same latent space
    X_test_latent = X_test @ U
    
    # Initialize output
    y_test = np.zeros(n_test)
    
    # Generate all polynomial terms
    terms = generate_polynomials(latent_dim, degree)
    
    # Use same coefficients as training data
    coeff_idx = 0
    for term in terms:
        if sum(term) > 0:  # Skip constant term
            term_value = np.ones(n_test)
            for dim, power in enumerate(term):
                if power > 0:
                    term_value *= X_test_latent[:, dim] ** power
            y_test += coeff_vec[coeff_idx] * term_value
            coeff_idx += 1
    
    # Add symmetric noise
    noise = noise_std * np.random.choice([-1, 1], size=n_test)
    y_test = y_test + noise
    
    return X_test, y_test

def generate_and_normalize_data(n_train, X_test_fixed, y_test_fixed, ambient_dim, latent_dim, degree, noise_std=0.1, random_state=None):
    """
    Generate training data and normalize both train and test sets using training statistics
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
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test_fixed = torch.FloatTensor(X_test_fixed)
    y_test_fixed = torch.FloatTensor(y_test_fixed)
    
    # Normalize based on training set statistics
    y_train_mean = torch.mean(y_train)
    y_train_std = torch.std(y_train)
    
    y_train_normalized = (y_train - y_train_mean) / y_train_std
    y_test_normalized = (y_test_fixed - y_train_mean) / y_train_std
    
    return X_train, y_train_normalized.reshape(-1, 1), X_test_fixed, y_test_normalized.reshape(-1, 1)



def save_dataset(X: torch.Tensor, y: torch.Tensor, path: str, rank: int, min_size_bytes: int = 1000):
    """Helper function to save dataset with built-in verification
    
    Args:
        X (torch.Tensor): Input features tensor
        y (torch.Tensor): Target values tensor
        path (str): Path where to save the dataset
        rank (int): MPI rank of the process saving the dataset
        min_size_bytes (int): Minimum expected file size in bytes
    
    Returns:
        bool: True if save was successful, False otherwise
    """
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
        return self.network(x).reshape(-1, 1)
    
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

def save_results(results: List[dict], results_dir: str, timestamp: str) -> bool:
    """Helper function to save results with proper error handling"""
    try:
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, f'results_{timestamp}.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving results: {e}")
        raise  # Re-raise the exception to properly handle it

def save_model(model: nn.Module, path: str) -> bool:
    """Helper function to save model with proper error handling"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        original_device = next(model.parameters()).device
        model = model.cpu()
        try:
            torch.save(model.state_dict(), path)
        finally:
            model = model.to(original_device)  # Ensure model returns to original device
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        raise

def train_and_evaluate(model: nn.Module,
                      X_train: torch.Tensor,
                      y_train: torch.Tensor,
                      X_test: torch.Tensor,
                      y_test: torch.Tensor,
                      batch_size: int,
                      epochs: int,
                      lr: float,
                      weight_decay: float,
                      mode: str) -> Tuple[float, float, float, dict]:
    
    """Train the neural network and return best test error, training errors, and error history"""
    device = next(model.parameters()).device  # Get device from model
    optimizer = create_layer_specific_optimizer(model, lr, weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.MSELoss()
    
    best_test_error = float('inf')
    
    # Get initial training error
    with torch.no_grad():
        initial_train_pred = model(X_train)
        initial_train_error = criterion(initial_train_pred, y_train).item()
    
    # Initialize error history
    error_history = {
        'train_errors': [],
        'test_errors': [],
        'epochs': []
    }
    
    # Create data loader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        # Mini-batch training
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        
        # Calculate errors
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            train_pred = model(X_train)
            train_error = criterion(train_pred, y_train).item()
            test_error = criterion(test_pred, y_test).item()
            best_test_error = min(best_test_error, test_error)
            
            # Save errors for this epoch
            error_history['train_errors'].append(train_error)
            error_history['test_errors'].append(test_error)
            error_history['epochs'].append(epoch)
            
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Current Test Error: {test_error:.6f}, Best Test Error: {best_test_error:.6f}")
            print(f"Training Error: {train_error:.6f}")
    
    # Get final training error
    with torch.no_grad():
        final_train_pred = model(X_train)
        final_train_error = criterion(final_train_pred, y_train).item()
    
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
def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Set device to specific GPU based on rank
    n_gpus = torch.cuda.device_count()
    if n_gpus > 0:
        gpu_id = rank % n_gpus
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    
    if rank == 0:
        print(f"Number of available GPUs: {n_gpus}")
        print(f"Master process using device: {device}")
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Parameters
    experiment_name = "low_dim_poly_prune_2801_false_lr_localnorm"
    ambient_dim = 20
    latent_dim = 3
    degree = 5
    noise_std = 0.0
    hidden_sizes = [400,2000]
    hidden_sizes.reverse()
    depths = [4,8]
    batch_size = 128
    epochs = 5000
    learning_rates = [0.01,0.001,0.05,0.0001]
    weight_decay = 1e-4
    mode = 'mup_pennington'
    n_train_sizes = [10,100,500,1000,2500,5000,10000,20000,40000]
    n_train_sizes.reverse()
    max_train_size = max(n_train_sizes)
    shuffle = False  # New parameter for shuffling labels
    
    # Master process generates the data with fixed seed
    if rank == 0:
        print("Master process generating initial polynomial...")
        torch.manual_seed(42)
        np.random.seed(42)
        try:
            # Generate full training set
            X_train_full, y_train_full, U, coeff_vec = generate_latent_poly_data(
                n_samples=max_train_size,
                ambient_dim=ambient_dim,
                latent_dim=latent_dim,
                degree=degree,
                noise_std=noise_std,
                random_state=42
            )
            
            # Calculate global normalization statistics
            global_mean = float(np.mean(y_train_full))
            global_std = float(np.std(y_train_full))
            
            # Generate fixed test set using same polynomial
            X_test_fixed, y_test_fixed = generate_fixed_test_data(
                ambient_dim=ambient_dim,
                latent_dim=latent_dim,
                degree=degree,
                n_test=10000,
                noise_std=noise_std,
                U=U,
                coeff_vec=coeff_vec,
                random_state=42
            )
            print(f"Generated master training set of size {max_train_size}")
            print(f"Global normalization stats - mean: {global_mean:.6f}, std: {global_std:.6f}")
        except Exception as e:
            print(f"Error generating data: {e}")
            comm.Abort()
    else:
        X_train_full = None
        y_train_full = None
        X_test_fixed = None
        y_test_fixed = None
        global_mean = None
        global_std = None

    # Broadcast datasets and normalization statistics to all workers
    X_train_full = comm.bcast(X_train_full, root=0)
    y_train_full = comm.bcast(y_train_full, root=0)
    X_test_fixed = comm.bcast(X_test_fixed, root=0)
    y_test_fixed = comm.bcast(y_test_fixed, root=0)
    global_mean = comm.bcast(global_mean, root=0)
    global_std = comm.bcast(global_std, root=0)
    
    # Create results directory and save hyperparameters (only master process)
    results_dir = f"/mnt/users/goringn/NNs_vs_Kernels/low_dim_poly/results/{experiment_name}"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') if rank == 0 else None
    timestamp = comm.bcast(timestamp, root=0)
    
    if rank == 0:
        try:
            os.makedirs(results_dir, exist_ok=True)
            
            # Save hyperparameters
            hyperparams = {
                'ambient_dim': ambient_dim,
                'latent_dim': latent_dim,
                'degree': degree,
                'noise_std': noise_std,
                'hidden_sizes': hidden_sizes,
                'depths': depths,
                'batch_size': batch_size,
                'epochs': epochs,
                'learning_rates': learning_rates,
                'weight_decay': weight_decay,
                'mode': mode,
                'n_train_sizes': n_train_sizes,
                'device': str(device),
                'num_workers': size,
                'num_gpus': n_gpus,
                'global_mean': global_mean,
                'global_std': global_std,
                'shuffle': shuffle  # Add shuffle parameter to hyperparams
            }
            
            hyperparams_path = os.path.join(results_dir, f'hyperparameters_{timestamp}.json')
            with open(hyperparams_path, 'w') as f:
                json.dump(hyperparams, f, indent=4)
        except Exception as e:
            print(f"Error saving hyperparameters: {e}")
            comm.Abort()
    
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
        print(f"Worker {rank} on {device} processing: {params}")
        
        try:
            # Use fixed seed for sampling
            sample_seed = hash(f"sample_{params['n_train']}") % (2**32)
            torch.manual_seed(sample_seed)
            np.random.seed(sample_seed)
            print(f"Using sample seed: {sample_seed}")
            
            # Sample training data
            indices = np.random.choice(max_train_size, size=params['n_train'], replace=False)
            X_train = X_train_full[indices]
            y_train = y_train_full[indices]
            
            if shuffle:
                # Calculate statistics of original y_train
                orig_mean = np.mean(y_train)
                orig_std = np.std(y_train)
                
                # Generate random Gaussian noise with same statistics
                np.random.seed(sample_seed)  # Use same seed for reproducibility
                y_train = np.random.normal(orig_mean, orig_std, size=y_train.shape)
            
            # Normalize using global statistics
            y_train_normalized = (y_train - global_mean) / global_std
            y_test_normalized = (y_test_fixed - global_mean) / global_std
            
            # Convert to PyTorch tensors
            X_train = torch.FloatTensor(X_train).to(device)
            y_train = torch.FloatTensor(y_train_normalized).reshape(-1, 1).to(device)
            X_test = torch.FloatTensor(X_test_fixed).to(device)
            y_test = torch.FloatTensor(y_test_normalized).reshape(-1, 1).to(device)
            
            # Model initialization
            model_seed = hash(f"model_{datetime.now()}_{rank}") % (2**32)
            torch.manual_seed(model_seed)
            
            # Create model prefix for file naming
            model_prefix = f'h{params["hidden_size"]}_d{params["depth"]}_n{params["n_train"]}_lr{params["lr"]}_{mode}'
            if shuffle:
                model_prefix += '_shuffled'
            
            # Save initial model
            initial_model = DeepNN(ambient_dim, params['hidden_size'], params['depth'], mode=mode).to(device)
            initial_model_path = os.path.join(results_dir, f'initial_model_{model_prefix}_{timestamp}_rank{rank}.pt')
            
            try:
                save_model(initial_model, initial_model_path)
            except Exception as e:
                print(f"Worker {rank}: Failed to save initial model: {e}")
                continue
                
            # Save dataset
            dataset_path = os.path.join(results_dir, f'dataset_{model_prefix}_{timestamp}_rank{rank}.pt')
            try:
                save_dataset(X_train, y_train, dataset_path, rank)
            except Exception as e:
                print(f"Worker {rank}: Failed to save dataset: {e}")
            
            # Train model
            model = DeepNN(ambient_dim, params['hidden_size'], params['depth'], mode=mode).to(device)
            model.load_state_dict(initial_model.state_dict())
            
            test_error, initial_train_error, final_train_error, error_history = train_and_evaluate(
                model, X_train, y_train, X_test, y_test,
                batch_size, epochs, params['lr'], weight_decay, mode
            )
            
            # Save final model
            final_model_path = os.path.join(results_dir, f'final_model_{model_prefix}_{timestamp}_rank{rank}.pt')
            try:
                save_model(model, final_model_path)
            except Exception as e:
                print(f"Worker {rank}: Failed to save final model: {e}")
            
            # Store results
            result = {
                'hidden_size': params['hidden_size'],
                'depth': params['depth'],
                'n_train': params['n_train'],
                'learning_rate': params['lr'],
                'mode': mode,
                'shuffle': shuffle,  # Add shuffle to results
                'test_error': test_error,
                'initial_train_error': initial_train_error,
                'final_train_error': final_train_error,
                'error_history': error_history,
                'worker_rank': rank,
                'sample_seed': sample_seed,
                'model_seed': model_seed,
                'global_mean': float(global_mean),
                'global_std': float(global_std),
                'device': str(device)
            }
            results.append(result)
            
            # Save intermediate results
            try:
                save_results(results, results_dir, f'{timestamp}_rank{rank}')
            except Exception as e:
                print(f"Worker {rank}: Failed to save intermediate results: {e}")
            
            print(f"Worker {rank} completed {params}")
            
        except Exception as e:
            print(f"Worker {rank}: Error processing parameters {params}: {e}")
            continue
    
    # Synchronize workers
    comm.Barrier()
    
    # Gather results
    all_results = comm.gather(results, root=0)
    
    # Master process combines and saves final results
    if rank == 0:
        try:
            combined_results = []
            for worker_results in all_results:
                combined_results.extend(worker_results)
            
            # Save to a temporary file first
            final_results_path = os.path.join(results_dir, f'final_results_{timestamp}.json')
            temp_path = final_results_path + '.tmp'
            
            # Write to temporary file
            with open(temp_path, 'w') as f:
                json.dump(combined_results, f, indent=4)
            
            # Rename temporary file to final file (atomic operation on most filesystems)
            os.replace(temp_path, final_results_path)
            
            print("All workers completed. Results combined and saved.")
        except Exception as e:
            print(f"Error saving final results: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
