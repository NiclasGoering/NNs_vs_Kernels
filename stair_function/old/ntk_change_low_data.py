#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Set, Tuple, Dict
import random
from functools import partial
import json
from datetime import datetime
import os
from mpi4py import MPI
from torch.func import functional_call, vmap, vjp, jvp, jacrev
import copy
import h5py

# Define print globally first  
print = partial(print, flush=True)

def shuffle_labels(y_train: torch.Tensor, seed: int = None) -> torch.Tensor:
    """Shuffle the training labels randomly"""
    if seed is not None:
        torch.manual_seed(seed)
    perm = torch.randperm(y_train.size(0))
    return y_train[perm]

# 2. Modify MSPFunction class to match the simpler version
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


def generate_master_dataset(P, d, master_size, n_test, msp, seed=42):
    """Generate master training set and test set with fixed seed"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)  # Add this for shuffling seed
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_train_master = (2 * torch.bernoulli(0.5 * torch.ones((master_size, d), dtype=torch.float32)) - 1).to(device)
    y_train_master = msp.evaluate(X_train_master)
    
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
        
    def compute_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        features = x
        for layer in list(self.network.children())[:-1]:
            features = layer(features)
        return features
        
    def get_layer_learning_rates(self, base_lr: float) -> List[float]:
        """Return list of learning rates for each layer"""
        return [base_lr * lr for lr in self.layer_lrs]

def compute_empirical_kernels(model: nn.Module, X1: torch.Tensor, X2: torch.Tensor, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """Compute empirical NNGP and NTK kernels using the Jacobian contraction method"""
    model.eval()
    device = X1.device
    
    def fnet_single(params, x):
        return functional_call(model, params, (x.unsqueeze(0),)).squeeze(0)
    
    def compute_kernel_batch(x1_batch: torch.Tensor, x2_batch: torch.Tensor):
        # Get parameters dictionary
        params = {k: v for k, v in model.named_parameters()}
        
        # Compute Jacobians
        jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1_batch)
        jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2_batch)
        
        # Process Jacobians - properly handle the dimensions
        jac1_flat = []
        jac2_flat = []
        for j1, j2 in zip(jac1.values(), jac2.values()):
            j1_shape = j1.shape
            j2_shape = j2.shape
            j1_reshaped = j1.reshape(j1_shape[0], -1)  # Flatten all dimensions after the first
            j2_reshaped = j2.reshape(j2_shape[0], -1)
            jac1_flat.append(j1_reshaped)
            jac2_flat.append(j2_reshaped)
        
        # Compute NTK by calculating the inner product of the flattened Jacobians
        ntk_result = sum(torch.matmul(j1, j2.t()) for j1, j2 in zip(jac1_flat, jac2_flat))
        
        # Compute NNGP (feature map inner product)
        with torch.no_grad():
            feat1 = model.compute_feature_map(x1_batch)
            feat2 = model.compute_feature_map(x2_batch)
            nngp_result = torch.matmul(feat1, feat2.T) / feat1.shape[1]
        
        return nngp_result.detach(), ntk_result.detach()
    
    n1, n2 = X1.shape[0], X2.shape[0]
    nngp = torch.zeros((n1, n2), device=device)
    ntk = torch.zeros((n1, n2), device=device)
    
    for i in range(0, n1, batch_size):
        i_end = min(i + batch_size, n1)
        for j in range(0, n2, batch_size):
            j_end = min(j + batch_size, n2)
            
            nngp_batch, ntk_batch = compute_kernel_batch(
                X1[i:i_end], 
                X2[j:j_end]
            )
            nngp[i:i_end, j:j_end] = nngp_batch
            ntk[i:i_end, j:j_end] = ntk_batch
    
    return nngp.cpu().numpy(), ntk.cpu().numpy()

def create_layer_specific_optimizer(model: DeepNN, base_lr: float, weight_decay: float):
    """Create optimizer with layer-specific learning rates"""
    if model.mode not in ['spectral', 'mup_pennington']:
        if model.mode == 'special':
            return optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
        else:  # standard
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

def save_kernels(nngp: np.ndarray, ntk: np.ndarray, filename: str, metadata: dict = None):
    """Save kernels to HDF5 file with optional metadata"""
    with h5py.File(filename, 'w') as f:
        f.create_dataset('nngp', data=nngp)
        f.create_dataset('ntk', data=ntk)
        if metadata:
            for key, value in metadata.items():
                f.attrs[key] = value

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
                      save_dir: str,
                      experiment_name: str) -> Tuple[float, float, float, dict]:
    
    # Compute and save initial kernels
    print("Computing initial kernels...")
    initial_nngp, initial_ntk = compute_empirical_kernels(model, X_train, X_train)
    kernel_filename = os.path.join(save_dir, f'{experiment_name}_initial_kernels.h5')
    save_kernels(initial_nngp, initial_ntk, kernel_filename, {'epoch': 0})
    print(f"Initial kernels saved to {kernel_filename}")
    
    optimizer = create_layer_specific_optimizer(model, lr, weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_test_error = float('inf')
    best_model_state = None
    best_epoch = -1
    
    with torch.no_grad():
        initial_train_pred = model(X_train)
        initial_test_pred = model(X_test)
        initial_train_error = torch.mean((initial_train_pred - y_train) ** 2).item()
        initial_test_error = torch.mean((initial_test_pred - y_test) ** 2).item()
        
        print(f"Initial predictions stats:")
        print(f"Train - mean: {torch.mean(initial_train_pred):.6f}, std: {torch.std(initial_train_pred):.6f}")
        print(f"Test  - mean: {torch.mean(initial_test_pred):.6f}, std: {torch.std(initial_test_pred):.6f}")
    
    error_history = {
        'train_errors': [],
        'test_errors': [],
        'epochs': [],
        'best_epoch': None
    }
    
    for epoch in range(epochs):
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
        
        with torch.no_grad():
            test_pred = model(X_test)
            train_pred = model(X_train)
            train_error = torch.mean((train_pred - y_train) ** 2).item()
            test_error = torch.mean((test_pred - y_test) ** 2).item()
            
            # Save best model state and epoch
            if test_error < best_test_error:
                best_test_error = test_error
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
            
            error_history['train_errors'].append(train_error)
            error_history['test_errors'].append(test_error)
            error_history['epochs'].append(epoch)
            
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Test Error: {test_error:.6f}, Best: {best_test_error:.6f}")
            print(f"Training Error: {train_error:.6f}")
    
    # Record best epoch in history
    error_history['best_epoch'] = best_epoch
    
    # Load best model state and compute kernels for it
    model.load_state_dict(best_model_state)
    print(f"Computing kernels for best model (epoch {best_epoch})...")
    final_nngp, final_ntk = compute_empirical_kernels(model, X_train, X_train)
    kernel_filename = os.path.join(save_dir, f'{experiment_name}_best_model_epoch{best_epoch}_kernels.h5')
    save_kernels(final_nngp, final_ntk, kernel_filename, {'epoch': best_epoch})
    print(f"Best model kernels saved to {kernel_filename}")
    
    # Save best model
    model_path = os.path.join(save_dir, f'best_model_epoch{best_epoch}_{experiment_name}.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Best model saved to {model_path}")
    
    with torch.no_grad():
        final_train_pred = model(X_train)
        final_train_error = torch.mean((final_train_pred - y_train) ** 2).item()
    
    return best_test_error, initial_train_error, final_train_error, error_history


def get_parameter_combinations(hidden_sizes, depths, n_train_sizes, learning_rates, modes):
    """Generate all possible hyperparameter combinations"""
    combinations = []
    for hidden_size in hidden_sizes:
        for depth in depths:
            for n_train in n_train_sizes:
                for lr in learning_rates:
                    for mode in modes:
                        combinations.append({
                            'hidden_size': hidden_size,
                            'depth': depth,
                            'n_train': n_train,
                            'lr': lr,
                            'mode': mode
                        })
    return combinations


def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if rank == 0:
        print(f"Master process using device: {device}")
    
    # Parameters 
    experiment_name = "msp_NTK_1401_shuffledtrue"
    P = 8 
    d = 30
    master_size = 40000  # Size of master training set
    hidden_sizes = [400]
    hidden_sizes.reverse()
    depths = [4]
    n_test = 1000
    batch_size = 64
    epochs = 5000
    learning_rates = [0.001]
    weight_decay = 1e-4
    modes = ['mup_pennington']
    shuffled = True  # Add shuffling parameter
    n_train_sizes = [10,50,100,200,500,1000,2000,5000,10000]
    n_train_sizes.reverse()
    
    # Define MSP sets
    msp_sets = [{7},{2,7},{0,2,7},{5,7,4},{1},{0,4},{3,7},{0,1,2,3,4,6,7}]
    
    # Create results directory
    results_dir = f"/mnt/users/goringn/NNs_vs_Kernels/stair_function/results/{experiment_name}"
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
            'modes': modes,
            'shuffled': shuffled,
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
    
    # Generate master dataset
    if rank == 0:
        print("Generating master dataset...")
    X_train_master, y_train_master, X_test, y_test = generate_master_dataset(
        P, d, master_size, n_test, msp
    )
    
    # Generate all parameter combinations
    all_combinations = get_parameter_combinations(hidden_sizes, depths, n_train_sizes, learning_rates, modes)
    
    # Distribute work among workers
    worker_combinations = []
    for i in range(len(all_combinations)):
        if i % size == rank:
            worker_combinations.append(all_combinations[i])
    
    # Process combinations assigned to this worker
    for params in worker_combinations:
        print(f"Worker {rank} processing: {params}")
        
        # Sample training data with fixed seed for reproducibility
        sample_seed = hash(f"sample_{params['n_train']}")
        torch.manual_seed(sample_seed)
        indices = torch.randperm(master_size)[:params['n_train']]
        X_train = X_train_master[indices]
        y_train = y_train_master[indices]
        
        # Apply shuffling if enabled
        if shuffled:
            shuffle_seed = hash(f"shuffle_{params['n_train']}_{timestamp}_{rank}")
            y_train = shuffle_labels(y_train, seed=shuffle_seed)
            print(f"Labels shuffled with seed: {shuffle_seed}")
            params['shuffled'] = True
            params['shuffle_seed'] = shuffle_seed
        else:
            params['shuffled'] = False
            params['shuffle_seed'] = None

        # Model configuration name
        config_name = f"h{params['hidden_size']}_d{params['depth']}_n{params['n_train']}_lr{params['lr']}"
        if shuffled:
            config_name += '_shuffled'
        experiment_config = f"{experiment_name}_{config_name}_{timestamp}_rank{rank}"
        
        # Initialize model
        model = DeepNN(d, params['hidden_size'], params['depth'], mode=params['mode']).to(device)
        
        # Save initial model
        initial_model_path = os.path.join(results_dir, f'initial_model_{experiment_config}.pt')
        torch.save(model.state_dict(), initial_model_path)
        
        # Train model and compute kernels
        test_error, initial_train_error, final_train_error, error_history = train_and_evaluate(
            model, msp, X_train, y_train, X_test, y_test,
            batch_size, epochs, params['lr'], weight_decay, params['mode'],
            results_dir, experiment_config
        )
        
        # Save final model
        final_model_path = os.path.join(results_dir, f'final_model_{experiment_config}.pt')
        torch.save(model.state_dict(), final_model_path)
        
        # Save training history
        history = {
            'test_error': test_error,
            'initial_train_error': initial_train_error,
            'final_train_error': final_train_error,
            'error_history': error_history,
            'parameters': params,
            'sample_seed': sample_seed,
            'shuffled': shuffled,
            'shuffle_seed': params.get('shuffle_seed')
        }
        
        history_path = os.path.join(results_dir, f'history_{experiment_config}.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
            
        print(f"Worker {rank} completed {config_name}")
    
    # Wait for all workers to complete
    comm.Barrier()
    if rank == 0:
        print("All workers completed successfully.")


if __name__ == "__main__":
    main()