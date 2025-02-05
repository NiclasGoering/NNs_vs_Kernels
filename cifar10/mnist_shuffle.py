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
from scipy.linalg import eigh
from torchvision import datasets, transforms
import torch.nn.functional as F

# Define print globally first  
print = partial(print, flush=True)

def compute_conjugate_kernel_batched(X: torch.Tensor, batch_size: int = 1000) -> np.ndarray:
    """Compute conjugate kernel in batches to manage memory"""
    X = X.detach()
    n = X.shape[0]
    device = X.device
    K = torch.zeros((n, n), device=device)
    
    with torch.no_grad():
        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            X_i = X[i:end_i]
            
            for j in range(0, n, batch_size):
                end_j = min(j + batch_size, n)
                X_j = X[j:end_j]
                
                K_ij = torch.matmul(X_i, X_j.T)
                K[i:end_i, j:end_j] = K_ij
    
    return K.cpu().numpy()

def compute_layer_activations(model: nn.Module, X: torch.Tensor) -> List[torch.Tensor]:
    """Compute activations at each layer"""
    activations = []
    x = X
    
    with torch.no_grad():
        for layer in model.network[:-1]:  # Exclude final layer
            if isinstance(layer, nn.Linear):
                x = layer(x)
            elif isinstance(layer, nn.ReLU):
                x = layer(x)
                activations.append(x)
                
    return activations

def compute_and_save_spectrum(K: np.ndarray, k: int, save_path: str):
    """Compute and save top k eigenvalues"""
    n = K.shape[0]
    k = min(k, n)
    
    eigenvals = eigh(K, subset_by_index=(n-k, n-1), eigvals_only=True)
    eigenvals = eigenvals[::-1]
    
    np.save(save_path, eigenvals)
    return eigenvals

class DeepNN(nn.Module):
    def __init__(self, d: int, hidden_size: int, depth: int, mode: str = 'special', gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma
        torch.set_default_dtype(torch.float32)
        
        self.mode = mode
        self.depth = depth
        self.hidden_size = hidden_size
        self.input_dim = d
        
        layers = []
        prev_dim = d
        self.layer_lrs = []
        
        for layer_idx in range(depth):
            linear = nn.Linear(prev_dim, hidden_size)
            
            if mode == 'special':
                gain = nn.init.calculate_gain('relu')
                std = gain / np.sqrt(prev_dim)
                nn.init.normal_(linear.weight, mean=0.0, std=std)
                nn.init.zeros_(linear.bias)
                self.layer_lrs.append(1.0)
                
            elif mode == 'spectral':
                fan_in = prev_dim
                fan_out = hidden_size
                std = (1.0 / np.sqrt(fan_in)) * min(1.0, np.sqrt(fan_out / fan_in))
                nn.init.normal_(linear.weight, mean=0.0, std=std)
                nn.init.zeros_(linear.bias)
                self.layer_lrs.append(float(fan_out) / fan_in)
                
            elif mode == 'mup_pennington':
                if layer_idx == 0:
                    std = 1.0 / np.sqrt(prev_dim)
                    lr_scale = 1.0
                else:
                    std = 1.0 / np.sqrt(prev_dim)
                    lr_scale = 1.0 / prev_dim
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
        
        # Final layer (10 outputs for MNIST)
        final_layer = nn.Linear(prev_dim, 10)
        if mode == 'special':
            nn.init.normal_(final_layer.weight, std=0.01)
            self.layer_lrs.append(1.0)
        elif mode == 'spectral':
            fan_in = prev_dim
            fan_out = 10
            std = (1.0 / np.sqrt(fan_in)) * min(1.0, np.sqrt(fan_out / fan_in))
            nn.init.normal_(final_layer.weight, std=std)
            self.layer_lrs.append(float(fan_out) / fan_in)
        elif mode == 'mup_pennington':
            std = 1.0 / np.sqrt(prev_dim)
            lr_scale = 1.0 / prev_dim
            nn.init.normal_(final_layer.weight, std=std)
            self.layer_lrs.append(lr_scale)
        else:
            nn.init.xavier_uniform_(final_layer.weight)
            self.layer_lrs.append(1.0)
            
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x) / float(self.gamma)
    
    def get_layer_learning_rates(self, base_lr: float) -> List[float]:
        return [base_lr * lr for lr in self.layer_lrs]

def create_layer_specific_optimizer(model: DeepNN, base_lr: float, weight_decay: float):
    """Create optimizer with layer-specific learning rates"""
    if model.mode not in ['spectral', 'mup_pennington']:
        if model.mode == 'special':
            return optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
        else:
            return optim.SGD(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    
    layer_lrs = model.get_layer_learning_rates(base_lr)
    param_groups = []
    
    linear_layer_idx = 0
    for name, param in model.named_parameters():
        if 'weight' in name or 'bias' in name:
            param_groups.append({
                'params': [param],
                'lr': layer_lrs[linear_layer_idx // 2],
                'weight_decay': weight_decay
            })
            if 'bias' in name:
                linear_layer_idx += 1
    
    if model.mode == 'mup_pennington':
        return optim.Adam(param_groups, lr=base_lr)
    else:
        return optim.SGD(param_groups, lr=base_lr)

def save_results(results: List[dict], results_dir: str, timestamp: str):
    try:
        results_path = os.path.join(results_dir, f'results_{timestamp}.json')
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print(f"Error saving results: {e}")

def save_model(model: nn.Module, path: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_cpu = model.cpu()
        torch.save(model_cpu.state_dict(), path)
        model.to(next(model.parameters()).device)
    except Exception as e:
        print(f"Error saving model: {e}")

def shuffle_labels(labels: torch.Tensor, seed: int = None) -> torch.Tensor:
    """Shuffle the labels randomly"""
    if seed is not None:
        torch.manual_seed(seed)
    perm = torch.randperm(labels.size(0))
    return labels[perm]

def train_and_evaluate(model: nn.Module, 
                      X_train: torch.Tensor,
                      y_train: torch.Tensor,
                      X_test: torch.Tensor,
                      y_test: torch.Tensor,
                      batch_size: int,
                      epochs: int,
                      lr: float,
                      weight_decay: float,
                      mode: str,
                      results_dir: str,
                      timestamp: str,
                      rank: int,
                      gamma: float = 1.0) -> Tuple[float, float, float, dict]:
    
    optimizer = create_layer_specific_optimizer(model, lr, weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_test_error = float('inf')
    
    # Get initial training error
    with torch.no_grad():
        initial_train_pred = model(X_train)
        initial_test_pred = model(X_test)
        initial_train_error = criterion(initial_train_pred, y_train).item()
        initial_test_error = criterion(initial_test_pred, y_test).item()
        
        print(f"Initial predictions stats:")
        print(f"Train - mean: {torch.mean(initial_train_pred.float()):.6f}, std: {torch.std(initial_train_pred.float()):.6f}")
        print(f"Initial train accuracy: {(initial_train_pred.argmax(1) == y_train).float().mean():.4f}")
        print(f"Initial test accuracy: {(initial_test_pred.argmax(1) == y_test).float().mean():.4f}")
        
        # Compute initial conjugate kernels
        print("Computing initial conjugate kernels...")
        # initial_activations = compute_layer_activations(model, X_train)
        # for layer_idx, activations in enumerate(initial_activations):
        #     K = compute_conjugate_kernel_batched(activations)
        #     k = min(5000, K.shape[0])
        #     spectrum_path = os.path.join(
        #         results_dir, 
        #         f'initial_spectrum_layer{layer_idx+1}_h{model.hidden_size}_d{model.depth}_n{len(X_train)}_lr{lr}_{timestamp}_rank{rank}.npy'
        #     )
        #     compute_and_save_spectrum(K, k, spectrum_path)
        #     del K
        #     torch.cuda.empty_cache()
        K=None
        k=None
    
    error_history = {
        'train_errors': [],
        'test_errors': [],
        'train_acc': [],
        'test_acc': [],
        'epochs': []
    }
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            train_pred = model(X_train)
            
            train_error = criterion(train_pred, y_train).item()
            test_error = criterion(test_pred, y_test).item()
            train_acc = (train_pred.argmax(1) == y_train).float().mean().item()
            test_acc = (test_pred.argmax(1) == y_test).float().mean().item()
            
            best_test_error = min(best_test_error, test_error)
            
            error_history['train_errors'].append(train_error)
            error_history['test_errors'].append(test_error)
            error_history['train_acc'].append(train_acc)
            error_history['test_acc'].append(test_acc)
            error_history['epochs'].append(epoch)
            
        if epoch % 500 == 0:
            print(f"Epoch {epoch}")
            print(f"Train Error: {train_error:.6f}, Train Acc: {train_acc:.4f}")
            print(f"Test Error: {test_error:.6f}, Test Acc: {test_acc:.4f}")
    
    # Compute final conjugate kernels
    print("Computing final conjugate kernels...")
    #final_activations = compute_layer_activations(model, X_train)
    # for layer_idx, activations in enumerate(final_activations):
    #     K = compute_conjugate_kernel_batched(activations)
    #     k = min(2000, K.shape[0])
    #     spectrum_path = os.path.join(
    #         results_dir, 
    #         f'final_spectrum_layer{layer_idx+1}_h{model.hidden_size}_d{model.depth}_n{len(X_train)}_lr{lr}_{timestamp}_rank{rank}.npy'
    #     )
    #     compute_and_save_spectrum(K, k, spectrum_path)
    #     del K
    #     torch.cuda.empty_cache()
    K=k=None
    
    with torch.no_grad():
        final_train_pred = model(X_train)
        final_train_error = criterion(final_train_pred, y_train).item()
    
    return best_test_error, initial_train_error, final_train_error, error_history
def get_parameter_combinations(hidden_sizes, depths, n_train_sizes, learning_rates, gammas):
    """Generate all possible hyperparameter combinations"""
    combinations = []
    # Convert single gamma to list if necessary
    if not isinstance(gammas, (list, tuple)):
        gammas = [gammas]
        
    for hidden_size in hidden_sizes:
        for depth in depths:
            for n_train in n_train_sizes:
                for lr in learning_rates:
                    for gamma in gammas:
                        combinations.append({
                            'hidden_size': hidden_size,
                            'depth': depth,
                            'n_train': n_train,
                            'lr': lr,
                            'gamma': float(gamma)
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



def load_mnist(device):
    """Load and preprocess MNIST dataset from local directory"""
    import numpy as np
    import os
    import struct
    
    data_path = '/mnt/users/goringn/NNs_vs_Kernels/cifar10/data/MNIST/raw'
    
    def read_idx(filename):
        with open(filename, 'rb') as f:
            zero, data_type, dims = struct.unpack('>HBB', f.read(4))
            shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
    
    # Load training data from uncompressed files
    train_images = read_idx(os.path.join(data_path, 'train-images-idx3-ubyte'))
    train_labels = read_idx(os.path.join(data_path, 'train-labels-idx1-ubyte'))
    
    # Load test data from uncompressed files
    test_images = read_idx(os.path.join(data_path, 't10k-images-idx3-ubyte'))
    test_labels = read_idx(os.path.join(data_path, 't10k-labels-idx1-ubyte'))
    
    # Convert to torch tensors and move to device
    X_train = torch.from_numpy(train_images).float().view(-1, 28*28).to(device) / 255.0
    y_train = torch.from_numpy(train_labels).to(device)
    X_test = torch.from_numpy(test_images).float().view(-1, 28*28).to(device) / 255.0
    y_test = torch.from_numpy(test_labels).to(device)
    
    print(f"Loaded MNIST data:")
    print(f"Training: X shape={X_train.shape}, y shape={y_train.shape}")
    print(f"Test: X shape={X_test.shape}, y shape={y_test.shape}")
    print(f"Label distribution in training: {torch.bincount(y_train)}")
    
    return X_train, y_train, X_test, y_test



def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if rank == 0:
        print(f"Master process using device: {device}")
    
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Parameters 
    experiment_name = "mnist_mup_gamma_true"
    d = 28 * 28  # MNIST input dimension
    master_size = 60000  # Full MNIST training set size
    hidden_sizes = [400, 1000]
    hidden_sizes.reverse()
    depths = [4]
    n_test = 10000  # Full MNIST test set size
    batch_size = 64
    epochs = 5000
    learning_rates = [0.005]
    weight_decay = 1e-4
    mode = 'mup_pennington'
    shuffled = True  # Enable label shuffling
    n_train_sizes = [100, 500, 1000, 5000, 10000, 20000, 40000,60000]
    n_train_sizes.reverse()
    gamma = 1

    # Create results directory if doesn't exist (only master process)
    results_dir = f"/mnt/users/goringn/NNs_vs_Kernels/cifar10/results/{experiment_name}"
    if rank == 0:
        os.makedirs(results_dir, exist_ok=True)
        
        # Save hyperparameters
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        hyperparams = {
            'd': d,
            'hidden_sizes': hidden_sizes,
            'depths': depths,
            'n_test': n_test,
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rates': learning_rates,
            'weight_decay': weight_decay,
            'mode': mode,
            'shuffled': shuffled,
            'n_train_sizes': n_train_sizes,
            'device': str(device),
            'num_workers': size,
            'gamma': gamma
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
    
    # Master process loads MNIST dataset
    if rank == 0:
        print("Master process loading MNIST dataset...")
        X_train_master, y_train_master, X_test, y_test = load_mnist(device)
        print("Data loading complete")
        print(f"Master set shape: {X_train_master.shape}")
        print(f"Test set shape: {X_test.shape}")
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
    
    # Generate all parameter combinations
    all_combinations = get_parameter_combinations(hidden_sizes, depths, n_train_sizes, learning_rates, gamma)
    
    # Distribute work among workers
    worker_combinations = []
    for i in range(len(all_combinations)):
        if i % size == rank:
            worker_combinations.append(all_combinations[i])
    
    # Process combinations assigned to this worker
    results = []
    for params in worker_combinations:
        print(f"Worker {rank} processing: {params}")
        
        # Sample from master training set using fixed seed for this configuration
        sample_seed = hash(f"sample_{params['n_train']}")
        torch.manual_seed(sample_seed)
        indices = torch.randperm(master_size)[:params['n_train']]
        X_train = X_train_master[indices]
        y_train = y_train_master[indices]
        
        # If shuffling is enabled, shuffle the labels
        if shuffled:
            shuffle_seed = hash(f"shuffle_{params['n_train']}_{timestamp}_{rank}")
            y_train = shuffle_labels(y_train, seed=shuffle_seed)
            print(f"Labels shuffled with seed: {shuffle_seed}")
            print(f"Label distribution after shuffling: {torch.bincount(y_train)}")
            params['shuffled'] = True
            params['shuffle_seed'] = shuffle_seed
        else:
            params['shuffled'] = False
            params['shuffle_seed'] = None
        
        # Save the training dataset for this configuration
        model_prefix = f'h{params["hidden_size"]}_d{params["depth"]}_n{params["n_train"]}_lr{params["lr"]}_g{gamma}_{mode}'
        if shuffled:
            model_prefix += '_shuffled'
            
        # Save the training dataset
        train_dataset_path = os.path.join(results_dir, f'train_dataset_{model_prefix}_{timestamp}_rank{rank}.pt')
        save_dataset(X_train, y_train, train_dataset_path, rank)
        
        print(f"Sampled training set shape: {X_train.shape}")
        print(f"Sample seed used: {sample_seed}")
        
        # Use different seed for model initialization
        model_seed = hash(f"model_{datetime.now()}_{rank}")
        torch.manual_seed(model_seed)
        print(f"Model initialization seed: {model_seed}")
        
        # Initialize and save initial model
        initial_model = DeepNN(d, params['hidden_size'], params['depth'], mode=mode, gamma=gamma).to(device)
        initial_model_path = os.path.join(results_dir, f'initial_model_{model_prefix}_{timestamp}_rank{rank}.pt')
        #save_model(initial_model, initial_model_path)
        
        # Train model
        model = DeepNN(d, params['hidden_size'], params['depth'], mode=mode, gamma=gamma).to(device)
        model.load_state_dict(initial_model.state_dict())
        test_error, initial_train_error, final_train_error, error_history = train_and_evaluate(
            model, X_train, y_train, X_test, y_test,
            batch_size, epochs, params['lr'], weight_decay, mode,
            results_dir, timestamp, rank, gamma
        )
        
        # Save final model
        final_model_path = os.path.join(results_dir, f'final_model_{model_prefix}_{timestamp}_rank{rank}.pt')
        #save_model(model, final_model_path)
        
        # Store results
        result = {
            'hidden_size': params['hidden_size'],
            'depth': params['depth'],
            'n_train': params['n_train'],
            'learning_rate': params['lr'],
            'mode': mode,
            'gamma': gamma,
            'shuffled': shuffled,
            'shuffle_seed': params.get('shuffle_seed'),
            'test_error': test_error,
            'initial_train_error': initial_train_error,
            'final_train_error': final_train_error,
            'error_history': error_history,
            'worker_rank': rank,
            'sample_seed': sample_seed,
            'model_seed': model_seed,
            'initial_model_path': initial_model_path,
            'final_model_path': final_model_path,
            'train_dataset_path': train_dataset_path
        }
        results.append(result)
        
        # Save intermediate results for this worker
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