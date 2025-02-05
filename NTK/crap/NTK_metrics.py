#!/usr/bin/env python3
import numpy as np
import os
import glob
import re
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from sklearn.kernel_ridge import KernelRidge
from scipy.linalg import sqrtm
from torch.func import functional_call, vmap, jacrev

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
        self.layer_lrs = []
        
        for layer_idx in range(depth):
            linear = nn.Linear(prev_dim, hidden_size)
            
            if mode == 'special':
                gain = nn.init.calculate_gain('relu')
                std = gain / np.sqrt(prev_dim)
                nn.init.normal_(linear.weight, mean=0.0, std=std)
                nn.init.zeros_(linear.bias)
                self.layer_lrs.append(1.0)
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
            
            layers.extend([linear, nn.ReLU()])
            prev_dim = hidden_size
        
        final_layer = nn.Linear(prev_dim, 1)
        if mode == 'special':
            nn.init.normal_(final_layer.weight, std=0.01)
            self.layer_lrs.append(1.0)
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
        return self.network(x).squeeze()
        
    def compute_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        features = x
        for layer in list(self.network.children())[:-1]:
            features = layer(features)
        return features

def load_kernels(kernel_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load NNGP and NTK kernels from files"""
    nngp = np.load(f"{kernel_path}_nngp.npy")
    ntk = np.load(f"{kernel_path}_ntk.npy")
    return nngp, ntk

def parse_kernel_params(filename: str) -> dict:
    """Extract parameters from kernel filename"""
    pattern = r'h(\d+)_d(\d+)_n(\d+)_lr([\d\.]+)_(\w+)'
    match = re.search(pattern, filename)
    if match:
        return {
            'hidden_size': int(match.group(1)),
            'depth': int(match.group(2)),
            'n_train': int(match.group(3)),
            'lr': float(match.group(4)),
            'mode': match.group(5),
            'd': 30  # Fixed input dimension for this experiment
        }
    raise ValueError(f"Couldn't parse parameters from filename: {filename}")

def compute_empirical_kernels_test_train(model: nn.Module, X_train: torch.Tensor, 
                                   X_test: torch.Tensor, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """Compute empirical NNGP and NTK kernels between test and train sets"""
    model.eval()
    device = X_train.device
    
    def fnet_single(params, x):
        return functional_call(model, params, (x.unsqueeze(0),)).squeeze(0)
    
    def compute_kernel_batch(x1_batch: torch.Tensor, x2_batch: torch.Tensor):
        params = {k: v.to(device) for k, v in model.named_parameters()}
        jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1_batch)
        jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2_batch)
        
        jac1_flat = []
        jac2_flat = []
        for j1, j2 in zip(jac1.values(), jac2.values()):
            j1_shape = j1.shape
            j2_shape = j2.shape
            j1_reshaped = j1.reshape(j1_shape[0], -1)
            j2_reshaped = j2.reshape(j2_shape[0], -1)
            jac1_flat.append(j1_reshaped)
            jac2_flat.append(j2_reshaped)
        
        ntk_result = sum(torch.matmul(j1, j2.t()) for j1, j2 in zip(jac1_flat, jac2_flat))
        
        with torch.no_grad():
            feat1 = model.compute_feature_map(x1_batch)
            feat2 = model.compute_feature_map(x2_batch)
            nngp_result = torch.matmul(feat1, feat2.T) / feat1.shape[1]
        
        return nngp_result.detach(), ntk_result.detach()
    
    try:
        n1, n2 = X_test.shape[0], X_train.shape[0]
        nngp = torch.zeros((n1, n2), device=device)
        ntk = torch.zeros((n1, n2), device=device)
        
        for i in range(0, n1, batch_size):
            i_end = min(i + batch_size, n1)
            for j in range(0, n2, batch_size):
                j_end = min(j + batch_size, n2)
                
                nngp_batch, ntk_batch = compute_kernel_batch(
                    X_test[i:i_end], 
                    X_train[j:j_end]
                )
                nngp[i:i_end, j:j_end] = nngp_batch
                ntk[i:i_end, j:j_end] = ntk_batch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return nngp.cpu().numpy(), ntk.cpu().numpy()
    
    except torch.cuda.OutOfMemoryError:
        print("GPU out of memory. Try reducing batch size or using CPU.")
        raise
    except Exception as e:
        print(f"Error computing kernels: {str(e)}")
        raise

def compute_test_error(K_train: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                      K_test_train: np.ndarray) -> float:
    """Compute test error using kernel ridge regression"""
    reg = KernelRidge(kernel='precomputed', alpha=1e-6)
    reg.fit(K_train, y_train)
    y_pred = reg.predict(K_test_train)
    return np.mean((y_pred - y_test) ** 2)

def compute_cka(K1: np.ndarray, K2: np.ndarray) -> float:
    """Compute Centered Kernel Alignment between two kernels"""
    def center_kernel(K):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H
    
    K1_centered = center_kernel(K1)
    K2_centered = center_kernel(K2)
    
    hsic = np.sum(K1_centered * K2_centered)
    norm1 = np.sqrt(np.sum(K1_centered * K1_centered))
    norm2 = np.sqrt(np.sum(K2_centered * K2_centered))
    
    return hsic / (norm1 * norm2)

def compute_norm_frob_diff(K1: np.ndarray, K2: np.ndarray) -> float:
    """Compute normalized Frobenius norm between two kernels"""
    diff = K1 - K2
    norm_diff = np.sqrt(np.sum(diff ** 2))
    norm_K1 = np.sqrt(np.sum(K1 ** 2))
    return norm_diff / norm_K1

def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        print("CUDA not available. Using CPU instead.")
        return torch.device('cpu')

def process_kernels(base_dir: str, results_dir: str):
    """Process all kernels and compute metrics"""
    # Set up device
    device = get_device()
    
    # Find all kernel files
    kernel_files = glob.glob(os.path.join(base_dir, "kernel_analysis/*_initial_kernels_nngp.npy"))
    
    # Group by hidden size
    kernels_by_hidden = {}
    for kernel_file in kernel_files:
        base_name = os.path.basename(kernel_file).replace('_initial_kernels_nngp.npy', '')
        params = parse_kernel_params(base_name)
        h = params['hidden_size']
        if h not in kernels_by_hidden:
            kernels_by_hidden[h] = []
        kernels_by_hidden[h].append((base_name, params))
    
    # Initialize results dictionaries
    metrics = {h: {
        'n_train': [],
        'test_error_initial_ntk': [],
        'test_error_final_ntk': [],
        'test_error_initial_nngp': [],
        'test_error_final_nngp': [],
        'cka_ntk': [],
        'cka_nngp': [],
        'frob_norm_ntk': [],
        'frob_norm_nngp': []
    } for h in kernels_by_hidden.keys()}
    
    # Load test set using most recent test dataset
    test_files = glob.glob(os.path.join(base_dir, 'test_dataset_*.pt'))
    latest_test_file = max(test_files)  # Gets the most recent test dataset
    test_data = torch.load(latest_test_file)
    X_test = test_data['X'].numpy()
    y_test = test_data['y'].numpy()
    
    # Process each hidden size group
    for hidden_size, kernel_group in kernels_by_hidden.items():
        print(f"Processing hidden size {hidden_size}")
        
        for base_name, params in kernel_group:
            # Load training data
            train_data = torch.load(os.path.join(base_dir, f'train_dataset_{base_name}.pt'))
            y_train = train_data['y'].numpy()
            
            # Load kernels
            initial_nngp, initial_ntk = load_kernels(os.path.join(base_dir, f"kernel_analysis/{base_name}_initial_kernels"))
            final_nngp, final_ntk = load_kernels(os.path.join(base_dir, f"kernel_analysis/{base_name}_final_kernels"))
            
            # We already have test data loaded from earlier
            
            # Load model and compute proper test-train kernels
            model = DeepNN(
                d=params['d'],
                hidden_size=params['hidden_size'],
                depth=params['depth'],
                mode=params['mode']
            ).to(device)
            
            # Convert data to torch tensors
            X_test_tensor = torch.tensor(X_test, device=device)
            X_train_tensor = torch.tensor(train_data['X'].numpy(), device=device)
            
            # Compute initial test-train kernels
            model.load_state_dict(torch.load(os.path.join(base_dir, f'initial_model_{base_name}.pt'), map_location=device))
            K_test_train_nngp_initial, K_test_train_ntk_initial = compute_empirical_kernels_test_train(
                model, X_train_tensor, X_test_tensor)
                
            # Compute final test-train kernels
            model.load_state_dict(torch.load(os.path.join(base_dir, f'final_model_{base_name}.pt'), map_location=device))
            K_test_train_nngp_final, K_test_train_ntk_final = compute_empirical_kernels_test_train(
                model, X_train_tensor, X_test_tensor)
            
            # Calculate metrics
            metrics[hidden_size]['n_train'].append(params['n_train'])
            
            # Test errors
            metrics[hidden_size]['test_error_initial_ntk'].append(
                compute_test_error(initial_ntk, y_train, y_test, K_test_train_ntk_initial))
            metrics[hidden_size]['test_error_final_ntk'].append(
                compute_test_error(final_ntk, y_train, y_test, K_test_train_ntk_final))
            metrics[hidden_size]['test_error_initial_nngp'].append(
                compute_test_error(initial_nngp, y_train, y_test, K_test_train_nngp_initial))
            metrics[hidden_size]['test_error_final_nngp'].append(
                compute_test_error(final_nngp, y_train, y_test, K_test_train_nngp_final))
            
            # CKA
            metrics[hidden_size]['cka_ntk'].append(compute_cka(initial_ntk, final_ntk))
            metrics[hidden_size]['cka_nngp'].append(compute_cka(initial_nngp, final_nngp))
            
            # Frobenius norms
            metrics[hidden_size]['frob_norm_ntk'].append(compute_norm_frob_diff(final_ntk, initial_ntk))
            metrics[hidden_size]['frob_norm_nngp'].append(compute_norm_frob_diff(final_nngp, initial_nngp))
    
    # Save results
    os.makedirs(results_dir, exist_ok=True)
    for hidden_size, results in metrics.items():
        for metric_name, values in results.items():
            if metric_name == 'n_train':
                continue
            
            # Sort by n_train
            n_train = np.array(results['n_train'])
            metric_values = np.array(values)
            sort_idx = np.argsort(n_train)
            
            # Save sorted arrays
            save_path = os.path.join(results_dir, f'h{hidden_size}_{metric_name}_vs_ntrain.npy')
            np.save(save_path, np.vstack((n_train[sort_idx], metric_values[sort_idx])))

def main():
    base_dir = "/mnt/users/goringn/NNs_vs_Kernels/stair_function/results/msp_1601_NTK_norm_mup"
    results_dir = os.path.join(base_dir, "kernel_metrics")
    process_kernels(base_dir, results_dir)

if __name__ == "__main__":
    main()