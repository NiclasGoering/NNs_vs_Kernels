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
import torch
import numpy as np
from typing import Dict, Tuple


class DeepNN(nn.Module):
    def __init__(self, d: int, hidden_size: int, depth: int, mode: str = 'special', gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma
        # ... rest of initialization code stays the same ...

        
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
        gamma = float(self.gamma)  # Ensure gamma is a float
        return self.network(x).squeeze() / gamma
    
    def get_layer_learning_rates(self, base_lr: float) -> List[float]:
        """Return list of learning rates for each layer"""
        return [base_lr * lr for lr in self.layer_lrs]



def load_dataset(dataset_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load dataset from saved file"""
    data = torch.load(dataset_path)
    return data['X'], data['y']

def get_top_eigenfunctions(features: torch.Tensor, k: int) -> torch.Tensor:
    """Compute top k eigenfunctions from feature representation"""
    with torch.no_grad():
        K = torch.matmul(features, features.T)
        _, V = torch.linalg.eigh(K)
        return V[:, -k:]  # Top k eigenfunctions

def compute_eigenfunction_similarity(orig_eigenfuncs: torch.Tensor, 
                                  new_eigenfuncs: torch.Tensor) -> float:
    """Compute similarity between two sets of eigenfunctions"""
    # Use canonical correlations to handle sign ambiguity
    corrs = torch.abs(torch.matmul(orig_eigenfuncs.T, new_eigenfuncs))
    return torch.mean(torch.diag(corrs)).item()


def get_model_architecture(model_state):
    """Extract correct architecture from model state dict"""
    # Get max layer index
    layer_indices = []
    for key in model_state.keys():
        if 'network.' in key and '.weight' in key:
            layer_idx = int(key.split('.')[1])
            layer_indices.append(layer_idx)
    
    max_layer = max(layer_indices)
    # Depth is number of hidden layers (exclude output)
    # Every 2 indices is one layer (weight + bias)
    depth = max_layer // 2
    
    # Input dimension from first layer
    d = model_state['network.0.weight'].shape[1]
    # Hidden size from first hidden layer
    hidden_size = model_state['network.0.weight'].shape[0]
    
    print(f"\nModel state analysis:")
    print(f"Layer indices found: {sorted(layer_indices)}")
    print(f"Max layer index: {max_layer}")
    print(f"Calculated depth: {depth}")
    
    return d, hidden_size, depth


import time

def prune_layer_batch(model, layer_name, param, indices_to_try, orig_eigenfuncs, X_train, epsilon):
    """Prune a batch of weights from one layer at once"""
    with torch.no_grad():
        # Save original values
        original_values = param.data.clone()
        
        # Zero out all weights in batch
        param.data.flatten()[indices_to_try] = 0
        
        # Compute new eigenfunctions
        new_features = model.network[:-1](X_train)
        new_eigenfuncs = get_top_eigenfunctions(new_features, orig_eigenfuncs.shape[1])
        similarity = compute_eigenfunction_similarity(orig_eigenfuncs, new_eigenfuncs)
        
        if 1 - similarity > epsilon:
            # Restore all weights if similarity too low
            param.data.copy_(original_values)
            return 0
        else:
            # Keep weights zeroed
            return len(indices_to_try)

def prune_model(model_path: str, 
                train_dataset_path: str, 
                test_dataset_path: str,
                k: int = 5, 
                epsilon: float = 0.01,
                save_path: str = "pruned_model.pt",
                batch_size: int = 1000) -> Dict:
    """
    Prune model while keeping top k eigenfunctions within epsilon similarity
    """
    print("Starting model pruning process...")
    
    # Load model and data
    print(f"\nLoading model from {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model_state = torch.load(model_path, weights_only=True)
    
    # Print model state keys for debugging
    print("Model state keys:", model_state.keys())
    
    # Get architecture
    d, hidden_size, depth = get_model_architecture(model_state)
    print(f"\nDetected architecture:")
    print(f"Input dim: {d}")
    print(f"Hidden size: {hidden_size}")
    print(f"Depth: {depth}")
    
    # Create model and verify before loading state
    model = DeepNN(d, hidden_size, depth).to(device)
    print("\nCreated model state dict:", model.state_dict().keys())
    print("Loading state dict...")
    model.load_state_dict(model_state)
    
    # Load datasets
    print("\nLoading datasets...")
    X_train, y_train = load_dataset(train_dataset_path)
    X_test, y_test = load_dataset(test_dataset_path)
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)
    
    # Get initial performance
    print("\nComputing initial performance...")
    with torch.no_grad():
        initial_train_pred = model(X_train)
        initial_test_pred = model(X_test)
        initial_train_error = torch.mean((initial_train_pred - y_train) ** 2).item()
        initial_test_error = torch.mean((initial_test_pred - y_test) ** 2).item()
        print(f"Initial train error: {initial_train_error:.6f}")
        print(f"Initial test error: {initial_test_error:.6f}")
        
        print("\nGetting original eigenfunctions...")
        orig_features = model.network[:-1](X_train)
        orig_eigenfuncs = get_top_eigenfunctions(orig_features, k)
        print(f"Computed top {k} eigenfunctions")
    
    # Count initial parameters
    initial_nonzero = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nInitial parameter count: {initial_nonzero}")
    
    # Initialize masks
    masks = {name: torch.ones_like(param) for name, param in model.named_parameters() 
            if 'weight' in name}
    
    # Prune each layer
    total_pruned = 0
    start_time = time.time()
    
    for name, param in model.named_parameters():
        if 'weight' not in name:
            continue
            
        print(f"\nPruning layer: {name}")
        print(f"Layer shape: {param.shape}")
        layer_start = time.time()
        
        # Sort weights by magnitude
        weights = param.data.flatten()
        indices = torch.argsort(weights.abs())
        
        # Prune in batches
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            
            # Save original values
            original_values = param.data.clone()
            
            # Try pruning batch
            with torch.no_grad():
                # Zero out batch of weights
                param.data.flatten()[batch_indices] = 0
                
                # Check impact on eigenfunctions
                new_features = model.network[:-1](X_train)
                new_eigenfuncs = get_top_eigenfunctions(new_features, k)
                similarity = compute_eigenfunction_similarity(orig_eigenfuncs, new_eigenfuncs)
                
                if 1 - similarity > epsilon:
                    # Restore weights if similarity too low
                    param.data.copy_(original_values)
                else:
                    # Keep weights zeroed and update mask
                    total_pruned += len(batch_indices)
                    flat_mask = masks[name].flatten()
                    flat_mask[batch_indices] = 0
                    masks[name] = flat_mask.view(param.shape)
            
            # Report progress
            current_nonzero = sum((p != 0).sum().item() 
                                for p in model.parameters() if p.requires_grad)
            elapsed = time.time() - start_time
            
            print(f"\rProgress: {i+len(batch_indices)}/{len(indices)} weights "
                  f"({(i+len(batch_indices))/len(indices)*100:.1f}%) "
                  f"| Params: {current_nonzero} ({current_nonzero/initial_nonzero*100:.1f}%) "
                  f"| Pruned: {total_pruned} "
                  f"| Time: {elapsed:.1f}s", end="")
        
        layer_time = time.time() - layer_start
        print(f"\nLayer completed in {layer_time:.1f}s")
    
    print("\nPruning complete!")
    
    # Get final performance
    with torch.no_grad():
        final_train_pred = model(X_train)
        final_test_pred = model(X_test)
        final_train_error = torch.mean((final_train_pred - y_train) ** 2).item()
        final_test_error = torch.mean((final_test_pred - y_test) ** 2).item()
    
    final_nonzero = sum((param != 0).sum().item() 
                       for param in model.parameters() if p.requires_grad)
    
    # Save model
    print("\nSaving pruned model...")
    pruned_state = {
        'state_dict': model.state_dict(),
        'masks': masks,
        'sparsity': 1 - (final_nonzero / initial_nonzero)
    }
    torch.save(pruned_state, save_path)
    print(f"Saved to {save_path}")
    
    results = {
        'initial_train_error': initial_train_error,
        'initial_test_error': initial_test_error,
        'final_train_error': final_train_error,
        'final_test_error': final_test_error,
        'initial_params': initial_nonzero,
        'final_params': final_nonzero,
        'sparsity': 1 - (final_nonzero / initial_nonzero)
    }
    
    print("\nFinal Results:")
    print(f"Initial Parameters: {results['initial_params']}")
    print(f"Final Parameters: {results['final_params']} ({final_nonzero/initial_nonzero*100:.1f}%)")
    print(f"Parameters Pruned: {total_pruned}")
    print(f"\nErrors:")
    print(f"Initial Train: {initial_train_error:.6f}")
    print(f"Initial Test:  {initial_test_error:.6f}")
    print(f"Final Train:   {final_train_error:.6f}")
    print(f"Final Test:    {final_test_error:.6f}")
    
    return results


# Example usage with paths
model_path = "/mnt/users/goringn/NNs_vs_Kernels/stair_function/results/msp_anthro_false_mup_lr0005_gamma_1_modelsaved/final_model_h400_d4_n10000_lr0.005_g1.0_mup_pennington_20250125_153135_rank0.pt"
train_dataset_path = "/mnt/users/goringn/NNs_vs_Kernels/stair_function/results/msp_anthro_false_mup_lr0005_gamma_1_modelsaved/train_dataset_h400_d4_n10000_lr0.005_g1.0_mup_pennington_20250125_153135_rank0.pt"
test_dataset_path = "stair_function/results/msp_anthro_false_mup_lr0005_gamma_1_modelsaved/test_dataset_20250125_153135.pt"

results = prune_model(
    model_path,
    train_dataset_path,
    test_dataset_path,
    k=5,
    epsilon=0.01
)