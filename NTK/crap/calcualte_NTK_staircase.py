#!/usr/bin/env python3
import torch
import jax
import jax.numpy as jnp
from jax import random
import neural_tangents as nt
import numpy as np
from typing import Tuple, Dict, List, Sequence, Optional
import os
import torch.nn as nn
import re
from glob import glob
from neural_tangents import stax
from functools import partial

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

def get_ntk_fns(input_dim: int, hidden_size: int, depth: int, output_dim: int = 1):
    """Create neural network using stax API with specified architecture and multiple NTK implementations.
    
    Args:
        input_dim: Input dimension
        hidden_size: Size of hidden layers
        depth: Number of hidden layers
        output_dim: Output dimension (default 1)
        
    Returns:
        tuple: (init_fn, apply_fn, params, ntk_fns)
    """
    layers = []
    
    # Input layer
    layers.extend([
        stax.Dense(hidden_size),
        stax.Relu()
    ])
    
    # Hidden layers
    for _ in range(depth - 1):
        layers.extend([
            stax.Dense(hidden_size),
            stax.Relu()
        ])
    
    # Output layer
    layers.append(stax.Dense(output_dim))
    
    # Create the network
    init_fn, apply_fn, _ = stax.serial(*layers)
    
    # Setup for NTK computation
    kwargs = dict(
        f=apply_fn,
        trace_axes=(),
        vmap_axes=0
    )
    
    # Create different NTK implementations
    ntk_fns = {
        'jacobian_contraction': jax.jit(nt.empirical_ntk_fn(
            **kwargs, implementation=nt.NtkImplementation.JACOBIAN_CONTRACTION)),
        'ntvp': jax.jit(nt.empirical_ntk_fn(
            **kwargs, implementation=nt.NtkImplementation.NTK_VECTOR_PRODUCTS)),
        'structured_derivatives': jax.jit(nt.empirical_ntk_fn(
            **kwargs, implementation=nt.NtkImplementation.STRUCTURED_DERIVATIVES)),
        'auto': jax.jit(nt.empirical_ntk_fn(
            **kwargs, implementation=nt.NtkImplementation.AUTO))
    }
    
    # Initialize parameters with dummy input
    _, params = init_fn(random.PRNGKey(0), (-1, input_dim))
    
    return init_fn, apply_fn, params, ntk_fns

def convert_pytorch_to_stax(state_dict: Dict, input_dim: int, hidden_size: int, depth: int) -> Tuple:
    """Convert PyTorch state dict to stax parameters
    
    Args:
        state_dict: PyTorch state dictionary
        input_dim: Input dimension
        hidden_size: Hidden layer size
        depth: Number of layers
        
    Returns:
        tuple: (init_fn, apply_fn, params, ntk_fns)
    """
    # Get stax functions and empty parameters structure
    init_fn, apply_fn, _, ntk_fns = get_ntk_fns(input_dim, hidden_size, depth)
    
    # Convert weights from PyTorch format to stax format
    weight_keys = [k for k in state_dict.keys() if 'weight' in k]
    
    # Create list of (weights, biases) tuples
    params = []
    for key in weight_keys:
        # Get weights and biases
        weights = jnp.array(state_dict[key].numpy())
        biases = jnp.array(state_dict[key.replace('weight', 'bias')].numpy())
        
        # Add tuple of (weights.T, biases) - note transpose for stax format
        params.append((weights.T, biases))
    
    # Convert list to tuple for immutability
    params = tuple(params)
    
    return init_fn, apply_fn, params, ntk_fns

def validate_conversion(pytorch_model: torch.nn.Module, 
                       apply_fn: callable, 
                       params: Sequence,
                       X_val: torch.Tensor) -> Tuple[float, float]:
    """Validate conversion from PyTorch to stax
    
    Args:
        pytorch_model: Original PyTorch model
        apply_fn: Stax apply function
        params: Stax parameters
        X_val: Validation data
        
    Returns:
        tuple: (absolute_difference, relative_difference)
    """
    pytorch_model.eval()
    
    # Get PyTorch output
    with torch.no_grad():
        pytorch_output = pytorch_model(X_val).numpy()
    
    # Get stax output
    X_val_jax = jnp.array(X_val.numpy())
    stax_output = apply_fn(params, X_val_jax)
    
    # Compare outputs
    abs_diff = np.mean(np.abs(pytorch_output - stax_output))
    rel_diff = abs_diff / (np.mean(np.abs(pytorch_output)) + 1e-8)
    
    print("\nModel Conversion Validation:")
    print(f"PyTorch output stats - Mean: {pytorch_output.mean():.6f}, Std: {pytorch_output.std():.6f}")
    print(f"Stax output stats    - Mean: {stax_output.mean():.6f}, Std: {stax_output.std():.6f}")
    print(f"Mean absolute difference: {abs_diff:.6f}")
    print(f"Relative difference: {rel_diff:.6f}")
    
    return abs_diff, rel_diff

def compute_empirical_ntk(ntk_fns: Dict[str, callable], params: Sequence, 
                         x1: jnp.ndarray, x2: Optional[jnp.ndarray] = None) -> Dict[str, jnp.ndarray]:
    """Compute empirical NTK using all implementations
    
    Args:
        ntk_fns: Dictionary of NTK functions
        params: Stax parameters
        x1: First batch of inputs
        x2: Optional second batch of inputs
        
    Returns:
        Dictionary of NTK matrices from different implementations
    """
    if x2 is None:
        x2 = x1
        
    results = {}
    for name, fn in ntk_fns.items():
        print(f"\nComputing NTK using {name} implementation...")
        ntk = fn(x1, x2, params)
        results[name] = np.array(ntk)
        print(f"NTK shape: {ntk.shape}")
    
    # Validate that all implementations give similar results
    print("\nValidating consistency between implementations:")
    base_impl = 'structured_derivatives'
    base_ntk = results[base_impl]
    base_mean = jnp.mean(jnp.abs(base_ntk))
    
    for name, ntk in results.items():
        if name != base_impl:
            max_diff = jnp.max(jnp.abs(ntk - base_ntk)) / base_mean
            print(f"Max relative difference ({base_impl} vs {name}): {max_diff:.8f}")
    
    # Return the structured derivatives implementation by default
    return results['structured_derivatives']

def save_ntk(ntk: np.ndarray, save_path: str):
    """Save NTK matrix to file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Saving NTK to: {save_path}")
    np.save(save_path, ntk)
    print(f"Successfully saved NTK of shape {ntk.shape}")

def parse_model_info(filename: str) -> Dict:
    """Extract model information from filename"""
    pattern = r'(?P<type>initial|final)_model_h(?P<hidden>\d+)_d(?P<depth>\d+)_n(?P<ntrain>\d+)_lr(?P<lr>[\d\.]+)_(?P<mode>\w+)_(?P<timestamp>\d+)_rank(?P<rank>\d+).pt'
    match = re.match(pattern, os.path.basename(filename))
    if match:
        return match.groupdict()
    return None

def find_matching_dataset(model_info: Dict, data_files: List[str]) -> str:
    """Find corresponding dataset file for a model"""
    pattern = f'train_dataset_h{model_info["hidden"]}_d{model_info["depth"]}_n{model_info["ntrain"]}_lr{model_info["lr"]}_{model_info["mode"]}_{model_info["timestamp"]}_rank{model_info["rank"]}.pt'
    for file in data_files:
        if os.path.basename(file) == pattern:
            return file
    return None

def load_pytorch_model_and_data(model_path: str, data_path: str) -> Tuple[Dict, torch.Tensor]:
    """Load PyTorch model weights and training data"""
    print(f"Loading model from: {model_path}")
    print(f"Loading data from: {data_path}")
    
    state_dict = torch.load(model_path, weights_only=True)
    data = torch.load(data_path, weights_only=True)
    X_train = data['X']
    
    print(f"Successfully loaded model and data")
    return state_dict, X_train

def main():
    # Hyperparameters
    hyperparameters = {
        'input_dim': 30,
        'input_dir': "/mnt/users/goringn/NNs_vs_Kernels/stair_function/results/msp_1601_NTK_norm_mup",
        'output_dir': "/mnt/users/goringn/NNs_vs_Kernels/NTK/results"
    }
    
    print(f"\nProcessing all models in: {hyperparameters['input_dir']}")
    print(f"Saving NTK results to: {hyperparameters['output_dir']}\n")
    
    # Get all model and dataset files
    model_files = glob(os.path.join(hyperparameters['input_dir'], '*model*.pt'))
    data_files = glob(os.path.join(hyperparameters['input_dir'], '*dataset*.pt'))
    
    # Group initial and final models
    models_info = {}
    for model_file in model_files:
        info = parse_model_info(model_file)
        if info:
            key = f"h{info['hidden']}_d{info['depth']}_n{info['ntrain']}"
            if key not in models_info:
                models_info[key] = {'initial': None, 'final': None, 'info': info}
            models_info[key][info['type']] = model_file
    
    # Process each model pair
    for config_key, model_pair in models_info.items():
        if not (model_pair['initial'] and model_pair['final']):
            print(f"Skipping incomplete model pair for {config_key}")
            continue
            
        info = model_pair['info']
        print(f"\nProcessing models for configuration: {config_key}")
        
        # Find corresponding dataset
        dataset_path = find_matching_dataset(info, data_files)
        if not dataset_path:
            print(f"Could not find matching dataset for {config_key}")
            continue
        
        # Load models and data
        initial_state_dict, X_train = load_pytorch_model_and_data(model_pair['initial'], dataset_path)
        final_state_dict, _ = load_pytorch_model_and_data(model_pair['final'], dataset_path)
        
        # Convert to JAX array
        X_train_jax = jnp.array(X_train.numpy())
        
        # Convert initial model to stax
        _, initial_apply_fn, initial_params, initial_ntk_fns = convert_pytorch_to_stax(
            initial_state_dict,
            hyperparameters['input_dim'],
            int(info['hidden']),
            int(info['depth'])
        )
        
        # Convert final model to stax
        _, final_apply_fn, final_params, final_ntk_fns = convert_pytorch_to_stax(
            final_state_dict,
            hyperparameters['input_dim'],
            int(info['hidden']),
            int(info['depth'])
        )
        
        # Validate conversions
        print("\nValidating initial model conversion...")
        pytorch_model = DeepNN(hyperparameters['input_dim'], int(info['hidden']), 
                             int(info['depth']), mode='mup_pennington')
        pytorch_model.load_state_dict(initial_state_dict)
        initial_abs_diff, initial_rel_diff = validate_conversion(
            pytorch_model,
            initial_apply_fn,
            initial_params,
            X_train[:100]  # Validate on first 100 samples
        )
        
        print("\nValidating final model conversion...")
        pytorch_model.load_state_dict(final_state_dict)
        final_abs_diff, final_rel_diff = validate_conversion(
            pytorch_model,
            final_apply_fn,
            final_params,
            X_train[:100]
        )
        
        # Compute NTKs
        print("\nComputing NTKs...")
        initial_empirical_ntk = compute_empirical_ntk(initial_ntk_fns, initial_params, X_train_jax)
        final_empirical_ntk = compute_empirical_ntk(final_ntk_fns, final_params, X_train_jax)
        
        # Save NTKs
        base_filename = f"{config_key}_lr{info['lr']}_{info['mode']}_{info['timestamp']}"
        save_ntk(initial_empirical_ntk, 
                os.path.join(hyperparameters['output_dir'], 
                            f'initial_empirical_ntk_{base_filename}.npy'))
        save_ntk(final_empirical_ntk, 
                os.path.join(hyperparameters['output_dir'], 
                            f'final_empirical_ntk_{base_filename}.npy'))
        
        print(f"\nCompleted processing for {config_key}")
    
    print("\nAll models have been processed!")

if __name__ == "__main__":
    main()