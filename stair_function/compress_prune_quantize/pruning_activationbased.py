#!/usr/bin/env python3
import os
import re
import glob
import torch
import numpy as np
import pickle
from torch import nn
import copy
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class DeepNN(nn.Module):
    def __init__(self, d: int, hidden_size: int, depth: int, mode: str = 'special', gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma
        torch.set_default_dtype(torch.float32)
        
        layers = []
        prev_dim = d
        for _ in range(depth):
            linear = nn.Linear(prev_dim, hidden_size)
            
            if mode == 'special':
                gain = nn.init.calculate_gain('relu')
                std = gain / np.sqrt(prev_dim)
                nn.init.normal_(linear.weight, mean=0.0, std=std)
                nn.init.zeros_(linear.bias)
            else:
                nn.init.xavier_uniform_(linear.weight)
                nn.init.zeros_(linear.bias)
            
            layers.extend([linear, nn.ReLU()])
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
        return self.network(x).squeeze() / float(self.gamma)

def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        mse = torch.mean((y_pred - y) ** 2).item()
    return mse

def get_activation_importance(model, X, batch_size=1000):
    """
    Compute importance scores based on activation patterns
    """
    model.eval()
    device = next(model.parameters()).device
    activations_dict = {}
    hooks = []
    
    # Register hooks for each layer
    def get_hook(name):
        def hook(module, input, output):
            if name not in activations_dict:
                activations_dict[name] = []
            activations_dict[name].append(output.detach())
        return hook
    
    # Add hooks to all linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(get_hook(name)))
    
    # Process data in batches
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            _ = model(batch_X)
    
    # Compute importance scores
    importance_scores = {}
    for name in activations_dict:
        # Concatenate all batches
        activations = torch.cat(activations_dict[name], dim=0)
        
        # Compute activation statistics
        mean_activation = torch.mean(activations, dim=0)
        std_activation = torch.std(activations, dim=0)
        
        # Higher mean and lower std indicates consistent, important features
        importance = mean_activation.abs() / (std_activation + 1e-10)
        importance_scores[name] = importance
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()
        
    return importance_scores

def pattern_based_pruning(model, X, pruning_ratio):
    """
    Prune based on activation patterns
    """
    pruned_model = copy.deepcopy(model)
    importance_scores = get_activation_importance(pruned_model, X)
    masks = {}
    
    # Calculate importance for each weight based on input/output activations
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Linear):
            if name in importance_scores:
                output_importance = importance_scores[name]
                
                # For input importance, use previous layer's output importance
                # For first layer, use uniform importance
                prev_name = list(importance_scores.keys())[list(importance_scores.keys()).index(name)-1] if name != list(importance_scores.keys())[0] else None
                input_importance = importance_scores[prev_name] if prev_name else torch.ones(module.weight.shape[1], device=module.weight.device)
                
                # Compute weight importance as outer product of input and output importance
                weight_importance = torch.outer(output_importance, input_importance)
                
                # Get threshold
                threshold = torch.quantile(weight_importance.abs().flatten(), pruning_ratio)
                
                # Create and apply mask
                mask = (weight_importance.abs() >= threshold).float()
                module.weight.data *= mask
                masks[name] = mask
    
    return pruned_model, masks

def magnitude_pruning(model, pruning_ratio):
    """
    Traditional magnitude-based pruning
    """
    pruned_model = copy.deepcopy(model)
    masks = {}
    
    # Get all weights
    all_weights = []
    for name, param in pruned_model.named_parameters():
        if 'weight' in name:
            all_weights.extend(param.data.abs().flatten().cpu().numpy())
    
    # Calculate threshold
    threshold = np.percentile(all_weights, pruning_ratio * 100)
    
    # Apply pruning
    for name, param in pruned_model.named_parameters():
        if 'weight' in name:
            mask = (param.data.abs() >= threshold).float()
            param.data *= mask
            masks[name] = mask
    
    return pruned_model, masks

def extract_n_train(filename):
    """Extract n_train value from filename."""
    match = re.search(r'n(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None

def process_directory(input_dir, output_dir, d, hidden_size, depth, mode, gamma, lr):
    """Process all models in directory with both pruning methods."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all final model files
    print("Looking for model files...")
    model_files = glob.glob(os.path.join(input_dir, f'final_model_h{hidden_size}_d{depth}_n*_lr{lr}_*.pt'))
    print(f"Found {len(model_files)} model files")
    
    # Define pruning ratios
    pruning_ratios = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
    
    results = {
        'original': {'train': {}, 'test': {}},
        'magnitude': {f'pruned_{ratio}': {'train': {}, 'test': {}} for ratio in pruning_ratios},
        'pattern': {f'pruned_{ratio}': {'train': {}, 'test': {}} for ratio in pruning_ratios}
    }
    
    # Load test data
    print("Loading test data...")
    test_data_file = glob.glob(os.path.join(input_dir, 'test_dataset_*.pt'))[0]
    test_data = torch.load(test_data_file)
    X_test, y_test = test_data['X'].to(device), test_data['y'].to(device)
    
    for model_file in sorted(model_files):
        n_train = extract_n_train(model_file)
        if n_train is None:
            continue
        
        print(f"\n{'='*50}")
        print(f"Processing model with n_train = {n_train}")
        
        # Load training data
        print("Loading training data...")
        train_data_file = glob.glob(os.path.join(input_dir, f'train_dataset_h{hidden_size}_d{depth}_n{n_train}_lr{lr}_*.pt'))[0]
        train_data = torch.load(train_data_file)
        X_train, y_train = train_data['X'].to(device), train_data['y'].to(device)
        
        # Load and evaluate original model
        print("Loading and evaluating original model...")
        model = DeepNN(d, hidden_size, depth, mode=mode, gamma=gamma).to(device)
        state_dict = torch.load(model_file)
        model.load_state_dict(state_dict)
        
        results['original']['train'][n_train] = evaluate_model(model, X_train, y_train)
        results['original']['test'][n_train] = evaluate_model(model, X_test, y_test)
        
        # Test different pruning ratios
        for ratio in pruning_ratios:
            print(f"\nTesting pruning ratio: {ratio}")
            
            # Magnitude pruning
            print("Applying magnitude-based pruning...")
            mag_pruned_model, mag_masks = magnitude_pruning(model, ratio)
            results['magnitude'][f'pruned_{ratio}']['train'][n_train] = evaluate_model(mag_pruned_model, X_train, y_train)
            results['magnitude'][f'pruned_{ratio}']['test'][n_train] = evaluate_model(mag_pruned_model, X_test, y_test)
            
            # Pattern-based pruning
            print("Applying pattern-based pruning...")
            pat_pruned_model, pat_masks = pattern_based_pruning(model, X_train, ratio)
            results['pattern'][f'pruned_{ratio}']['train'][n_train] = evaluate_model(pat_pruned_model, X_train, y_train)
            results['pattern'][f'pruned_{ratio}']['test'][n_train] = evaluate_model(pat_pruned_model, X_test, y_test)
            
            # Save pruned models
            save_prefix = f'h{hidden_size}_d{depth}_n{n_train}_lr{lr}_g{gamma}_{mode}'
            torch.save({
                'model_state_dict': mag_pruned_model.state_dict(),
                'masks': mag_masks
            }, os.path.join(output_dir, f'magnitude_pruned_{ratio}_{save_prefix}.pt'))
            
            torch.save({
                'model_state_dict': pat_pruned_model.state_dict(),
                'masks': pat_masks
            }, os.path.join(output_dir, f'pattern_pruned_{ratio}_{save_prefix}.pt'))
    
    # Save results
    results_file = os.path.join(output_dir, f'advanced_pruning_results_true_h400_g1_{timestamp}.pkl')
    print(f"\nSaving results to {results_file}")
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    return results

def main():
    # Set your parameters directly here
    input_dir = "/mnt/users/goringn/NNs_vs_Kernels/stair_function/results/msp_anthro_true_mup_lr0005_gamma_1_modelsaved"
    output_dir = "/mnt/users/goringn/NNs_vs_Kernels/stair_function/results/pruning"
    d = 30
    hidden_size = 400
    depth = 4
    mode = 'mup_pennington'
    gamma = 1.0
    lr = 0.005
    
    print(f"Processing models with parameters:")
    print(f"Hidden size: {hidden_size}")
    print(f"Depth: {depth}")
    print(f"Learning rate: {lr}")
    print(f"Gamma: {gamma}")
    print(f"Mode: {mode}")
    
    results = process_directory(
        input_dir,
        output_dir,
        d,
        hidden_size,
        depth,
        mode,
        gamma,
        lr
    )
    
    print("Pruning analysis complete. Results saved in output directory.")

if __name__ == "__main__":
    main()