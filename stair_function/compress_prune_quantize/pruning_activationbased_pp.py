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
    def __init__(self, d: int, hidden_size: int, depth: int, mode: str = 'mup_pennington', gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma
        torch.set_default_dtype(torch.float32)
        
        layers = []
        prev_dim = d
        for _ in range(depth):
            linear = nn.Linear(prev_dim, hidden_size)
            
            if mode == 'mup_pennington':
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
        if mode == 'mup_pennington':
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
    
    def get_hook(name):
        def hook(module, input, output):
            if name not in activations_dict:
                activations_dict[name] = []
            activations_dict[name].append(output.detach())
        return hook
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(get_hook(name)))
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            _ = model(batch_X)
    
    importance_scores = {}
    for name in activations_dict:
        activations = torch.cat(activations_dict[name], dim=0)
        mean_activation = torch.mean(activations, dim=0)
        std_activation = torch.std(activations, dim=0)
        importance = mean_activation.abs() / (std_activation + 1e-10)
        importance_scores[name] = importance
    
    for hook in hooks:
        hook.remove()
        
    return importance_scores

def pattern_based_pruning(model, X, pruning_ratio):
    pruned_model = copy.deepcopy(model)
    importance_scores = get_activation_importance(pruned_model, X)
    masks = {}
    
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Linear):
            if name in importance_scores:
                output_importance = importance_scores[name]
                prev_name = list(importance_scores.keys())[list(importance_scores.keys()).index(name)-1] if name != list(importance_scores.keys())[0] else None
                input_importance = importance_scores[prev_name] if prev_name else torch.ones(module.weight.shape[1], device=module.weight.device)
                weight_importance = torch.outer(output_importance, input_importance)
                threshold = torch.quantile(weight_importance.abs().flatten(), pruning_ratio)
                mask = (weight_importance.abs() >= threshold).float()
                module.weight.data *= mask
                masks[name] = mask
    
    return pruned_model, masks

def magnitude_pruning(model, pruning_ratio):
    pruned_model = copy.deepcopy(model)
    masks = {}
    
    all_weights = []
    for name, param in pruned_model.named_parameters():
        if 'weight' in name:
            all_weights.extend(param.data.abs().flatten().cpu().numpy())
    
    threshold = np.percentile(all_weights, pruning_ratio * 100)
    
    for name, param in pruned_model.named_parameters():
        if 'weight' in name:
            mask = (param.data.abs() >= threshold).float()
            param.data *= mask
            masks[name] = mask
    
    return pruned_model, masks

def extract_model_params(filename):
    """Extract model parameters from filename."""
    pattern = r'h(\d+)_d(\d+)_n(\d+)_lr([\d\.]+)_'
    match = re.search(pattern, filename)
    if match:
        return {
            'hidden_size': int(match.group(1)),
            'depth': int(match.group(2)),
            'n_train': int(match.group(3)),
            'lr': float(match.group(4))
        }
    return None

def process_directory(input_dir, output_dir, hidden_size, depth, lr, mode, gamma):
    """Process all models in directory with both pruning methods."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all final model files (both shuffled and non-shuffled)
    print("Looking for model files...")
    model_pattern1 = f'final_model_h{hidden_size}_d{depth}_n*_lr{lr}_mup_pennington_shuffled_*_rank*.pt'
    model_pattern2 = f'final_model_h{hidden_size}_d{depth}_n*_lr{lr}_mup_pennington_*_rank*.pt'
    
    model_files1 = glob.glob(os.path.join(input_dir, model_pattern1))
    model_files2 = glob.glob(os.path.join(input_dir, model_pattern2))
    
    # Combine and remove duplicates while preserving order
    model_files = list(dict.fromkeys(model_files1 + model_files2))
    
    print(f"Found {len(model_files)} model files")
    print(f"Patterns searched:")
    print(f"- {model_pattern1}")
    print(f"- {model_pattern2}")
    
    pruning_ratios = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
    
    results = {
        'original': {'train': {}, 'test': {}},
        'magnitude': {f'pruned_{ratio}': {'train': {}, 'test': {}} for ratio in pruning_ratios},
        'pattern': {f'pruned_{ratio}': {'train': {}, 'test': {}} for ratio in pruning_ratios}
    }
    
    for model_file in sorted(model_files):
        params = extract_model_params(model_file)
        if params is None:
            continue
        
        print(f"\n{'='*50}")
        print(f"Processing model with params: {params}")
        
        # Look for both shuffled and non-shuffled dataset patterns
        dataset_pattern1 = f'dataset_h{params["hidden_size"]}_d{params["depth"]}_n{params["n_train"]}_lr{params["lr"]}_mup_pennington_shuffled_*_rank*.pt'
        dataset_pattern2 = f'dataset_h{params["hidden_size"]}_d{params["depth"]}_n{params["n_train"]}_lr{params["lr"]}_mup_pennington_*_rank*.pt'
        
        dataset_files1 = glob.glob(os.path.join(input_dir, dataset_pattern1))
        dataset_files2 = glob.glob(os.path.join(input_dir, dataset_pattern2))
        
        # Combine and remove duplicates
        dataset_files = list(dict.fromkeys(dataset_files1 + dataset_files2))
        
        if not dataset_files:
            print(f"Warning: No matching dataset found for {model_file}")
            continue
            
        dataset_file = dataset_files[0]
        
        # Load dataset
        print("Loading dataset...")
        dataset = torch.load(dataset_file)
        X_train, y_train = dataset['X'].to(device), dataset['y'].to(device)
        
        # Load and evaluate original model
        print("Loading and evaluating original model...")
        model = DeepNN(X_train.shape[1], params['hidden_size'], params['depth'], 
                      mode=mode, gamma=gamma).to(device)
        state_dict = torch.load(model_file)
        model.load_state_dict(state_dict)
        
        # Store original performance
        train_error = evaluate_model(model, X_train, y_train)
        results['original']['train'][params['n_train']] = train_error
        
        # Test different pruning ratios
        for ratio in pruning_ratios:
            print(f"\nTesting pruning ratio: {ratio}")
            
            # Magnitude pruning
            print("Applying magnitude-based pruning...")
            mag_pruned_model, mag_masks = magnitude_pruning(model, ratio)
            results['magnitude'][f'pruned_{ratio}']['train'][params['n_train']] = \
                evaluate_model(mag_pruned_model, X_train, y_train)
            
            # Pattern-based pruning
            print("Applying pattern-based pruning...")
            pat_pruned_model, pat_masks = pattern_based_pruning(model, X_train, ratio)
            results['pattern'][f'pruned_{ratio}']['train'][params['n_train']] = \
                evaluate_model(pat_pruned_model, X_train, y_train)
            
            # Save pruned models
            base_name = os.path.basename(model_file)
            save_prefix = f'h{params["hidden_size"]}_d{params["depth"]}_n{params["n_train"]}_lr{params["lr"]}_g{gamma}_{mode}'
            
            torch.save({
                'model_state_dict': mag_pruned_model.state_dict(),
                'masks': mag_masks
            }, os.path.join(output_dir, f'magnitude_pruned_{ratio}_{save_prefix}.pt'))
            
            torch.save({
                'model_state_dict': pat_pruned_model.state_dict(),
                'masks': pat_masks
            }, os.path.join(output_dir, f'pattern_pruned_{ratio}_{save_prefix}.pt'))
    
    # Save results
    results_file = os.path.join(output_dir, f'pruning_results_false_{timestamp}.pkl')
    print(f"\nSaving results to {results_file}")
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    return results

def main():
    # Set parameters
    input_dir = "/mnt/users/goringn/NNs_vs_Kernels/low_dim_poly/results/low_dim_poly_prune_2801_false"
    output_dir = "/mnt/users/goringn/NNs_vs_Kernels/low_dim_poly/results/pruning"
    
    # Fixed hyperparameters
    hidden_size = 400
    depth = 4
    lr = 0.001
    mode = 'mup_pennington'
    gamma = 1.0
    
    print(f"Processing models with parameters:")
    print(f"Hidden Size: {hidden_size}")
    print(f"Depth: {depth}")
    print(f"Learning Rate: {lr}")
    print(f"Mode: {mode}")
    print(f"Gamma: {gamma}")
    
    results = process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        hidden_size=hidden_size,
        depth=depth,
        lr=lr,
        mode=mode,
        gamma=gamma
    )
    
    print("Pruning analysis complete. Results saved in output directory.")

if __name__ == "__main__":
    main()