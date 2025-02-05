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
        return self.network(x).squeeze() / float(self.gamma)

def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        mse = torch.mean((y_pred - y) ** 2).item()
    return mse

def prune_model(model, pruning_ratio):
    """
    Prune model weights based on magnitude.
    Returns pruned model and mask of retained weights.
    """
    pruned_model = copy.deepcopy(model)
    masks = {}
    
    # Get all weights
    all_weights = []
    for name, param in pruned_model.named_parameters():
        if 'weight' in name:  # Only prune weights, not biases
            all_weights.extend(param.data.abs().flatten().cpu().numpy())
    
    # Calculate threshold
    all_weights = np.array(all_weights)
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
    """Process all models in directory with pruning analysis."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all final model files in input directory
    model_files = glob.glob(os.path.join(input_dir, f'final_model_h{hidden_size}_d{depth}_n*_lr{lr}_*.pt'))
    
    # Define pruning ratios to test
    pruning_ratios = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
    
    results = {
        'original': {'train': {}, 'test': {}},
    }
    for ratio in pruning_ratios:
        results[f'pruned_{ratio}'] = {'train': {}, 'test': {}}
    
    # Load test data
    test_data_file = glob.glob(os.path.join(input_dir, 'test_dataset_*.pt'))[0]
    test_data = torch.load(test_data_file)
    X_test, y_test = test_data['X'].to(device), test_data['y'].to(device)
    
    for model_file in sorted(model_files):
        n_train = extract_n_train(model_file)
        if n_train is None:
            continue
            
        print(f"Processing model with n_train = {n_train}")
        
        # Load training data
        train_data_file = glob.glob(os.path.join(input_dir, f'train_dataset_h{hidden_size}_d{depth}_n{n_train}_lr{lr}_*.pt'))[0]
        train_data = torch.load(train_data_file)
        X_train, y_train = train_data['X'].to(device), train_data['y'].to(device)
        
        # Load and evaluate original model
        model = DeepNN(d, hidden_size, depth, mode=mode, gamma=gamma).to(device)
        state_dict = torch.load(model_file)
        model.load_state_dict(state_dict)
        
        # Evaluate original model
        results['original']['train'][n_train] = evaluate_model(model, X_train, y_train)
        results['original']['test'][n_train] = evaluate_model(model, X_test, y_test)
        
        # Test different pruning ratios
        for ratio in pruning_ratios:
            print(f"Testing pruning ratio: {ratio}")
            pruned_model, masks = prune_model(model, ratio)
            
            # Evaluate pruned model
            results[f'pruned_{ratio}']['train'][n_train] = evaluate_model(pruned_model, X_train, y_train)
            results[f'pruned_{ratio}']['test'][n_train] = evaluate_model(pruned_model, X_test, y_test)
            
            # Save pruned model and masks
            save_prefix = f'h{hidden_size}_d{depth}_n{n_train}_lr{lr}_g{gamma}_{mode}'
            torch.save({
                'model_state_dict': pruned_model.state_dict(),
                'masks': masks
            }, os.path.join(output_dir, f'pruned_model_{ratio}_{save_prefix}.pt'))
    
    # Save results
    results_file = os.path.join(output_dir, f'pruning_results_{timestamp}.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    return results

def main():
    # Set your parameters directly here
    input_dir = "/mnt/users/goringn/NNs_vs_Kernels/stair_function/results/msp_anthro_true_mup_lr0005_gamma_001_modelsaved"
    output_dir = input_dir #"/path/to/output"
    d = 30
    hidden_size = 2000
    depth = 4
    mode = 'mup_pennington'
    gamma = 0.01
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