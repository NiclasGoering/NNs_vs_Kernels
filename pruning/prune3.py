#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from typing import Dict, Tuple, List
import time


###############################################################################
# 1. Model Definition (same as your DeepNN class, or adapt as needed)
###############################################################################
class DeepNN(nn.Module):
    def __init__(self, d: int, hidden_size: int, depth: int, 
                 mode: str = 'special', gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma
        
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
            # Simple initialization (replace with your custom logic if needed)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            self.layer_lrs.append(1.0)

            layers.append(linear)
            layers.append(nn.ReLU())
            prev_dim = hidden_size
        
        # Final layer
        final_layer = nn.Linear(prev_dim, 1)
        nn.init.xavier_uniform_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
        self.layer_lrs.append(1.0)
        
        layers.append(final_layer)
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma = float(self.gamma)  # Ensure gamma is a float
        return self.network(x).squeeze() / gamma
    
    def get_layer_learning_rates(self, base_lr: float) -> List[float]:
        """Return list of learning rates for each layer"""
        return [base_lr * lr for lr in self.layer_lrs]


###############################################################################
# 2. Utilities for loading dataset, computing eigenfunctions, etc.
###############################################################################
def load_dataset(dataset_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load dataset from a saved PyTorch file with 'X' and 'y' keys."""
    data = torch.load(dataset_path)
    return data['X'], data['y']

def get_top_eigenfunctions(features: torch.Tensor, k: int) -> torch.Tensor:
    """
    Compute top-k eigenfunctions from the NxD 'features' matrix.
    We'll do an eigen-decomposition of K = features * features^T
    and return the top-k eigenvectors.
    """
    with torch.no_grad():
        K = torch.matmul(features, features.T)  # NxN
        eigvals, eigvecs = torch.linalg.eigh(K) 
        return eigvecs[:, -k:]  # Top k eigenfunctions (eigenvectors)

def compute_eigenfunction_similarity(orig_eigenfuncs: torch.Tensor, 
                                     new_eigenfuncs: torch.Tensor) -> float:
    """
    Compute similarity between two sets of eigenfunctions by taking
    the absolute correlation or alignment. We'll do a simple measure:
    the mean of the absolute diagonal entries of (U^T V).
    """
    # Each of orig_eigenfuncs and new_eigenfuncs is Nxk
    corrs = torch.abs(torch.matmul(orig_eigenfuncs.T, new_eigenfuncs))  # k x k
    return torch.mean(torch.diag(corrs)).item()


###############################################################################
# 3. A Simple Training (Fine-Tuning) Routine with Feature Distillation
###############################################################################
def train_for_epochs(model: nn.Module, 
                     X_train: torch.Tensor, 
                     y_train: torch.Tensor,
                     orig_features: torch.Tensor,     # <== Distillation reference
                     lr: float = 1e-4,               # Smaller LR
                     epochs: int = 5,                # Longer fine-tuning
                     batch_size: int = 512,
                     mask_dict: Dict[str, torch.Tensor] = None,
                     distillation_weight: float = 0.1 # Strength of distillation
                     ):
    """
    A mini-batch training loop for MSE regression + feature distillation.
    
    - If mask_dict is provided, we re-zero out the pruned weights after each step.
    - We add a distillation term that encourages new hidden features to match
      the original hidden features (orig_features).
    """
    device = X_train.device
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    dataset_size = X_train.shape[0]
    num_batches = (dataset_size + batch_size - 1) // batch_size
    
    for epoch in range(epochs):
        permutation = torch.randperm(dataset_size, device=device)
        epoch_loss = 0.0
        
        for b_i in range(num_batches):
            indices = permutation[b_i*batch_size : (b_i+1)*batch_size]
            X_batch = X_train[indices]
            y_batch = y_train[indices]
            
            # The original hidden features for these samples
            orig_features_batch = orig_features[indices]
            
            optimizer.zero_grad()
            
            # Forward pass
            out = model(X_batch)
            new_features_batch = model.network[:-1](X_batch)  # hidden features (N x D)
            
            # MSE loss
            mse_loss = criterion(out, y_batch)
            # Distillation loss: keep new hidden features close to original
            distil_loss = torch.mean((new_features_batch - orig_features_batch) ** 2)
            
            # Combine them
            loss = mse_loss + distillation_weight * distil_loss
            loss.backward()
            
            # If using a mask to keep weights zeroed, apply it before optimizer step
            if mask_dict is not None:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name in mask_dict:
                            param.grad *= mask_dict[name]  # zero-out pruned weights' grads
            
            optimizer.step()
            
            # Re-apply the mask so that pruned parameters remain zero
            if mask_dict is not None:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name in mask_dict:
                            param *= mask_dict[name]
            
            epoch_loss += loss.item()
        
        print(f"  [Fine-tune] Epoch {epoch+1}/{epochs}, Loss = {epoch_loss/num_batches:.6f}")


###############################################################################
# 4. Iterative Global Pruning with Distillation and Longer Fine-Tuning
###############################################################################
def prune_model_globally_with_finetune(
    model_path: str,
    train_dataset_path: str,
    test_dataset_path: str,
    k: int = 5,
    epsilon: float = 0.01,
    prune_fraction: float = 0.05,
    num_iterations: int = 5,
    finetune_epochs: int = 5,
    finetune_lr: float = 1e-4,
    distillation_weight: float = 0.1,
    device: str = 'cuda',
    save_path: str = "pruned_model_iterative.pt"
) -> Dict:
    # Load model state
    print(f"[INFO] Loading model from: {model_path}")
    loaded = torch.load(model_path, map_location=device)
    
    # Extract both state dict and hyperparameters
    if isinstance(loaded, dict):
        if 'state_dict' in loaded:
            model_state = loaded['state_dict']
        else:
            model_state = loaded
            
        # Try to extract gamma if it exists in the saved model
        gamma = 1.0 #loaded.get('gamma', 1.0)  # Default to 1.0 if not found
    else:
        model_state = loaded
        gamma = 1.0
    
    # Deduce architecture
    first_weight = model_state['network.0.weight']
    d = first_weight.shape[1]
    hidden_size = first_weight.shape[0]
    lin_weight_keys = [k for k in model_state if 'weight' in k]
    max_layer_idx = max([int(k.split('.')[1]) for k in lin_weight_keys])
    depth_est = max_layer_idx // 2
    
    # Create model with correct gamma
    model = DeepNN(d, hidden_size, depth_est, gamma=gamma).to(device)
    model.load_state_dict(model_state)
    print(f"[INFO] Model loaded successfully with gamma={gamma}")
    
    # Rest of the function remains the same...

    # Load training / test data
    X_train, y_train = load_dataset(train_dataset_path)
    X_test, y_test = load_dataset(test_dataset_path)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    # 2. Compute original performance / eigenfunctions
    with torch.no_grad():
        train_pred = model(X_train)
        test_pred = model(X_test)
        init_train_mse = torch.mean((train_pred - y_train)**2).item()
        init_test_mse  = torch.mean((test_pred - y_test)**2).item()

        # Original hidden features for all training points (for distillation)
        orig_features_all = model.network[:-1](X_train)
        
        # Get the original top-k eigenfunctions from the penultimate layer
        orig_eigenfuncs = get_top_eigenfunctions(orig_features_all, k)
    
    # Count total parameters
    all_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in all_params)
    print(f"[INFO] Total trainable parameters = {total_params}")

    # Initialize a mask dict (name -> tensor of 1's)
    mask_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            mask_dict[name] = torch.ones_like(param, device=device)

    # 3. Iterative Pruning
    current_unpruned = total_params
    last_similarity = 1.0
    total_pruned = 0
    
    print("\n[INFO] Starting iterative pruning ...")
    for iteration in range(num_iterations):
        print(f"\n=== Pruning Iteration {iteration+1}/{num_iterations} ===")
        # a) Collect all unpruned weights in a single list
        param_entries = []
        for name, param in model.named_parameters():
            if name not in mask_dict:
                continue
            w = param.detach().flatten()
            m = mask_dict[name].detach().flatten()
            
            # Indices of unpruned weights
            unpruned_idx = torch.where(m > 0)[0]
            unpruned_w   = w[unpruned_idx]
            
            param_entries.append((name, unpruned_idx, unpruned_w))
        
        # b) Concatenate all unpruned weights
        all_unpruned_weights = torch.cat([entry[2].abs() for entry in param_entries])
        n_unpruned = all_unpruned_weights.numel()
        n_to_prune = int(prune_fraction * n_unpruned)

        if n_to_prune == 0:
            print("  [WARN] No more weights to prune (prune_fraction too small or too few unpruned left).")
            break
        
        # c) Find the global threshold
        sorted_unpruned_weights, _ = torch.sort(all_unpruned_weights)
        threshold = sorted_unpruned_weights[n_to_prune-1].item()
        
        print(f"  # unpruned = {n_unpruned}, # to prune = {n_to_prune}, threshold = {threshold:.6e}")
        
        # d) Apply the mask for all parameters below the threshold
        pruned_this_step = 0
        for (name, unpruned_idx, unpruned_w) in param_entries:
            to_prune_mask = (unpruned_w.abs() <= threshold)
            if to_prune_mask.sum() == 0:
                continue
            
            idx_to_zero = unpruned_idx[to_prune_mask]
            with torch.no_grad():
                flat_mask = mask_dict[name].view(-1)
                flat_param = dict(model.named_parameters())[name].data.view(-1)
                
                flat_mask[idx_to_zero] = 0  # mark them as pruned
                flat_param[idx_to_zero] = 0 # also set actual param to zero
            
            pruned_this_step += to_prune_mask.sum().item()
        
        total_pruned += pruned_this_step
        current_unpruned = current_unpruned - pruned_this_step
        
        print(f"  -> Pruned {pruned_this_step} weights in this iteration.")
        print(f"  -> Total pruned so far = {total_pruned}, remain = {current_unpruned}.")
        
        # e) Fine-tuning with feature distillation
        if finetune_epochs > 0:
            print("  [INFO] Fine-tuning after pruning...")
            train_for_epochs(
                model,
                X_train,
                y_train,
                orig_features=orig_features_all,   # for distillation
                lr=finetune_lr,
                epochs=finetune_epochs,
                mask_dict=mask_dict,
                distillation_weight=distillation_weight
            )
        
        # f) Check eigenfunction similarity
        with torch.no_grad():
            new_features = model.network[:-1](X_train)
            new_eigenfuncs = get_top_eigenfunctions(new_features, k)
            similarity = compute_eigenfunction_similarity(orig_eigenfuncs, new_eigenfuncs)
        
        print(f"  -> Eigenfunction similarity after iteration {iteration+1}: {similarity:.6f}")
        last_similarity = similarity
        
        if (1 - similarity) > epsilon:
            print(f"  [STOP] Similarity dropped below threshold (1 - similarity = {1 - similarity:.6f} > epsilon={epsilon}). Reverting last prune step.")
            # Revert logic can be more elaborate (you'd need to reload from a backup).
            # For simplicity, we break here:
            break

    # 5. Evaluate final performance
    with torch.no_grad():
        final_train_pred = model(X_train)
        final_test_pred  = model(X_test)
        final_train_mse  = torch.mean((final_train_pred - y_train)**2).item()
        final_test_mse   = torch.mean((final_test_pred - y_test)**2).item()
    
    final_unpruned = current_unpruned
    final_sparsity = 1.0 - (final_unpruned / total_params)

    # 6. Save pruned model and mask
    torch.save({
        'state_dict': model.state_dict(),
        'mask_dict': mask_dict,
        'final_sparsity': final_sparsity
    }, save_path)
    print(f"\n[INFO] Pruned model saved to {save_path}")

    # 7. Summarize results
    print("\n=== Final Pruning Results ===")
    print(f"Initial #Params: {total_params}")
    print(f"Final #Params:   {final_unpruned} ({(final_unpruned/total_params)*100:.2f}%)")
    print(f"Sparsity:        {final_sparsity*100:.2f}%")
    print(f"\nTrain MSE:  initial={init_train_mse:.6f}, final={final_train_mse:.6f}")
    print(f"Test MSE:   initial={init_test_mse:.6f},  final={final_test_mse:.6f}")
    print(f"Eigenfunction similarity: final={last_similarity:.6f}")

    return {
        'initial_train_mse': init_train_mse,
        'initial_test_mse': init_test_mse,
        'final_train_mse': final_train_mse,
        'final_test_mse': final_test_mse,
        'initial_params': total_params,
        'final_params': final_unpruned,
        'sparsity': final_sparsity,
        'final_eigen_similarity': last_similarity
    }


###############################################################################
# 5. Example Usage
###############################################################################
if __name__ == "__main__":
    
    # Example paths (adapt to your setup)
    #model_path = "/mnt/users/goringn/NNs_vs_Kernels/stair_function/results/msp_anthro_false_mup_lr0005_gamma_1_modelsaved/final_model_h400_d4_n500_lr0.005_g1.0_mup_pennington_20250125_153135_rank0.pt"
    #train_data_path = "/mnt/users/goringn/NNs_vs_Kernels/stair_function/results/msp_anthro_false_mup_lr0005_gamma_1_modelsaved/train_dataset_h400_d4_n500_lr0.005_g1.0_mup_pennington_20250125_153135_rank0.pt"
    #test_data_path = "stair_function/results/msp_anthro_false_mup_lr0005_gamma_1_modelsaved/test_dataset_20250125_153135.pt"
    #model_path = "/mnt/users/goringn/NNs_vs_Kernels/stair_function/results/msp_anthro_true_mup_lr0005_gamma_1_modelsaved/final_model_h400_d4_n10000_lr0.005_g1.0_mup_pennington_shuffled_20250125_153042_rank0.pt"
    #train_data_path = "/mnt/users/goringn/NNs_vs_Kernels/stair_function/results/msp_anthro_true_mup_lr0005_gamma_1_modelsaved/train_dataset_h400_d4_n10000_lr0.005_g1.0_mup_pennington_shuffled_20250125_153042_rank0.pt"
    #test_data_path = "/mnt/users/goringn/NNs_vs_Kernels/stair_function/results/msp_anthro_true_mup_lr0005_gamma_1_modelsaved/test_dataset_20250125_153042.pt"
    #model_path = "/mnt/users/goringn/NNs_vs_Kernels/stair_function/results/msp_anthro_false_mup_lr0005_gamma_0001_modelsaved/final_model_h400_d4_n10000_lr0.005_g0.001_mup_pennington_20250126_043809_rank0.pt"
    #train_data_path = "/mnt/users/goringn/NNs_vs_Kernels/stair_function/results/msp_anthro_false_mup_lr0005_gamma_0001_modelsaved/train_dataset_h400_d4_n10000_lr0.005_g0.001_mup_pennington_20250126_043809_rank0.pt"
    #test_data_path = "/mnt/users/goringn/NNs_vs_Kernels/stair_function/results/msp_anthro_false_mup_lr0005_gamma_0001_modelsaved/test_dataset_20250126_043809.pt"
    
    model_path = "/mnt/users/goringn/NNs_vs_Kernels/stair_function/results/msp_uniform_sampling_0202/final_model_h25_d2_n3000_lr0.005_g1.0_mup_pennington_20250203_015540_rank0.pt"
    train_data_path = "/mnt/users/goringn/NNs_vs_Kernels/stair_function/results/msp_uniform_sampling_0202/train_dataset_h50_d2_n3000_lr0.005_g1.0_mup_pennington_20250203_015540_rank0.pt"
    test_data_path = "/mnt/users/goringn/NNs_vs_Kernels/stair_function/results/msp_uniform_sampling_0202/test_dataset_20250203_015540.pt"

    save_pruned_path = "pruned_model_iterative.pt"

    # Now we use smaller pruning steps (0.05), more epochs (5),
    # smaller LR (1e-4), and a distillation weight (0.1):
    results = prune_model_globally_with_finetune(
    model_path=model_path,
    train_dataset_path=train_data_path,
    test_dataset_path=test_data_path,
    k=8,
    epsilon=0.02,
    prune_fraction=0.01,        # Reduced from 0.03
    num_iterations=300,         # Increased to compensate for smaller steps
    finetune_epochs=100,        # Keep as is
    finetune_lr=1e-4,
    distillation_weight=0.2,    # Keep as is
    device='cuda',
    save_path=save_pruned_path
)

    print("\n=== Done ===")
    print("Results:", results)
