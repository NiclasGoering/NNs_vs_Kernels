import torch
import torch.nn as nn
import os
import json
import torch.func
import numpy as np
from glob import glob
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from typing import Optional

class DeepNN(nn.Module):
    def __init__(self, d: int, hidden_size: int, depth: int, mode: str = 'special'):
        super().__init__()
        self.mode = mode
        self.depth = depth
        self.hidden_size = hidden_size
        self.input_dim = d
        
        layers = []
        prev_dim = d
        self.layer_lrs = []
        
        for layer_idx in range(depth):
            linear = nn.Linear(prev_dim, hidden_size)
            
            if mode == 'mup_pennington':
                if layer_idx == 0:
                    std = 1.0 / np.sqrt(prev_dim)
                    lr_scale = 1.0
                else:
                    std = 1.0 / np.sqrt(prev_dim)
                    lr_scale = 1.0 / prev_dim
                nn.init.normal_(linear.weight, mean=0.0, std=std)
                nn.init.zeros_(linear.bias)
                self.layer_lrs.append(lr_scale)
            elif mode == 'special':
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
            else:  # standard
                nn.init.xavier_uniform_(linear.weight)
                nn.init.zeros_(linear.bias)
                self.layer_lrs.append(1.0)
            
            layers.extend([linear, nn.ReLU()])
            prev_dim = hidden_size
        
        final_layer = nn.Linear(prev_dim, 1)
        if mode == 'mup_pennington':
            std = 1.0 / np.sqrt(prev_dim)
            lr_scale = 1.0 / prev_dim
            nn.init.normal_(final_layer.weight, std=std)
            self.layer_lrs.append(lr_scale)
        elif mode == 'special':
            nn.init.normal_(final_layer.weight, std=0.01)
            self.layer_lrs.append(1.0)
        elif mode == 'spectral':
            fan_in = prev_dim
            fan_out = 1
            std = (1.0 / np.sqrt(fan_in)) * min(1.0, np.sqrt(fan_out / fan_in))
            nn.init.normal_(final_layer.weight, std=std)
            self.layer_lrs.append(float(fan_out) / fan_in)
        else:
            nn.init.xavier_uniform_(final_layer.weight)
            self.layer_lrs.append(1.0)
            
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze()


def empirical_ntk_implicit_batched(func, params, x1, x2, batch_size=100, compute='full'):
    device = x1.device
    n1, n2 = x1.shape[0], x2.shape[0]
    
    # Keep params as dictionary throughout
    param_keys = list(params.keys())
    
    def get_ntk_slice_batch(x1_batch, x2_batch):
        def func_x1(param_dict):
            # Ensure output has proper dimensions
            out = func(param_dict, x1_batch)
            return out.unsqueeze(-1) if out.ndim == 1 else out

        def func_x2(param_dict):
            # Ensure output has proper dimensions
            out = func(param_dict, x2_batch)
            return out.unsqueeze(-1) if out.ndim == 1 else out

        output, vjp_fn = torch.func.vjp(func_x1, params)
        
        def get_ntk_slice(vec):
            vjps = vjp_fn(vec)[0]
            _, jvps = torch.func.jvp(func_x2, (params,), (vjps,))
            return jvps

        basis = torch.eye(output.numel(), device=device)
        return torch.vmap(get_ntk_slice)(basis)

    # Process in batches
    result = torch.zeros((n1, n2), device=device)
    
    for i in range(0, n1, batch_size):
        i_end = min(i + batch_size, n1)
        for j in range(0, n2, batch_size):
            j_end = min(j + batch_size, n2)
            
            # Clear cache before batch computation
            torch.cuda.empty_cache()
            
            x1_batch = x1[i:i_end]
            x2_batch = x2[j:j_end]
            
            # Get batch result and ensure proper dimensions
            batch_result = get_ntk_slice_batch(x1_batch, x2_batch)
            
            # Extract the scalar value for each pair of inputs
            if compute == 'full':
                batch_result = batch_result.squeeze(-1).squeeze(-1)
            
            result[i:i_end, j:j_end] = batch_result
    
    return result


def compute_cka(K1, K2):
    """Compute Centered Kernel Alignment between two kernel matrices."""
    # Center the kernel matrices
    def center(K):
        n = K.shape[0]
        I = torch.eye(n, device=K.device)
        H = I - torch.ones_like(K) / n
        return H @ K @ H
    
    K1_centered = center(K1)
    K2_centered = center(K2)
    
    # Compute CKA
    hsic = torch.trace(K1_centered @ K2_centered)
    norm1 = torch.sqrt(torch.trace(K1_centered @ K1_centered))
    norm2 = torch.sqrt(torch.trace(K2_centered @ K2_centered))
    
    return hsic / (norm1 * norm2)

def compute_norm_frobenius(K1, K2):
    """Compute normalized Frobenius norm between two kernel matrices."""
    diff = K1 - K2
    frob_norm = torch.norm(diff, p='fro')
    norm_factor = torch.sqrt(torch.numel(K1))
    return frob_norm / norm_factor

@dataclass
class ModelConfig:
    hidden_size: int
    depth: int
    n_train: int
    learning_rate: float
    mode: str
    timestamp: str
    rank: int

class ModelLoader:
    @staticmethod
    def parse_model_name(filename: str) -> Optional[ModelConfig]:
        """Parse a model filename into its components."""
        # Remove 'final_model_' prefix and '.pt' suffix
        name = filename.replace('final_model_', '').replace('.pt', '')
        
        # Split into components
        parts = name.split('_')
        try:
            # Extract values from parts
            hidden_size = int(parts[0].replace('h', ''))
            depth = int(parts[1].replace('d', ''))
            n_train = int(parts[2].replace('n', ''))
            lr = float(parts[3].replace('lr', ''))
            mode = '_'.join(parts[4:6])  # mup_pennington
            timestamp = '_'.join(parts[6:8])  # 20250116_171034
            rank = int(parts[8].replace('rank', ''))
            
            return ModelConfig(
                hidden_size=hidden_size,
                depth=depth,
                n_train=n_train,
                learning_rate=lr,
                mode=mode,
                timestamp=timestamp,
                rank=rank
            )
        except Exception as e:
            print(f"Error parsing filename {filename}: {str(e)}")
            return None

    @staticmethod
    def get_model_files(base_path: str) -> List[Tuple[str, ModelConfig]]:
        """Get all model files and their configurations."""
        files = glob(os.path.join(base_path, 'final_model_*.pt'))
        results = []
        
        for file in files:
            filename = os.path.basename(file)
            config = ModelLoader.parse_model_name(filename)
            if config:
                results.append((file, config))
        
        return results

def main():
    # Base paths
    base_path = "/mnt/users/goringn/NNs_vs_Kernels/stair_function/results/msp_1601_NTK_norm_mup"
    output_path = os.path.join(base_path, "ntk_results")
    
    print("\nStarting NTK computation script")
    print(f"Base path: {base_path}")
    print(f"Output path: {output_path}")
    
    try:
        # Load hyperparameters
        hyperparams_file = os.path.join(base_path, 'hyperparameters_20250116_171034.json')
        if not os.path.exists(hyperparams_file):
            raise ValueError(f"Hyperparameters file not found at {hyperparams_file}")
            
        with open(hyperparams_file, 'r') as f:
            hyperparams = json.load(f)
        print("\nLoaded hyperparameters:", json.dumps(hyperparams, indent=2))
        
        # Get and process model files
        print("\nSearching for model configurations...")
        model_files = ModelLoader.get_model_files(base_path)
        
        if not model_files:
            raise ValueError("No valid model files found")
            
        print(f"\nFound {len(model_files)} valid model configurations")
        for filepath, config in model_files:
            print(f"\nModel: {os.path.basename(filepath)}")
            print(f"Config: {config}")
        
        # Process each model
        all_metrics = []
        for filepath, config in model_files:
            print(f"\nProcessing model: {os.path.basename(filepath)}")
            
            # Load datasets
            test_data_path = os.path.join(base_path, f'test_dataset_{config.timestamp}.pt')
            train_data_path = os.path.join(base_path, 
                f'train_dataset_h{config.hidden_size}_d{config.depth}_n{config.n_train}_lr{config.learning_rate}_{config.mode}_{config.timestamp}_rank{config.rank}.pt')
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load test dataset
            test_dataset = torch.load(test_data_path)
            X_test = test_dataset['X'].to(device)
            
            # Load train dataset
            train_dataset = torch.load(train_data_path)
            X_train = train_dataset['X'].to(device)
            
            # Initialize model
            d = X_train.shape[1]
            model = DeepNN(d, config.hidden_size, config.depth, mode=config.mode).to(device)
            
            # Get model paths
            model_prefix = f'h{config.hidden_size}_d{config.depth}_n{config.n_train}_lr{config.learning_rate}_{config.mode}_{config.timestamp}_rank{config.rank}'
            initial_model_path = os.path.join(base_path, f'initial_model_{model_prefix}.pt')
            final_model_path = filepath
            
            # Compute NTKs and metrics
            try:
                # Load models
                initial_model_weights = torch.load(initial_model_path)
                final_model_weights = torch.load(final_model_path)
                
                def compute_model_ntk(model_weights):
                    model.load_state_dict(model_weights)
                    
                    # Keep parameters as dictionary
                    params = dict(model.named_parameters())
                    
                    # Define a functional version of the model
                    def fnet_single(params_dict, x):
                        # Use functional call with dictionary parameters
                        x = x.unsqueeze(0) if x.ndim == 1 else x
                        return torch.func.functional_call(model, params_dict, (x,)).squeeze()
                    
                    # Compute NTK with batching
                    try:
                        return empirical_ntk_implicit_batched(fnet_single, params, X_train, X_test, 
                                                            batch_size=50,  # Adjust this based on your GPU memory
                                                            compute='full')
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            # If OOM occurs, clear cache and try with smaller batch
                            torch.cuda.empty_cache()
                            return empirical_ntk_implicit_batched(fnet_single, params, X_train, X_test,
                                                                batch_size=10,  # Much smaller batch size
                                                                compute='full')
                        else:
                            raise

                
                
                initial_ntk = compute_model_ntk(initial_model_weights)
                final_ntk = compute_model_ntk(final_model_weights)
                
                # Calculate metrics
                n_train = X_train.shape[0]
                initial_kernel = initial_ntk[:n_train, :n_train, 0, 0]
                final_kernel = final_ntk[:n_train, :n_train, 0, 0]
                
                cka = compute_cka(initial_kernel, final_kernel)
                norm_frob = compute_norm_frobenius(initial_kernel, final_kernel)
                
                # Save NTKs
                os.makedirs(output_path, exist_ok=True)
                np.save(os.path.join(output_path, f'initial_ntk_{model_prefix}.npy'), initial_ntk.cpu().numpy())
                np.save(os.path.join(output_path, f'final_ntk_{model_prefix}.npy'), final_ntk.cpu().numpy())
                
                metrics = {
                    'cka': cka.item(),
                    'norm_frob': norm_frob.item(),
                    'n_train': config.n_train,
                    'hidden_size': config.hidden_size,
                    'depth': config.depth,
                    'learning_rate': config.learning_rate,
                    'mode': config.mode
                }
                
                all_metrics.append(metrics)
                print(f"Successfully computed metrics: CKA={cka.item():.4f}, Norm={norm_frob.item():.4f}")
                
            except Exception as e:
                print(f"Error processing model {model_prefix}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # Organize and save results
        if all_metrics:
            grouped_results = defaultdict(list)
            for metrics in all_metrics:
                key = (metrics['hidden_size'], metrics['depth'], metrics['learning_rate'], metrics['mode'])
                grouped_results[key].append((metrics['n_train'], metrics['cka'], metrics['norm_frob']))
            
            for params, results in grouped_results.items():
                hidden_size, depth, lr, mode = params
                filename = f'metrics_h{hidden_size}_d{depth}_lr{lr}_{mode}.npy'
                results_array = np.array(results)
                np.save(os.path.join(output_path, filename), results_array)
                
                print(f"\nResults for h={hidden_size}, d={depth}, lr={lr}, mode={mode}:")
                print("n_train\tCKA\tNorm")
                for n_train, cka, norm in sorted(results, key=lambda x: x[0]):
                    print(f"{n_train}\t{cka:.4f}\t{norm:.4f}")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        print("\nTraceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()