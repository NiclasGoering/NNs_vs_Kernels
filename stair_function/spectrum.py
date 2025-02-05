import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Set, Tuple, Dict
import torch.nn as nn
import os
from scipy.linalg import eigh
import copy
from scipy.linalg import eigh
import copy
from torch.func import functional_call, vmap, vjp, jvp, jacrev
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Set, Tuple, Dict
import torch.nn as nn
import os
from scipy.linalg import eigh
import copy
from torch.func import functional_call, vmap, vjp, jvp, jacrev

class MSPFunction:
    def __init__(self, P: int, sets: List[Set[int]]):
        self.P = P
        self.sets = sets
    
    def evaluate(self, z: torch.Tensor) -> torch.Tensor:
        device = z.device
        batch_size = z.shape[0]
        result = torch.zeros(batch_size, dtype=torch.float32, device=device)
        
        # Vectorized computation for all sets
        for S in self.sets:
            indices = torch.tensor(list(S), device=device)
            selected_features = z[:, indices]
            term = torch.prod(selected_features, dim=1)
            result += term
            
        return result

class DeepNN(nn.Module):
    def __init__(self, d: int, hidden_size: int, depth: int, mode: str = 'standard'):
        super().__init__()
        
        torch.set_default_dtype(torch.float32)
        self.mode = mode
        self.depth = depth
        self.hidden_size = hidden_size
        self.input_dim = d
        
        # Use Sequential to maintain compatibility with saved models
        layers = []
        prev_dim = d
        
        for _ in range(depth):
            linear = nn.Linear(prev_dim, hidden_size)
            if mode == 'standard':
                nn.init.normal_(linear.weight, std=1.0)
                nn.init.zeros_(linear.bias)
            
            layers.extend([linear, nn.ReLU()])
            prev_dim = hidden_size
        
        final_layer = nn.Linear(prev_dim, 1)
        nn.init.normal_(final_layer.weight, std=1.0)
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze()

    def compute_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        features = x
        for layer in list(self.network.children())[:-1]:  # All layers except the last
            features = layer(features)
        return features

def generate_datasets(P: int, d: int, n_train: int, n_test: int, msp: MSPFunction, seed: int = 42):
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate data directly on GPU
    X_train = (2 * torch.bernoulli(0.5 * torch.ones((n_train, d), device=device)) - 1)
    y_train = msp.evaluate(X_train)
    
    X_test = (2 * torch.bernoulli(0.5 * torch.ones((n_test, d), device=device)) - 1)
    y_test = msp.evaluate(X_test)
    
    return X_train, y_train, X_test, y_test

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
    
    # Process in batches
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
    
    return nngp.detach().cpu().numpy(), ntk.detach().cpu().numpy()

@torch.no_grad()
def compute_spectrum(kernel: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    # Move computation to GPU if kernel is large
    if kernel.size > 1000000:  # Threshold for GPU usage
        kernel_torch = torch.from_numpy(kernel).cuda()
        eigenvals, eigenvecs = torch.linalg.eigh(kernel_torch)
        eigenvals = eigenvals.cpu().numpy()
        eigenvecs = eigenvecs.cpu().numpy()
    else:
        eigenvals, eigenvecs = eigh(kernel)
    
    idx = eigenvals.argsort()[::-1][:k]
    return eigenvals[idx], eigenvecs[:, idx]

def compute_eigenfunction_values(kernel: np.ndarray, eigenvecs: np.ndarray, y_train: np.ndarray):
    # Move computations to GPU for large matrices
    if kernel.size > 1000000:
        kernel_torch = torch.from_numpy(kernel).cuda()
        eigenvecs_torch = torch.from_numpy(eigenvecs).cuda()
        y_train_torch = torch.from_numpy(y_train).cuda().reshape(-1, 1)
        
        coeffs = torch.matmul(eigenvecs_torch.T, y_train_torch)
        function_values = torch.matmul(kernel_torch, eigenvecs_torch)
        result = function_values * coeffs.T
        
        return result.cpu().numpy()
    else:
        y_train = y_train.reshape(-1, 1)
        coeffs = np.dot(eigenvecs.T, y_train)
        function_values = np.dot(kernel, eigenvecs)
        return function_values * coeffs.T

def plot_spectra_comparison(spectra: Dict[str, np.ndarray], save_path: str, title: str):
    """Plot spectrum comparison for multiple models"""
    plt.figure(figsize=(12, 6))
    
    styles = {
        'initial': '--',
        'final': '-'
    }
    colors = {
        'standard': 'blue',
        'shuffled': 'red'
    }
    
    for model_name, spectrum in spectra.items():
        is_initial = 'initial' in model_name
        is_shuffled = 'shuffled' in model_name
        style = styles['initial'] if is_initial else styles['final']
        color = colors['shuffled'] if is_shuffled else colors['standard']
        
        plt.semilogy(np.arange(1, len(spectrum) + 1), spectrum, 
                    linestyle=style, color=color, 
                    label=model_name, linewidth=2)
    
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue (log scale)')
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_eigenfunctions_comparison(eigenfuncs_dict: Dict[str, np.ndarray], 
                                 save_path: str, title: str, eigenfunction_idx: int):
    """Plot comparison of a specific eigenfunction across models"""
    plt.figure(figsize=(12, 6))
    
    # Calculate common bins
    all_values = np.concatenate([ef[:, eigenfunction_idx] for ef in eigenfuncs_dict.values()])
    min_val = np.min(all_values)
    max_val = np.max(all_values)
    bins = np.linspace(min_val, max_val, 50)
    
    styles = {
        'initial': '--',
        'final': '-'
    }
    colors = {
        'standard': 'blue',
        'shuffled': 'red'
    }
    
    for model_name, eigenfuncs in eigenfuncs_dict.items():
        is_initial = 'initial' in model_name
        is_shuffled = 'shuffled' in model_name
        style = styles['initial'] if is_initial else styles['final']
        color = colors['shuffled'] if is_shuffled else colors['standard']
        
        plt.hist(eigenfuncs[:, eigenfunction_idx], bins=bins,
                histtype='step', linewidth=2,
                linestyle=style, color=color,
                density=True, label=model_name)
    
    plt.xlabel('Eigenfunction Value')
    plt.ylabel('Density')
    plt.title(f'{title} - Eigenfunction {eigenfunction_idx + 1}')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_models(model_paths: Dict[str, str], 
                  X_train: torch.Tensor, y_train: torch.Tensor,
                  X_test: torch.Tensor, y_test: torch.Tensor,
                  d: int, hidden_size: int, depth: int,
                  k_spectrum: int, k_eigenfuncs: int, output_dir: str,
                  batch_size: int = 32):
    """Analyze and compare multiple models with GPU optimization"""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move data to GPU once
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    # Storage for results
    nngp_spectra = {}
    ntk_spectra = {}
    nngp_eigenfuncs = {}
    ntk_eigenfuncs = {}
    
    # Analyze each model
    for model_name, model_path in model_paths.items():
        print(f"Analyzing {model_name}...")
        
        # Load model to GPU and ensure parameters require gradients
        model = DeepNN(d, hidden_size, depth).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        # Ensure parameters require gradients
        for param in model.parameters():
            param.requires_grad_(True)
        
        # Compute empirical kernels with batching
        nngp_train, ntk_train = compute_empirical_kernels(model, X_train, X_train, batch_size)
        
        # Compute spectra
        nngp_vals, nngp_vecs = compute_spectrum(nngp_train, k_spectrum)
        ntk_vals, ntk_vecs = compute_spectrum(ntk_train, k_spectrum)
        
        # Store spectra
        nngp_spectra[model_name] = nngp_vals
        ntk_spectra[model_name] = ntk_vals
        
        # Compute eigenfunctions (using only k_eigenfuncs)
        if k_eigenfuncs > 0:
            nngp_vecs_reduced = nngp_vecs[:, :k_eigenfuncs]
            ntk_vecs_reduced = ntk_vecs[:, :k_eigenfuncs]
            
            nngp_funcs = compute_eigenfunction_values(nngp_train, nngp_vecs_reduced, y_train.cpu().numpy())
            ntk_funcs = compute_eigenfunction_values(ntk_train, ntk_vecs_reduced, y_train.cpu().numpy())
            
            nngp_eigenfuncs[model_name] = nngp_funcs
            ntk_eigenfuncs[model_name] = ntk_funcs
    
    # Plot spectra comparisons
    plot_spectra_comparison(
        nngp_spectra,
        os.path.join(output_dir, 'empirical_nngp_spectra_comparison.png'),
        'Empirical NNGP Kernel Spectra Comparison'
    )
    plot_spectra_comparison(
        ntk_spectra,
        os.path.join(output_dir, 'empirical_ntk_spectra_comparison.png'),
        'Empirical NTK Kernel Spectra Comparison'
    )
    
    # Plot eigenfunction comparisons
    if k_eigenfuncs > 0:
        for i in range(k_eigenfuncs):
            plot_eigenfunctions_comparison(
                nngp_eigenfuncs,
                os.path.join(output_dir, f'empirical_nngp_eigenfunction_{i+1}_comparison.png'),
                'Empirical NNGP', 
                i
            )
            plot_eigenfunctions_comparison(
                ntk_eigenfuncs,
                os.path.join(output_dir, f'empirical_ntk_eigenfunction_{i+1}_comparison.png'),
                'Empirical NTK', 
                i
            )

if __name__ == "__main__":
    # Enable GPU optimization
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("CUDA is available. Running on GPU.")
    else:
        print("CUDA is not available. Running on CPU.")
    
    # Model parameters
    P = 8
    d = 30
    hidden_size = 400
    depth = 4
    k_spectrum = 20000  # For computing and plotting spectrum
    k_eigenfuncs = 20  # For computing and plotting eigenfunctions
    
    # Dataset parameters
    n_train = 20000
    n_test = 10000
    
    # Define MSP sets
    msp_sets = [{7}, {2,7}, {0,2,7}, {5,7,4}, {1}, {0,4}, {3,7}, {0,1,2,3,4,6,7}]
    
    # Initialize MSP function
    msp = MSPFunction(P, msp_sets)
    
    # Generate datasets
    print("Generating datasets...")
    X_train, y_train, X_test, y_test = generate_datasets(P, d, n_train, n_test, msp)
    
    # Model paths
    model_paths = {
        'standard_initial': '/mnt/users/goringn/NNs_vs_Kernels/stair_function/results/msp_standard_false3/initial_model_h400_d4_n20000_lr0.05_standard_20241223_174158_rank0.pt',
        'standard_final': '/mnt/users/goringn/NNs_vs_Kernels/stair_function/results/msp_standard_false3/final_model_h400_d4_n20000_lr0.05_standard_20241223_174158_rank0.pt',
        'shuffled_initial': '/mnt/users/goringn/NNs_vs_Kernels/stair_function/results/msp_NN_grid_1612_nogrokk_mup_pennington/initial_model_h400_d4_n10000_lr0.05_mup_pennington_20241219_163433_rank61.pt',
        'shuffled_final': '/mnt/users/goringn/NNs_vs_Kernels/stair_function/results/msp_NN_grid_1612_nogrokk_mup_pennington/final_model_h400_d4_n10000_lr0.05_mup_pennington_20241219_163433_rank61.pt'
    }
    
    output_dir = "/mnt/users/goringn/NNs_vs_Kernels/stair_function/empirical_kernel_analysis_comparisonmup"
    
    # Run analysis with batching
    print(f"Starting analysis... Output will be saved to {output_dir}")
    analyze_models(
        model_paths, 
        X_train, y_train, 
        X_test, y_test,
        d, hidden_size, depth, 
        k_spectrum, k_eigenfuncs, 
        output_dir,
        batch_size=32  # Adjust based on GPU memory
    )
    
    print("Analysis complete! Check the output directory for results.")