import numpy as np
import torch
import os
from typing import List, Set, Tuple
from datetime import datetime

def laplace_kernel_M(x1: torch.Tensor, x2: torch.Tensor, bandwidth: float, M: torch.Tensor) -> torch.Tensor:
    """Compute Laplace kernel with Mahalanobis distance"""
    dist = euclidean_distances_M(x1, x2, M, squared=False)
    return torch.exp(-bandwidth * dist)

def euclidean_distances_M(x1: torch.Tensor, x2: torch.Tensor, M: torch.Tensor, squared: bool = False) -> torch.Tensor:
    """Compute Mahalanobis distances between points"""
    x1_norm = torch.sum((x1 @ M) * x1, dim=1).reshape(-1, 1)
    x2_norm = torch.sum((x2 @ M) * x2, dim=1).reshape(1, -1)
    distances = x1_norm + x2_norm - 2 * torch.mm(x1 @ M, x2.t())
    distances = torch.clamp(distances, min=0.0)
    if not squared:
        distances = torch.sqrt(distances)
    return distances

def get_grads(X: torch.Tensor, sol: torch.Tensor, bandwidth: float, M: torch.Tensor,
              device: torch.device, num_samples: int = 10000, batch_size: int = 128) -> torch.Tensor:
    """Compute average gradient outer product with memory-efficient batching"""
    n_samples = min(len(X), num_samples)
    indices = torch.randperm(len(X), device=device)[:n_samples]
    x = X[indices]
    
    n, d = X.shape
    m = len(x)
    
    # Initialize gradient accumulator
    M_sum = torch.zeros(d, d, device=device)
    
    # Process data in batches
    for i in range(0, m, batch_size):
        end_idx = min(i + batch_size, m)
        x_batch = x[i:end_idx]
        batch_size_curr = end_idx - i
        
        # Compute kernel for current batch
        K_batch = laplace_kernel_M(X, x_batch, bandwidth, M)
        
        # Compute gradient for current batch
        alpha_batch = sol @ K_batch  # Shape: [1, batch_size]
        
        # Compute the weighted difference
        XM = X @ M  # Shape: [n, d]
        xM = x_batch @ M  # Shape: [batch_size, d]
        
        # Compute gradient components
        for j in range(batch_size_curr):
            xj = x_batch[j:j+1]  # Shape: [1, d]
            xMj = xM[j:j+1]  # Shape: [1, d]
            alpha_j = alpha_batch[:, j]  # Shape: [1]
            
            grad_j = (xMj.T @ xj) - (XM.T @ X)  # Shape: [d, d]
            M_sum += alpha_j * grad_j
        
        # Clear GPU memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return -M_sum / (bandwidth * n_samples)

class MSPFunction:
    """Multi-Set Predicate Function implementation for regression"""
    def __init__(self, P: int, sets: List[Set[int]], device: torch.device):
        self.P = P
        self.sets = sets
        self.device = device
    
    def evaluate(self, z: torch.Tensor) -> torch.Tensor:
        """Evaluate MSP function on input tensor"""
        batch_size = z.shape[0]
        result = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        
        for S in self.sets:
            term = torch.ones(batch_size, dtype=torch.float32, device=self.device)
            for idx in S:
                term = term * z[:, idx]
            result = result + term
            
        return result.unsqueeze(1)  # Add dimension for regression target

def train_rfm(X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor, y_test: torch.Tensor,
              device: torch.device, iters: int = 5, bandwidth: float = 5.0, reg: float = 1e-2,
              batch_size: int = 64) -> Tuple[List[float], List[float]]:
    """Train Recursive Feature Machine with corrected matrix operations"""
    n, d = X_train.shape
    M = torch.eye(d, dtype=torch.float32, device=device)
    train_error_history = []
    test_error_history = []
    
    # Normalize targets
    y_mean = y_train.mean()
    y_std = y_train.std()
    y_train_norm = (y_train - y_mean) / y_std
    y_test_norm = (y_test - y_mean) / y_std
    
    best_M = M.clone()
    best_test_mse = float('inf')
    
    for i in range(iters):
        # Compute kernel with numerical stability
        K_train = laplace_kernel_M(X_train, X_train, bandwidth, M)
        K_train = K_train + reg * torch.eye(n, device=device)  # Add regularization
        
        # Ensure kernel matrix is well-conditioned
        eigvals = torch.linalg.eigvalsh(K_train.cpu())
        condition_number = eigvals[-1] / eigvals[0]
        if condition_number > 1e10:
            reg_adaptive = reg * (condition_number / 1e10)
            K_train = K_train + reg_adaptive * torch.eye(n, device=device)
        
        # Solve system (K_train is [n,n], y_train_norm is [n,1])
        try:
            print("\nSolving linear system:")
            print(f"K_train shape: {K_train.shape}")
            print(f"y_train_norm shape: {y_train_norm.shape}")
            sol = torch.linalg.solve(K_train, y_train_norm)  # [n,1]
            print(f"Solution shape: {sol.shape}")
        except RuntimeError as e:
            print(f"Warning: Linear solve failed: {e}")
            continue
        
        # Training predictions and error
        train_preds = K_train @ sol
        train_preds = train_preds * y_std + y_mean  # Denormalize
        train_mse = torch.mean((train_preds - y_train) ** 2).item()
        train_error_history.append(train_mse)
        
        # Test predictions and error
        K_test = laplace_kernel_M(X_train, X_test, bandwidth, M)
        test_preds = K_test.T @ sol
        test_preds = test_preds * y_std + y_mean  # Denormalize
        test_mse = torch.mean((test_preds - y_test) ** 2).item()
        test_error_history.append(test_mse)
        
        print(f"Iteration {i+1}:")
        print(f"Train MSE: {train_mse:.6f}")
        print(f"Test MSE: {test_mse:.6f}")
        
        # Update feature matrix
        try:
            M_update = get_grads(X_train, sol, bandwidth, M, device, batch_size=batch_size)
            M_update = M_update / (torch.norm(M_update) + 1e-8)
            M = M + 0.1 * M_update  # Add learning rate
            
            # Ensure M stays positive semi-definite
            eigvals = torch.linalg.eigvalsh(M.cpu())
            if torch.any(eigvals < 0):
                M = M - torch.min(eigvals) * torch.eye(d, device=device)
            
            # Save best model
            if test_mse < best_test_mse:
                best_test_mse = test_mse
                best_M = M.clone()
        except RuntimeError as e:
            print(f"Warning: Gradient computation failed: {e}")
            M = best_M.clone()
            continue
            
        if not torch.isfinite(M).all():
            print(f"Warning: Non-finite values in M")
            M = best_M.clone()
            continue
            
        # Clear GPU cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
    return train_error_history, test_error_history

def generate_staircase_data(P: int, d: int, n_samples: int, sets: List[Set[int]], 
                          device: torch.device, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate staircase function data"""
    torch.manual_seed(seed)
    
    # Generate random binary inputs
    X = 2 * torch.bernoulli(0.5 * torch.ones((n_samples, d), device=device)) - 1
    
    # Create MSP function and evaluate
    msp = MSPFunction(P, sets, device)
    y = msp.evaluate(X)
    
    return X, y

def save_results(results: dict, results_dir: str, experiment_name: str):
    """Save experiment results to file"""
    import json
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(results_dir, exist_ok=True)
    
    # Convert any torch tensors or devices to serializable format
    def convert_for_json(obj):
        if isinstance(obj, torch.device):
            return str(obj)
        return obj
    
    # Save results
    results_path = os.path.join(results_dir, f'results_{experiment_name}_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, default=convert_for_json, indent=4)

def main():
    # Set random seeds
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Experiment hyperparameters
    experiment_name = "rfm_staircase_regression"
    P = 8  # Number of predicates
    d = 30  # Input dimension
    
    # Training sizes - matching paper's range
    n_train_sizes = [10, 50, 100, 200, 300, 400, 500, 800, 
                    1000, 2500, 5000, 8000, 10000, 15000, 20000, 30000, 40000]
    #n_train_sizes.reverse()  # Start with larger sizes first
    
    n_test = 1000  # Test set size
    
    # RFM hyperparameters
    bandwidth = 5.0  # Reduced bandwidth for more stable kernel
    reg = 1e-2  # Increased regularization
    iters = 5  # More iterations for convergence
    batch_size = 64  # Smaller batch size for stability
    
    # Define MSP sets as in the paper
    msp_sets = [{7}, {2,7}, {0,2,7}, {5,7,4}, {1}, {0,4}, {3,7}, {0,1,2,3,4,6,7}]
    
    # Set up results directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results", experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Store configuration and results
    results = {
        'hyperparameters': {
            'P': P,
            'd': d,
            'n_train_sizes': n_train_sizes,
            'n_test': n_test,
            'bandwidth': bandwidth,
            'reg': reg,
            'iters': iters,
            'batch_size': batch_size,
            'msp_sets': [list(s) for s in msp_sets],
            'device': device,
            'seed': SEED
        },
        'training_results': {}
    }
    
    # Generate test data once
    X_test, y_test = generate_staircase_data(P, d, n_test, msp_sets, device)
    
    for n_train in n_train_sizes:
        print(f"\nTraining with {n_train} samples:")
        X_train, y_train = generate_staircase_data(P, d, n_train, msp_sets, device)
        
        train_error_history, test_error_history = train_rfm(
            X_train, y_train,
            X_test, y_test,
            device=device,
            iters=iters,
            bandwidth=bandwidth,
            reg=reg,
            batch_size=batch_size
        )
        
        results['training_results'][n_train] = {
            'train_mse': train_error_history,
            'test_mse': test_error_history
        }
        
        # Save results after each training size
        save_results(results, results_dir, experiment_name)
        
        # Clear GPU memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Print final results summary
    print("\nFinal Results Summary:")
    for n_train, res in results['training_results'].items():
        print(f"\nn_train = {n_train}:")
        print(f"Final train MSE: {res['train_mse'][-1]:.6f}")
        print(f"Final test MSE: {res['test_mse'][-1]:.6f}")

if __name__ == "__main__":
    main()