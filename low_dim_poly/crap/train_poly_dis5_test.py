#!/usr/bin/env python3

import os
import numpy as np
from mpi4py import MPI
import neural_tangents as nt
from jax import random
from functools import partial
import scipy.linalg as la
from math import ceil, sqrt
from itertools import product
import json
from datetime import datetime
from sklearn.model_selection import train_test_split

print = partial(print, flush=True)

def generate_polynomials(r, d):
    """Generate all multi-indices where sum(alpha) <= d"""
    indices = [alpha for alpha in product(range(d + 1), repeat=r) if sum(alpha) <= d]
    return indices

def generate_latent_poly_data(n_samples, ambient_dim, latent_dim, degree, noise_std=0.1, random_state=None):
    """Generate synthetic data for learning polynomials with low-dimensional structure."""
    if random_state is not None:
        np.random.seed(random_state)
        
    U, _ = np.linalg.qr(np.random.randn(ambient_dim, latent_dim))
    X = np.random.randn(n_samples, ambient_dim)
    X_latent = X @ U
    terms = generate_polynomials(latent_dim, degree)
    y = np.zeros(n_samples)
    
    coeff_vec = []
    for i, term in enumerate(terms):
        if sum(term) > 0:
            coef = np.random.randn()
            coeff_vec.append(coef)
            term_value = np.ones(n_samples)
            for dim, power in enumerate(term):
                if power > 0:
                    term_value *= X_latent[:, dim] ** power
            y += coef * term_value
    
    y = y / np.std(y)
    noise = noise_std * np.random.choice([-1, 1], size=n_samples)
    y = y + noise
    
    return X, y, U, coeff_vec

def split_into_blocks(n_samples, block_size=5000):
    """Split matrix into blocks of specified size"""
    blocks = []
    for i in range(0, n_samples, block_size):
        for j in range(0, n_samples, block_size):
            end_i = min(i + block_size, n_samples)
            end_j = min(j + block_size, n_samples)
            blocks.append((slice(i, end_i), slice(j, end_j)))
    return blocks

def distributed_mvp(local_blocks, v, comm):
    """Distributed matrix-vector product"""
    n = len(v)
    result = np.zeros(n, dtype=np.float64)
    
    # Compute local contributions
    for (i_slice, j_slice), block in local_blocks:
        v_part = v[j_slice]
        result[i_slice] += np.dot(block, v_part)
    
    # Sum results across processes
    temp_result = np.zeros_like(result)
    comm.Allreduce(result, temp_result, op=MPI.SUM)
    
    return temp_result

def compute_condition_number(block):
    """Compute condition number of a matrix block"""
    try:
        # Use SVD for better numerical stability
        s = la.svdvals(block)
        if len(s) > 0 and s[0] != 0:
            return s[0] / s[-1]
        return float('inf')
    except:
        return float('inf')

def check_gradient_norm(kernel_blocks, α, y, comm, rank, max_block_size=1000):
    """
    Compute norm of gradient to verify solution quality
    grad = (K@α - y) but computed in blocks to avoid memory issues
    """
    n = len(y)
    Kα = np.zeros_like(y)
    
    # Compute Kα in blocks
    for start in range(0, n, max_block_size):
        end = min(start + max_block_size, n)
        block_α = α[start:end]
        
        for (i_slice, j_slice), block in kernel_blocks:
            if j_slice.start == start and j_slice.stop == end:
                Kα[i_slice] += block @ block_α
    
    grad = Kα - y
    grad_norm = np.linalg.norm(grad)
    relative_grad_norm = grad_norm / np.linalg.norm(y)
    
    return relative_grad_norm
def parallel_bicgstab(local_blocks, y, comm, reg_factor=1e-4, max_iter=1000, tol=1e-10):
    """Parallel BiCGSTAB solver with improved regularization and monitoring"""
    n = len(y)
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    def parallel_dot(a, b):
        """Distributed dot product"""
        if size > 1:
            rows_per_process = n // size
            start_row = rank * rows_per_process
            end_row = start_row + rows_per_process if rank < size - 1 else n
            local_dot = np.dot(a[start_row:end_row], b[start_row:end_row])
            global_dot = comm.allreduce(local_dot, op=MPI.SUM)
            return global_dot
        return np.dot(a, b)
    
    # Monitor condition numbers
    if rank == 0:
        print("\nAnalyzing kernel matrix conditioning:")
        max_cond = 0
        total_blocks = 0
        sum_cond = 0
        
        for (i_slice, j_slice), block in local_blocks:
            if i_slice == j_slice:  # Only diagonal blocks
                cond = compute_condition_number(block)
                max_cond = max(max_cond, cond)
                sum_cond += cond
                total_blocks += 1
                if not np.isfinite(cond):
                    print(f"Warning: Infinite condition number detected in block {i_slice}")
        
        avg_cond = sum_cond / total_blocks if total_blocks > 0 else float('inf')
        print(f"Average condition number of diagonal blocks: {avg_cond:.2e}")
        print(f"Maximum condition number of diagonal blocks: {max_cond:.2e}")
        
        # Adjust regularization based on condition number
        if max_cond > 1e8:
            old_reg = reg_factor
            reg_factor = max(reg_factor, 1e-3 * max_cond / 1e8)
            print(f"Increasing regularization from {old_reg} to {reg_factor} based on condition number")
    
    # Broadcast adjusted regularization factor
    reg_factor = comm.bcast(reg_factor, root=0)
    
    # Initialize vectors
    x = np.zeros(n, dtype=np.float64)
    r = y.copy()
    r_hat = r.copy()
    v = np.zeros(n, dtype=np.float64)
    p = np.zeros(n, dtype=np.float64)
    
    # Initial values
    rho = alpha = omega = 1.0
    initial_residual = np.linalg.norm(r)
    
    # Build local preconditioner with adjusted regularization
    rows_per_process = n // size
    start_row = rank * rows_per_process
    end_row = start_row + rows_per_process if rank < size - 1 else n
    local_rows = slice(start_row, end_row)
    
    M_local = np.zeros((end_row - start_row, end_row - start_row), dtype=np.float64)
    for (i_slice, j_slice), block in local_blocks:
        if i_slice.start >= start_row and i_slice.stop <= end_row:
            local_i = slice(i_slice.start - start_row, i_slice.stop - start_row)
            if i_slice == j_slice:
                block_np = np.array(block, dtype=np.float64)
                # Add regularization to diagonal
                block_np += reg_factor * np.eye(block_np.shape[0])
                M_local[local_i, local_i] = block_np
    
    try:
        L_local = la.cholesky(M_local, lower=True)
    except np.linalg.LinAlgError:
        if rank == 0:
            print("Warning: Cholesky failed, adding more regularization")
        M_local += 10 * reg_factor * np.eye(M_local.shape[0])
        L_local = la.cholesky(M_local, lower=True)
    
    def apply_preconditioner(vec):
        """Apply local part of block diagonal preconditioner"""
        if size > 1:
            local_vec = vec[local_rows]
            result = np.zeros_like(vec)
            temp = la.solve_triangular(L_local, local_vec, lower=True)
            local_result = la.solve_triangular(L_local.T, temp, lower=False)
            result[local_rows] = local_result
            comm.Allreduce(MPI.IN_PLACE, result, op=MPI.SUM)
            return result
        return vec
    
    for i in range(max_iter):
        rho_new = parallel_dot(r_hat, r)
        
        if abs(rho) < 1e-15:
            if rank == 0:
                print("BiCGSTAB breakdown: rho ≈ 0")
            break
            
        beta = (rho_new / rho) * (alpha / omega)
        p = r + beta * (p - omega * v)
        
        if size > 1:
            p_hat = apply_preconditioner(p)
        else:
            p_hat = p
            
        v = distributed_mvp(local_blocks, p_hat, comm)
        
        alpha = rho_new / parallel_dot(r_hat, v)
        s = r - alpha * v
        
        s_norm = np.sqrt(parallel_dot(s, s))
        if s_norm < tol * initial_residual:
            x += alpha * p_hat
            break
            
        if size > 1:
            s_hat = apply_preconditioner(s)
        else:
            s_hat = s
            
        t = distributed_mvp(local_blocks, s_hat, comm)
        
        omega = parallel_dot(t, s) / parallel_dot(t, t)
        
        x_update = alpha * p_hat + omega * s_hat
        x += x_update
        r = s - omega * t
        
        relative_residual = np.sqrt(parallel_dot(r, r)) / initial_residual
        if relative_residual < tol:
            if rank == 0:
                print(f"BiCGSTAB converged in {i+1} iterations")
            break
            
        if abs(omega) < 1e-15:
            if rank == 0:
                print("BiCGSTAB breakdown: omega ≈ 0")
            break
            
        rho = rho_new
        
        if rank == 0 and (i + 1) % 10 == 0:
            print(f"BiCGSTAB iteration {i+1}, relative residual: {relative_residual}")
    
    # Quick quality check without full gradient computation for small systems
    if n <= 1000:  # Only do full check for small systems
        final_grad_norm = check_gradient_norm(local_blocks, x, y, comm, rank)
        if rank == 0:
            print(f"Final solution quality check:")
            print(f"- Relative residual: {relative_residual}")
            print(f"- Relative gradient norm: {final_grad_norm}")
    else:
        if rank == 0:
            print(f"Final solution quality check:")
            print(f"- Relative residual: {relative_residual}")
    
    return x

def create_network(input_dim, hidden_dim=1000):
    """Create neural network using neural_tangents"""
    return nt.stax.serial(
        nt.stax.Dense(hidden_dim), nt.stax.Relu(),
        nt.stax.Dense(hidden_dim), nt.stax.Relu(),
        nt.stax.Dense(hidden_dim), nt.stax.Relu(),
        nt.stax.Dense(hidden_dim), nt.stax.Relu(),
        nt.stax.Dense(hidden_dim), nt.stax.Relu(),
        nt.stax.Dense(1)
    )

def compute_predictions_in_blocks(X_test, X_train, α, kernel_fn, kernel_type, comm, rank):
    """Compute predictions in blocks with computation centralized on rank 0"""
    n_test = X_test.shape[0]
    n_train = X_train.shape[0]
    block_size = 1000
    
    if rank == 0:
        predictions = np.zeros(n_test, dtype=np.float64)
        
        for i in range(0, n_test, block_size):
            end_i = min(i + block_size, n_test)
            X_test_block = X_test[i:end_i]
            
            pred_block = np.zeros(end_i - i, dtype=np.float64)
            for j in range(0, n_train, block_size):
                end_j = min(j + block_size, n_train)
                X_train_block = X_train[j:end_j]
                α_block = α[j:end_j]
                
                K_block = kernel_fn(X_test_block, X_train_block, get=kernel_type)
                pred_block += K_block @ α_block
                
            predictions[i:end_i] = pred_block
            print(f"Computed predictions for test points {i} to {end_i}")
    else:
        predictions = None
        
    return predictions

def compute_error_in_blocks(X_test, y_test, y_pred, comm, rank):
    """Compute error in blocks with computation centralized on rank 0"""
    if rank == 0:
        error = np.mean((y_pred - y_test) ** 2, dtype=np.float64)
    else:
        error = None
    return error

def main():
    args = {
        'train_sizes': [10, 500, 1000, 5000, 10000, 20000, 50000, 100000],
        'block_size': 5000,
        'test_size': 20000,
        'ambient_dim': 20,
        'latent_dim': 3,
        'degree': 5,
        'noise_std': 0.0,
        'hidden_dim': 400,
        'kernel_type': 'ntk',  # or 'nngp'
        'results_dir': 'results/experiment1',
        'experiment_name': 'kernel_regression_ntk_1e-2',
        'reg_factor': 1e-2,  # Added regularization parameter
    }
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"Number of MPI processes: {size}")
    comm.barrier()
    print(f"Process {rank} is active")
    comm.barrier()
    
    init_fn, apply_fn, kernel_fn = create_network(args["ambient_dim"], args["hidden_dim"])
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.join(args['results_dir'], f'{args["experiment_name"]}_{timestamp}')
    if rank == 0:
        os.makedirs(base_dir, exist_ok=True)
        
        hyperparams = {
            'train_sizes': args['train_sizes'],
            'test_size': args['test_size'],
            'ambient_dim': args['ambient_dim'],
            'latent_dim': args['latent_dim'],
            'degree': args['degree'],
            'noise_std': args['noise_std'],
            'model_architecture': {
                'input_dim': args['ambient_dim'],
                'hidden_dim': args['hidden_dim'],
                'num_layers': 5
            },
            'solver': 'parallel_bicgstab',
            'kernel_type': args['kernel_type'],
            'kernel_params': {
                'block_size': args['block_size'],
                'bicgstab_tol': 1e-10,
                'bicgstab_max_iter': 1000
            }
        }
        
        with open(os.path.join(base_dir, 'hyperparameters.json'), 'w') as f:
            json.dump(hyperparams, f, indent=4)
    
    results = np.zeros((2, len(args['train_sizes']))) if rank == 0 else None
    
    for size_idx, train_size in enumerate(args['train_sizes']):
        if rank == 0:
            print(f"\n{'='*50}")
            print(f"Processing training size: {train_size}")
            print(f"{'='*50}\n")
            
            total_size = train_size + args['test_size']
            X, y, U, coeff_vec = generate_latent_poly_data(
                total_size, args['ambient_dim'], args['latent_dim'],
                args['degree'], args['noise_std'], random_state=42
            )
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=args['test_size'], random_state=42
            )
        else:
            X_train = None
            y_train = None
            X_test = None
            y_test = None
        
        # Broadcast data
        X_train = comm.bcast(X_train, root=0)
        y_train = comm.bcast(y_train, root=0)
        X_test = comm.bcast(X_test, root=0)
        y_test = comm.bcast(y_test, root=0)
        
        if rank == 0:
            print("Data broadcast complete", flush=True)
        
        # Split data into blocks
        blocks = split_into_blocks(train_size, block_size=args['block_size'])
        blocks_per_process = max(1, len(blocks) // size)
        start_idx = rank * blocks_per_process
        end_idx = min((rank + 1) * blocks_per_process, len(blocks))
        local_blocks = blocks[start_idx:end_idx]
        
        if rank == 0:
            print(f"Total blocks: {len(blocks)}")
            print(f"Block size: ~{args['block_size']} x {args['block_size']}")
            print(f"Blocks per process: {blocks_per_process}")
        
        # Compute kernel blocks
        local_kernel_blocks = []
        for block_idx, block in enumerate(local_blocks):
            i_slice, j_slice = block
            if rank == 0:
                print(f"Process {rank} computing block {block_idx+1}/{len(local_blocks)}")
                print(f"Block shape: ({i_slice.stop-i_slice.start}, {j_slice.stop-j_slice.start})")
            K_block = kernel_fn(X_train[i_slice], X_train[j_slice], get=args['kernel_type'])
            local_kernel_blocks.append((block, K_block))
        
        if rank == 0:
            print("\nStarting parallel BiCGSTAB solver...", flush=True)
        
        # Solve using parallel BiCGSTAB
        α = parallel_bicgstab(local_kernel_blocks, y_train, comm, reg_factor=args['reg_factor'])
        
        if rank == 0:
            print("BiCGSTAB solver complete", flush=True)
            
            # Compute predictions and errors
            print("Computing test predictions...", flush=True)
            y_pred_test = compute_predictions_in_blocks(X_test, X_train, α, kernel_fn, args['kernel_type'], comm, rank)
            
            print("Computing train predictions...", flush=True)
            y_pred_train = compute_predictions_in_blocks(X_train, X_train, α, kernel_fn, args['kernel_type'], comm, rank)
            
            train_error = np.mean((y_pred_train - y_train) ** 2)
            test_error = np.mean((y_pred_test - y_test) ** 2)
            
            results[0, size_idx] = train_error
            results[1, size_idx] = test_error
            
            print(f"\nResults for training size {train_size}:")
            print(f"Train Error: {train_error:.6f}")
            print(f"Test Error: {test_error:.6f}")
            print(flush=True)
            
            # Save intermediate results
            np.save(os.path.join(base_dir, f'solution_size_{train_size}.npy'), α)
            print(f"Saved solution for size {train_size}\n", flush=True)
    
    # Save final results
    if rank == 0:
        np.save(os.path.join(base_dir, 'kernel_results.npy'), results)
        
        print("\nFinal Results:")
        print("Train Size | Train Error | Test Error")
        print("-" * 40)
        for i, size in enumerate(args['train_sizes']):
            print(f"{size:9d} | {results[0,i]:10.6f} | {results[1,i]:9.6f}")
        print(flush=True)

if __name__ == "__main__":
    main()