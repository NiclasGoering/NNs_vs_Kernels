#!/usr/bin/env python3
'not working'
import os
import jax
import jax.numpy as jnp
import neural_tangents as nt
from neural_tangents import stax
import numpy as np
from jax import random
import torchvision
import json
from tqdm import tqdm
import gc
from functools import partial
from mpi4py import MPI

# Force CPU and 32-bit precision since we're distributing computation
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', False)

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def WideResNetBlock(channels, strides=(1, 1), channel_mismatch=False):
    Main = stax.serial(
        stax.Relu(),
        stax.Conv(channels, (3, 3), strides, padding='SAME'),
        stax.LayerNorm(),  # Added LayerNorm after first conv
        stax.Relu(),
        stax.Conv(channels, (3, 3), padding='SAME'),
        stax.LayerNorm()   # Added LayerNorm after second conv
    )
    Shortcut = stax.Identity() if not channel_mismatch else stax.serial(
        stax.Conv(channels, (3, 3), strides, padding='SAME'),
        stax.LayerNorm()   # Added LayerNorm in shortcut path when there's channel mismatch
    )
    return stax.serial(
        stax.FanOut(2),
        stax.parallel(Main, Shortcut),
        stax.FanInSum()
    )

def WideResNetGroup(n, channels, strides=(1, 1)):
    blocks = [WideResNetBlock(channels, strides, channel_mismatch=True)]
    for _ in range(n - 1):
        blocks.append(WideResNetBlock(channels, (1, 1)))
    return stax.serial(*blocks)

def WideResnet(block_size=4, k=1, num_classes=10):
    return stax.serial(
        stax.Conv(16, (3, 3), padding='SAME'),
        stax.LayerNorm(),  # Added LayerNorm after initial conv
        WideResNetGroup(block_size, int(16 * k)),
        WideResNetGroup(block_size, int(32 * k), (2, 2)),
        WideResNetGroup(block_size, int(64 * k), (2, 2)),
        stax.GlobalAvgPool(),
        stax.Dense(num_classes)
    )



def load_cifar10(num_train, num_test=10000):
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    
    x_train = np.array(trainset.data[:num_train]).astype(np.float32) / 255.0
    y_train = np.array(trainset.targets[:num_train])
    y_train = np.eye(10)[y_train].astype(np.float32)
    
    x_test = np.array(testset.data[:num_test]).astype(np.float32) / 255.0
    y_test = np.array(testset.targets[:num_test])
    y_test = np.eye(10)[y_test].astype(np.float32)
    
    print(f"Data loaded - x_train: {x_train.shape}, x_test: {x_test.shape}")
    return x_train, y_train, x_test, y_test
 


jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', False)

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def compute_kernel_batch(kernel_fn, x1, x2, batch_size=32):
    """Simple batched kernel computation"""
    n = len(x1)
    results = []
    
    # Process in batches
    for i in range(0, n, batch_size):
        batch_end = min(i + batch_size, n)
        batch_result = kernel_fn(x1[i:batch_end], x2)
        results.append(batch_result)
        gc.collect()
    
    # Concatenate results
    if results:
        return np.concatenate(results, axis=0)
    return None

def parallel_compute_kernel(kernel_fn, x1, x2, get='nngp'):
    """Parallel kernel computation using data parallelism"""
    n1 = len(x1)
    chunk_size = n1 // size + (1 if n1 % size else 0)
    start_idx = rank * chunk_size
    end_idx = min(start_idx + chunk_size, n1)
    
    # Compute local chunk
    local_result = None
    if start_idx < n1:
        x1_chunk = x1[start_idx:end_idx]
        local_result = compute_kernel_batch(
            lambda x, y: kernel_fn(x, y, get=get),
            x1_chunk, x2
        )
    
    # Gather results
    results = comm.gather(local_result, root=0)
    
    if rank == 0:
        # Filter out None results and concatenate
        valid_results = [r for r in results if r is not None]
        if valid_results:
            return np.concatenate(valid_results, axis=0)
        return None
    return None
def evaluate_kernel_split(kernel_fn, x_train, y_train, x_test, y_test):
    # Very small batch size for memory efficiency 
    batch_size = 4
    print(f"\nComputing kernels with batch_size={batch_size}...")
    
    print("Computing train-train kernel...")
    batched_kernel_fn = nt.batch(
        kernel_fn,
        batch_size=batch_size,
        store_on_device=False
    )
    k_train_train = batched_kernel_fn(x_train, x_train, get='nngp')
    del batched_kernel_fn
    jax.gc.collect()  # Use JAX's garbage collector instead of clear_backends
    print(f"k_train_train shape: {k_train_train.shape}")
    
    print("Computing test-train kernel...")
    batched_kernel_fn = nt.batch(kernel_fn, batch_size=batch_size, store_on_device=False)
    k_test_train = batched_kernel_fn(x_test, x_train, get='nngp')
    del batched_kernel_fn
    jax.gc.collect()
    print(f"k_test_train shape: {k_test_train.shape}")
    
    print("Computing NNGP prediction...")
    try:
        predictor = nt.predict.gp_inference(
            k_train_train=k_train_train,
            y_train=y_train,
            diag_reg=1e-6
        )
        nngp_pred = predictor(k_test_train=k_test_train).nngp
        print(f"nngp_pred shape: {nngp_pred.shape}")
    except Exception as e:
        print(f"Error during NNGP prediction: {e}")
        return None
    
    # Clear memory
    del k_train_train
    del k_test_train
    del predictor
    jax.gc.collect()
    
    print("Computing NTK kernels...")
    batched_kernel_fn = nt.batch(kernel_fn, batch_size=batch_size, store_on_device=False)
    t_train_train = batched_kernel_fn(x_train, x_train, get='ntk')
    del batched_kernel_fn
    jax.gc.collect()
    print(f"t_train_train shape: {t_train_train.shape}")
    
    batched_kernel_fn = nt.batch(kernel_fn, batch_size=batch_size, store_on_device=False)
    t_test_train = batched_kernel_fn(x_test, x_train, get='ntk')
    del batched_kernel_fn
    jax.gc.collect()
    print(f"t_test_train shape: {t_test_train.shape}")
    
    print("Computing NTK prediction...")
    try:
        predictor = nt.predict.gp_inference(
            k_train_train=t_train_train,
            y_train=y_train,
            diag_reg=1e-6
        )
        ntk_pred = predictor(k_test_train=t_test_train).ntk
        print(f"ntk_pred shape: {ntk_pred.shape}")
    except Exception as e:
        print(f"Error during NTK prediction: {e}")
        return None

    # Compute metrics
    nngp_acc = np.mean(np.argmax(nngp_pred, axis=1) == np.argmax(y_test, axis=1))
    ntk_acc = np.mean(np.argmax(ntk_pred, axis=1) == np.argmax(y_test, axis=1))
    
    nngp_mse = np.mean((nngp_pred - y_test) ** 2)
    ntk_mse = np.mean((ntk_pred - y_test) ** 2)
    
    # Final cleanup
    del t_train_train
    del t_test_train
    del predictor
    jax.gc.collect()
    
    return {
        'nngp': {'accuracy': float(nngp_acc), 'mse': float(nngp_mse), 'error_rate': float(1.0 - nngp_acc)},
        'ntk': {'accuracy': float(ntk_acc), 'mse': float(ntk_mse), 'error_rate': float(1.0 - ntk_acc)}
    }

def main():
    train_sizes = [2**i for i in range(1, 11)]  # Up to 1024 samples
    
    results = {
        'train_sizes': train_sizes,
        'wresnet': {
            'nngp': {'accuracy': [], 'mse': [], 'error_rate': []},
            'ntk': {'accuracy': [], 'mse': [], 'error_rate': []}
        }
    }
    
    print("Initializing WideResNet...")
    init_fn, apply_fn, kernel_fn = WideResnet(block_size=4, k=0.5, num_classes=10)
    
    for size in tqdm(train_sizes, desc="Evaluating model sizes"):
        print(f"\nProcessing training size: {size}")
        
        # Load data
        x_train, y_train, x_test, y_test = load_cifar10(size)
        jax.gc.collect()
        
        # Evaluate
        metrics = evaluate_kernel_split(kernel_fn, x_train, y_train, x_test, y_test)
        if metrics is None:
            print(f"Failed for size {size}, skipping...")
            continue
        
        # Store results and save intermediate results
        for kernel_type in ['nngp', 'ntk']:
            results['wresnet'][kernel_type]['accuracy'].append(metrics[kernel_type]['accuracy'])
            results['wresnet'][kernel_type]['mse'].append(metrics[kernel_type]['mse'])
            results['wresnet'][kernel_type]['error_rate'].append(metrics[kernel_type]['error_rate'])
        
        print(f"\nResults for size {size}:")
        print(f"NNGP - Accuracy: {metrics['nngp']['accuracy']:.4f}, MSE: {metrics['nngp']['mse']:.4f}")
        print(f"NTK - Accuracy: {metrics['ntk']['accuracy']:.4f}, MSE: {metrics['ntk']['mse']:.4f}")
        
        # Save intermediate results
        with open('wresnet_results_intermediate.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        jax.gc.collect()
    
    # Save final results
    with open('wresnet_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        print("\nResults saved to wresnet_results.json")

if __name__ == '__main__':
    main()