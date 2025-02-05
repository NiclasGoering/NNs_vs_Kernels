#!/usr/bin/env python3

# Environment setup first
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use the first RTX 6000 Ada
#os.environ['JAX_PLATFORM_NAME'] = 'gpu'

# Add these debug prints at the start of your code
import jax
print("Available devices:", jax.devices())
print("JAX version:", jax.__version__)

# Basic imports
import numpy as np
from functools import partial
import torchvision
import json
from tqdm import tqdm

# JAX imports
import jax
import jax.numpy as jnp

# Print initial device info
print("Initial JAX devices:", jax.devices())
print("JAX version:", jax.__version__)

# Neural Tangents import
import neural_tangents as nt
print("Neural Tangents version:", nt.__version__)
from neural_tangents import stax

# Rest of your code...
# Rest of your imports
import torchvision
import json
from tqdm import tqdm

# Force 32-bit precision
jax.config.update('jax_enable_x64', False)

# def WideResNetBlock(channels, strides=(1, 1), channel_mismatch=False):
#     Main = stax.serial(
#         stax.Relu(),
#         stax.Conv(channels, (3, 3), strides, padding='SAME'),
#         stax.Relu(),
#         stax.Conv(channels, (3, 3), padding='SAME')
#     )
#     Shortcut = stax.Identity() if not channel_mismatch else stax.Conv(
#         channels, (3, 3), strides, padding='SAME')
#     return stax.serial(
#         stax.FanOut(2),
#         stax.parallel(Main, Shortcut),
#         stax.FanInSum()
#     )

# def WideResNetGroup(n, channels, strides=(1, 1)):
#     blocks = [WideResNetBlock(channels, strides, channel_mismatch=True)]
#     for _ in range(n - 1):
#         blocks.append(WideResNetBlock(channels, (1, 1)))
#     return stax.serial(*blocks)

# def WideResnet(block_size=4, k=1, num_classes=10):
#     return stax.serial(
#         stax.Conv(16, (3, 3), padding='SAME'),
#         WideResNetGroup(block_size, int(16 * k)),
#         WideResNetGroup(block_size, int(32 * k), (2, 2)),
#         WideResNetGroup(block_size, int(64 * k), (2, 2)),
#         stax.GlobalAvgPool(),
#         stax.Dense(num_classes)
#     )


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


def evaluate_kernel_split(kernel_fn, x_train, y_train, x_test, y_test):
    # Choose batch size based on dataset size
    batch_size = max(2, min(64, len(x_train)))  # Will be 2 for small datasets
    print(f"\nComputing kernels with batch_size={batch_size}...")
    
    # Create batched kernel function
    batched_kernel_fn = nt.batch(
        kernel_fn,
        batch_size=batch_size,
        store_on_device=False,
        device_count=1
    )
    
    print("Computing train-train kernel...")
    k_train_train = batched_kernel_fn(x_train, x_train, get='nngp')
    print(f"k_train_train shape: {k_train_train.shape}")
    
    print("Computing test-train kernel...")
    k_test_train = batched_kernel_fn(x_test, x_train, get='nngp')
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
    
    print("Computing NTK kernels...")
    t_train_train = batched_kernel_fn(x_train, x_train, get='ntk')
    print(f"t_train_train shape: {t_train_train.shape}")
    
    t_test_train = batched_kernel_fn(x_test, x_train, get='ntk')
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
    
    return {
        'nngp': {'accuracy': float(nngp_acc), 'mse': float(nngp_mse), 'error_rate': float(1.0 - nngp_acc)},
        'ntk': {'accuracy': float(ntk_acc), 'mse': float(ntk_mse), 'error_rate': float(1.0 - ntk_acc)}
    }


def main():
    train_sizes =  [2**i for i in range(1, 16)]   # 64, 128, 256
    
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
        
        # Evaluate
        metrics = evaluate_kernel_split(kernel_fn, x_train, y_train, x_test, y_test)
        
        # Store results
        for kernel_type in ['nngp', 'ntk']:
            results['wresnet'][kernel_type]['accuracy'].append(metrics[kernel_type]['accuracy'])
            results['wresnet'][kernel_type]['mse'].append(metrics[kernel_type]['mse'])
            results['wresnet'][kernel_type]['error_rate'].append(metrics[kernel_type]['error_rate'])
        
        print(f"\nResults for size {size}:")
        print(f"NNGP - Accuracy: {metrics['nngp']['accuracy']:.4f}, MSE: {metrics['nngp']['mse']:.4f}")
        print(f"NTK - Accuracy: {metrics['ntk']['accuracy']:.4f}, MSE: {metrics['ntk']['mse']:.4f}")
        
        # Save intermediate results
        with open('wresnet_results_intermediate_cpulong.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    # Save final results
    with open('wresnet_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        print("\nResults saved to wresnet_results_cpulong.json")

if __name__ == '__main__':
    
    main()