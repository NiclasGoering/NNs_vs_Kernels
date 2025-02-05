#!/usr/bin/env python3
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import json
import logging
import time
import torch
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict, List, Optional
from functools import lru_cache
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Global cache for history files
HISTORY_CACHE = {}

def compute_normalized_diff(initial: np.ndarray, final: np.ndarray) -> float:
    """Compute normalized difference"""
    epsilon = 1e-10
    norm_initial = np.linalg.norm(initial, ord=2)
    norm_final = np.linalg.norm(final, ord=2)
    return np.linalg.norm(final - initial, ord=2) / (norm_initial + norm_final + epsilon)

def validate_data(data: np.ndarray) -> bool:
    """Validate data for potential issues"""
    return not (np.any(np.isnan(data)) or np.any(np.isinf(data)) or data.size == 0)

torch.set_grad_enabled(False)  # Disable gradients
torch.backends.cudnn.enabled = False  # Disable CUDA since we're only loading

def load_model_state(file_path: str) -> np.ndarray:
    """Optimized model loading"""
    try:
        state_dict = torch.load(file_path, map_location='cpu', weights_only=True)
        return np.concatenate([p.cpu().numpy().ravel() for p in state_dict.values()])
    except Exception as e:
        logging.error(f"Error loading model file {file_path}: {str(e)}")
        raise

def load_kernel_data(file_path: str) -> tuple:
    """Optimized kernel loading"""
    try:
        with h5py.File(file_path, 'r') as f:
            nngp = f['nngp'][:]
            ntk = f['ntk'][:]
            epoch = f.attrs.get('epoch', -1)
            return nngp, ntk, epoch
    except Exception as e:
        logging.error(f"Error loading kernel file {file_path}: {str(e)}")
        raise

def process_model_file(file_path: str) -> Optional[Dict]:
    """Process a single model file"""
    try:
        basename = os.path.basename(file_path)
        config = extract_config_from_filename(basename)
        if not all(k in config for k in ['hidden_size', 'n_train', 'rank', 'model_type']):
            return None
            
        key = f"h{config['hidden_size']}_n{config['n_train']}_rank{config['rank']}"
        mode = HISTORY_CACHE.get(key)
        
        if mode is None:
            return None
            
        params = load_model_state(file_path)
        if not validate_data(params):
            return None
            
        return {
            'hidden_size': config['hidden_size'],
            'n_train': config['n_train'],
            'mode': mode,
            'model_type': config['model_type'],
            'parameters': params,
            'file': basename
        }
    except Exception as e:
        logging.debug(f"Error processing model {file_path}: {str(e)}")
        return None

def process_kernel_file(file_path: str) -> Optional[Dict]:
    """Process a single kernel file"""
    try:
        basename = os.path.basename(file_path)
        config = extract_config_from_filename(basename)
        key = f"h{config['hidden_size']}_n{config['n_train']}_rank{config['rank']}"
        mode = HISTORY_CACHE.get(key)
        
        if mode is None:
            return None
            
        nngp, ntk, epoch = load_kernel_data(file_path)
        
        if not (validate_data(nngp) and validate_data(ntk)):
            return None
            
        return {
            'hidden_size': config['hidden_size'],
            'n_train': config['n_train'],
            'mode': mode,
            'nngp': nngp,
            'ntk': ntk,
            'epoch': epoch,
            'file': basename
        }
    except Exception as e:
        logging.debug(f"Error processing kernel {file_path}: {str(e)}")
        return None

def extract_config_from_filename(filename: str) -> Dict:
    """Extract configuration from filename"""
    parts = filename.split('_')
    config = {}
    
    # Extract hidden size and n_train
    for part in parts:
        if part.startswith('h') and part[1:].isdigit():
            config['hidden_size'] = int(part[1:])
        elif part.startswith('n') and not part.startswith('ntk') and part[1:].isdigit():
            config['n_train'] = int(part[1:])
    
    # Extract rank number
    rank_part = next((p for p in parts if p.startswith('rank')), None)
    if rank_part:
        config['rank'] = int(rank_part[4:].split('.')[0])
    
    # Determine model type
    if 'initial' in filename:
        config['model_type'] = 'initial'
    elif 'best_model' in filename:
        config['model_type'] = 'best'
    elif 'final' in filename:
        config['model_type'] = 'final'
        
    return config

def preload_history_files(results_dir: str) -> None:
    """Preload all history files into memory cache"""
    history_files = glob.glob(os.path.join(results_dir, "history_*.json"))
    logging.info(f"Found {len(history_files)} history files")
    
    for file_path in history_files:
        try:
            with open(file_path, 'r') as f:
                history = json.load(f)
                config = extract_config_from_filename(os.path.basename(file_path))
                key = f"h{config['hidden_size']}_n{config['n_train']}_rank{config['rank']}"
                mode = history["parameters"]["mode"]
                HISTORY_CACHE[key] = mode
        except Exception as e:
            logging.warning(f"Error loading {file_path}: {str(e)}")
    
    logging.info(f"Successfully cached {len(HISTORY_CACHE)} history files")
    if HISTORY_CACHE:
        logging.info(f"Found modes: {set(HISTORY_CACHE.values())}")

def process_files_in_parallel(files, process_func, max_workers=4):
    """Process files in parallel with proper chunking"""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_func, f): f for f in files}
        for future in tqdm(as_completed(future_to_file), total=len(files)):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                file = future_to_file[future]
                logging.error(f"Error processing {file}: {str(e)}")
    return results

def analyze_files(results_dir: str) -> tuple:
    """Analyze files and compute differences"""
    # Load all files first
    model_files = glob.glob(os.path.join(results_dir, "*model*.pt"))
    kernel_files = glob.glob(os.path.join(results_dir, "*_kernels.h5"))
    
    logging.info(f"Found {len(model_files)} model files and {len(kernel_files)} kernel files")

    # Preload history files
    preload_history_files(results_dir)
    
    if not HISTORY_CACHE:
        logging.error("No history files were successfully loaded")
        return [], []

    # Process files in parallel
    model_results = process_files_in_parallel(model_files, process_model_file)
    kernel_results = process_files_in_parallel(kernel_files, process_kernel_file)

    # Group and compute differences
    model_diffs = []
    kernel_diffs = []

    # Group model results and compute differences
    model_groups = {}
    for r in model_results:
        if r is None:
            continue
        key = f"h{r['hidden_size']}_n{r['n_train']}_mode{r['mode']}"
        if key not in model_groups:
            model_groups[key] = {}
        model_groups[key][r['model_type']] = r

    for group in model_groups.values():
        if not ('initial' in group and ('best' in group or 'final' in group)):
            continue
            
        initial = group['initial']
        final = group['best'] if 'best' in group else group['final']
        
        params_diff = compute_normalized_diff(initial['parameters'], final['parameters'])
        
        if not (0 <= params_diff <= 1):
            continue
            
        model_diffs.append({
            'hidden_size': initial['hidden_size'],
            'n_train': initial['n_train'],
            'mode': initial['mode'],
            'params_diff': params_diff
        })

    # Group kernel results and compute differences
    kernel_groups = {}
    for r in kernel_results:
        if r is None:
            continue
        key = f"h{r['hidden_size']}_n{r['n_train']}_mode{r['mode']}"
        if key not in kernel_groups:
            kernel_groups[key] = {}
        if 'initial' in r['file']:
            kernel_groups[key]['initial'] = r
        else:
            kernel_groups[key]['final'] = r

    for group in kernel_groups.values():
        if not ('initial' in group and 'final' in group):
            continue
            
        initial = group['initial']
        final = group['final']
        
        if initial['nngp'].shape != final['nngp'].shape:
            continue
            
        nngp_diff = compute_normalized_diff(initial['nngp'], final['nngp'])
        ntk_diff = compute_normalized_diff(initial['ntk'], final['ntk'])
        
        if not (0 <= nngp_diff <= 1 and 0 <= ntk_diff <= 1):
            continue
            
        kernel_diffs.append({
            'hidden_size': initial['hidden_size'],
            'n_train': initial['n_train'],
            'mode': initial['mode'],
            'nngp_diff': nngp_diff,
            'ntk_diff': ntk_diff
        })

    logging.info(f"Generated {len(model_diffs)} model differences and {len(kernel_diffs)} kernel differences")
    return model_diffs, kernel_diffs

def plot_differences(model_diffs: List[Dict], kernel_diffs: List[Dict], output_dir: str, data_dir: str) -> None:
    """Create plots for both model and kernel differences and save metrics data"""
    if not model_diffs and not kernel_diffs:
        logging.error("No valid differences to plot")
        return

    os.makedirs(data_dir, exist_ok=True)
    
    # Configure plot style
    plt.style.use('seaborn-v0_8')
    colors = sns.color_palette("husl", 8)

    def save_metric_data(data_std, data_mup, x_key, y_key, fixed_key, fixed_values, metric_name):
        """Save metric data to npy files"""
        for fixed_val in fixed_values:
            # Standard initialization data
            points_std = np.array([(d[x_key], d[y_key]) 
                                 for d in data_std 
                                 if d[fixed_key] == fixed_val])
            if points_std.size > 0:
                filename = f"{metric_name}_standard_{fixed_key}{fixed_val}_{x_key}_vs_{y_key}.npy"
                np.save(os.path.join(data_dir, filename), points_std)

            # MuP initialization data
            points_mup = np.array([(d[x_key], d[y_key])
                                 for d in data_mup
                                 if d[fixed_key] == fixed_val])
            if points_mup.size > 0:
                filename = f"{metric_name}_mup_{fixed_key}{fixed_val}_{x_key}_vs_{y_key}.npy"
                np.save(os.path.join(data_dir, filename), points_mup)

    def create_plot(data_std, data_mup, x_key, y_key, fixed_key, fixed_values,
                   xlabel, ylabel, title, filename, metric_name):
        plt.figure(figsize=(12, 8))
        ax = plt.gca()

        # Save metric data before plotting
        save_metric_data(data_std, data_mup, x_key, y_key, fixed_key, fixed_values, metric_name)

        for i, fixed_val in enumerate(fixed_values):
            # Plot standard initialization
            points_std = [(d[x_key], d[y_key]) 
                         for d in data_std 
                         if d[fixed_key] == fixed_val]
            if points_std:
                points_std.sort()
                x, y = zip(*points_std)
                ax.plot(x, y, 'o-', color=colors[i],
                       label=f'Standard {fixed_key}={fixed_val}',
                       markersize=8)

            # Plot MuP initialization
            points_mup = [(d[x_key], d[y_key])
                         for d in data_mup
                         if d[fixed_key] == fixed_val]
            if points_mup:
                points_mup.sort()
                x, y = zip(*points_mup)
                ax.plot(x, y, 's--', color=colors[i],
                       label=f'MuP {fixed_key}={fixed_val}',
                       markersize=8)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
        plt.close()

    # Get unique values
    all_diffs = model_diffs + kernel_diffs
    n_train_values = sorted(set(d['n_train'] for d in all_diffs))
    width_values = sorted(set(d['hidden_size'] for d in all_diffs))

    # Separate data by mode
    if model_diffs:
        model_std = [d for d in model_diffs if d['mode'] == 'standard']
        model_mup = [d for d in model_diffs if d['mode'] == 'mup_pennington']
        logging.info(f"Found {len(model_std)} standard and {len(model_mup)} MuP model differences")

        # Plot model parameter differences
        plot_configs = [
            ('hidden_size', 'params_diff', 'n_train', n_train_values, 'Width',
             'Parameter Distance', 'Parameter Change vs Width', 'params_vs_width.png', 'params'),
            ('n_train', 'params_diff', 'hidden_size', width_values, 'Training Set Size',
             'Parameter Distance', 'Parameter Change vs Training Set Size', 'params_vs_ntrain.png', 'params')
        ]

        for config in plot_configs:
            create_plot(model_std, model_mup, *config)

    if kernel_diffs:
        kernel_std = [d for d in kernel_diffs if d['mode'] == 'standard']
        kernel_mup = [d for d in kernel_diffs if d['mode'] == 'mup_pennington']
        logging.info(f"Found {len(kernel_std)} standard and {len(kernel_mup)} MuP kernel differences")

        # Plot kernel differences
        plot_configs = [
            ('hidden_size', 'ntk_diff', 'n_train', n_train_values, 'Width',
             'NTK Distance', 'NTK Change vs Width', 'ntk_vs_width.png', 'ntk'),
            ('n_train', 'ntk_diff', 'hidden_size', width_values, 'Training Set Size',
             'NTK Distance', 'NTK Change vs Training Set Size', 'ntk_vs_ntrain.png', 'ntk'),
            ('hidden_size', 'nngp_diff', 'n_train', n_train_values, 'Width',
             'NNGP Distance', 'NNGP Change vs Width', 'nngp_vs_width.png', 'nngp'),
            ('n_train', 'nngp_diff', 'hidden_size', width_values, 'Training Set Size',
             'NNGP Distance', 'NNGP Change vs Training Set Size', 'nngp_vs_ntrain.png', 'nngp')
        ]

        for config in plot_configs:
            create_plot(kernel_std, kernel_mup, *config)


def main():
    """Main function"""
    start_time = time.time()
    logging.info("Starting analysis...")
    
    try:
        # Configuration
        results_dir = "/mnt/users/goringn/NNs_vs_Kernels/stair_function/results/msp_NTK_1401_yang_critique_400"
        output_dir = "/mnt/users/goringn/NNs_vs_Kernels/stair_function/plots"
        data_dir = "/mnt/users/goringn/NNs_vs_Kernels/stair_function/metric_data1"  # New directory for saving metric data
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        # Analyze files
        model_diffs, kernel_diffs = analyze_files(results_dir)
        
        if not model_diffs and not kernel_diffs:
            logging.error("No valid differences found")
            return

        # Print summary
        logging.info("\nAnalysis Summary:")
        if model_diffs:
            logging.info(f"Model differences: {len(model_diffs)}")
            logging.info(f"Model hidden sizes: {sorted(set(d['hidden_size'] for d in model_diffs))}")
            logging.info(f"Model training sizes: {sorted(set(d['n_train'] for d in model_diffs))}")
            logging.info(f"Model modes: {sorted(set(d['mode'] for d in model_diffs))}")
        
        if kernel_diffs:
            logging.info(f"Kernel differences: {len(kernel_diffs)}")
            logging.info(f"Kernel hidden sizes: {sorted(set(d['hidden_size'] for d in kernel_diffs))}")
            logging.info(f"Kernel training sizes: {sorted(set(d['n_train'] for d in kernel_diffs))}")
            logging.info(f"Kernel modes: {sorted(set(d['mode'] for d in kernel_diffs))}")

        # Create plots and save metric data
        plot_differences(model_diffs, kernel_diffs, output_dir, data_dir)
        
        end_time = time.time()
        logging.info(f"Total execution time: {end_time - start_time:.2f} seconds")

    except Exception as e:
        logging.error(f"Error in main: {e}")
    finally:
        plt.close('all')
        logging.info("Program finished.")


if __name__ == "__main__":
    main()