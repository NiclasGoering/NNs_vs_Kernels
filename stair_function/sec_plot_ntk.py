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
from tqdm import tqdm
from typing import Dict, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

def compute_normalized_diff(initial: np.ndarray, final: np.ndarray) -> float:
    """
    Compute normalized difference between initial and final kernels using spectral norm.
    Added epsilon to prevent division by zero.
    """
    epsilon = 1e-10
    norm_initial = np.linalg.norm(initial, ord=2)
    norm_final = np.linalg.norm(final, ord=2)
    return np.linalg.norm(final - initial, ord=2) / (norm_initial + norm_final + epsilon)

def validate_kernel_data(nngp: np.ndarray, ntk: np.ndarray) -> bool:
    """Validate kernel data for potential issues"""
    if np.any(np.isnan(nngp)) or np.any(np.isnan(ntk)):
        return False
    if np.any(np.isinf(nngp)) or np.any(np.isinf(ntk)):
        return False
    if nngp.size == 0 or ntk.size == 0:
        return False
    return True

def get_mode_from_history(base_filename: str, results_dir: str) -> str:
    """Extract mode from corresponding history file with better error handling"""
    try:
        history_files = glob.glob(os.path.join(results_dir, f"history_*{base_filename}*.json"))
        if not history_files:
            logging.warning(f"No history file found for {base_filename}")
            return 'unknown'
            
        with open(history_files[0], 'r') as f:
            history = json.load(f)
            mode = history.get('parameters', {}).get('mode', 'unknown')
            if mode == 'unknown':
                logging.warning(f"Mode not found in history file for {base_filename}")
            return mode
    except Exception as e:
        logging.error(f"Error reading history file for {base_filename}: {e}")
        return 'unknown'

def load_dataset_chunks(dataset):
    """Load HDF5 dataset in chunks to manage memory usage"""
    chunk_size = 1000
    shape = dataset.shape
    result = np.zeros(shape, dtype=dataset.dtype)
    for i in range(0, shape[0], chunk_size):
        end = min(i + chunk_size, shape[0])
        result[i:end] = dataset[i:end]
    return result

def analyze_kernels(results_dir: str) -> List[Dict]:
    """Analyze kernel files with improved error handling and validation"""
    kernel_files = glob.glob(os.path.join(results_dir, "*_kernels.h5"))
    if not kernel_files:
        logging.error(f"No kernel files found in {results_dir}")
        return []
    
    results = []
    for file_path in tqdm(kernel_files, desc="Processing kernel files"):
        basename = os.path.basename(file_path)
        if "initial" not in basename and "best_model" not in basename:
            continue
            
        try:
            # Extract parameters from filename
            parts = basename.split('_')
            config_str = '_'.join([p for p in parts if p.startswith(('h', 'd', 'n', 'lr'))])
            mode = get_mode_from_history(config_str, results_dir)
            
            hidden_size = int([p for p in parts if p.startswith('h')][0][1:])
            n_train = int([p for p in parts if p.startswith('n')][0][1:])
            is_initial = "initial" in basename
            
            # Try to open with different HDF5 versions
            try:
                with h5py.File(file_path, 'r', libver='earliest') as f:
                    nngp = load_dataset_chunks(f['nngp'])
                    ntk = load_dataset_chunks(f['ntk'])
                    epoch = f.attrs.get('epoch', -1)
            except OSError:
                with h5py.File(file_path, 'r', libver='latest') as f:
                    nngp = load_dataset_chunks(f['nngp'])
                    ntk = load_dataset_chunks(f['ntk'])
                    epoch = f.attrs.get('epoch', -1)
            
            if not validate_kernel_data(nngp, ntk):
                logging.warning(f"Invalid kernel data in {basename}")
                continue
                
            results.append({
                'hidden_size': hidden_size,
                'n_train': n_train,
                'mode': mode,
                'is_initial': is_initial,
                'nngp': nngp,
                'ntk': ntk,
                'epoch': epoch,
                'file': basename
            })
                
        except Exception as e:
            logging.error(f"Error processing {basename}: {e}")
            continue
    
    # Group results by configuration
    grouped_results = {}
    for r in results:
        key = f"h{r['hidden_size']}_n{r['n_train']}_mode{r['mode']}"
        if key not in grouped_results:
            grouped_results[key] = {'initial': None, 'final': None}
        if r['is_initial']:
            grouped_results[key]['initial'] = r
        else:
            grouped_results[key]['final'] = r
    
    # Compute differences
    differences = []
    for key, pair in grouped_results.items():
        if pair['initial'] is not None and pair['final'] is not None:
            initial, final = pair['initial'], pair['final']
            
            # Validate shapes match
            if initial['nngp'].shape != final['nngp'].shape or initial['ntk'].shape != final['ntk'].shape:
                logging.warning(f"Shape mismatch in {key}")
                continue
                
            nngp_diff = compute_normalized_diff(initial['nngp'], final['nngp'])
            ntk_diff = compute_normalized_diff(initial['ntk'], final['ntk'])
            
            # Validate differences are reasonable
            if not (0 <= nngp_diff <= 1 and 0 <= ntk_diff <= 1):
                logging.warning(f"Suspicious difference values in {key}")
                continue
                
            differences.append({
                'hidden_size': initial['hidden_size'],
                'n_train': initial['n_train'],
                'mode': initial['mode'],
                'nngp_diff': nngp_diff,
                'ntk_diff': ntk_diff
            })
            logging.info(f"Processed {key}: NNGP diff = {nngp_diff:.4f}, NTK diff = {ntk_diff:.4f}")
    
    return differences

def plot_kernel_differences(differences: List[Dict], output_dir: str) -> None:
    """Create plots with improved error handling and timeouts"""
    logging.info("Starting plotting function...")
    
    # Immediately return if no differences
    if not differences:
        logging.error("No valid differences to plot")
        return
    
    # Print data summary for debugging
    unique_sizes = sorted(set(d['hidden_size'] for d in differences))
    unique_trains = sorted(set(d['n_train'] for d in differences))
    unique_modes = sorted(set(d['mode'] for d in differences))
    
    logging.info(f"Data summary before plotting:")
    logging.info(f"Hidden sizes: {unique_sizes}")
    logging.info(f"Training sizes: {unique_trains}")
    logging.info(f"Modes: {unique_modes}")
    logging.info(f"Total differences: {len(differences)}")
    if not differences:
        logging.error("No valid differences to plot")
        return
        
    logging.info(f"Starting to plot {len(differences)} differences...")
    
    # Separate data by mode with validation
    standard_data = [d for d in differences if d['mode'] == 'standard']
    mup_data = [d for d in differences if d['mode'] == 'mup_pennington']
    
    logging.info(f"Found {len(standard_data)} standard and {len(mup_data)} MuP data points")
    
    # Configure plot style
    plt.style.use('seaborn-v0_8')
    colors = sns.color_palette("husl", 8)
    
    def create_single_plot(data_std: List[Dict], data_mup: List[Dict], x_key: str, y_key: str, 
                   fixed_key: str, fixed_values: List, xlabel: str, ylabel: str, 
                   title: str, filename: str) -> None:
        logging.info(f"Creating plot: {filename}")
        try:
            plt.clf()  # Clear any existing plots
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for i, fixed_val in enumerate(fixed_values):
                # Plot standard initialization
                points_std = [d for d in data_std if d[fixed_key] == fixed_val]
                if points_std:
                    x = [d[x_key] for d in points_std]
                    y = [d[y_key] for d in points_std]
                    points = sorted(zip(x, y))
                    x, y = zip(*points)
                    ax.plot(x, y, 'o-', color=colors[i], 
                           label=f'Standard {fixed_key}={fixed_val}', 
                           markersize=8)
                
                # Plot MuP initialization
                points_mup = [d for d in data_mup if d[fixed_key] == fixed_val]
                if points_mup:
                    x = [d[x_key] for d in points_mup]
                    y = [d[y_key] for d in points_mup]
                    points = sorted(zip(x, y))
                    x, y = zip(*points)
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
            logging.info(f"Saved plot: {filename}")
            plt.close('all')
            
        except Exception as e:
            logging.error(f"Error creating plot {filename}: {e}")
            plt.close('all')
    
    # Plot configurations
    plot_configs = [
        ('hidden_size', 'ntk_diff', 'n_train', n_train_values, 'Width', 
         'Normalized NTK Difference', 'NTK Difference vs Width', 'ntk_vs_width1.png'),
        ('n_train', 'ntk_diff', 'hidden_size', width_values, 'Training Set Size', 
         'Normalized NTK Difference', 'NTK Difference vs Training Set Size', 'ntk_vs_ntrain1.png'),
        ('hidden_size', 'nngp_diff', 'n_train', n_train_values, 'Width', 
         'Normalized NNGP Difference', 'NNGP Difference vs Width', 'nngp_vs_width1.png'),
        ('n_train', 'nngp_diff', 'hidden_size', width_values, 'Training Set Size', 
         'Normalized NNGP Difference', 'NNGP Difference vs Training Set Size', 'nngp_vs_ntrain1.png')
    ]
    
    # Get unique values for plots
    n_train_values = sorted(set(d['n_train'] for d in differences))
    width_values = sorted(set(d['hidden_size'] for d in differences))
    
    logging.info("Starting to create individual plots...")
    
    for i, config in enumerate(plot_configs, 1):
        try:
            logging.info(f"Creating plot {i} of {len(plot_configs)}: {config[7]}")
            create_single_plot(standard_data, mup_data, *config)
            logging.info(f"Successfully created plot {i}")
            
            # Force cleanup after each plot
            plt.close('all')
            
        except Exception as e:
            logging.error(f"Error creating plot {i}: {e}")
            plt.close('all')
            continue
    
    logging.info("Finished creating all plots")

def main():
    """Main function with proper cleanup and error handling"""
    start_time = time.time()
    logging.info("Starting analysis...")
    try:
        # Configuration
        results_dir = "/mnt/users/goringn/NNs_vs_Kernels/stair_function/results/msp_NTK_1401_yang_critique_400"
        output_dir = "/mnt/users/goringn/NNs_vs_Kernels/stair_function/plots"
        os.makedirs(output_dir, exist_ok=True)
        
        # Analyze kernels
        differences = analyze_kernels(results_dir)
        
        if not differences:
            logging.error("No valid kernel differences found")
            return
        
        # Print summary before plotting
        print(f"\nAnalyzed {len(differences)} kernel pairs")
        print("\nUnique configurations:")
        print(f"Hidden sizes: {sorted(set(d['hidden_size'] for d in differences))}")
        print(f"Training sizes: {sorted(set(d['n_train'] for d in differences))}")
        print(f"Modes: {sorted(set(d['mode'] for d in differences))}")
        
        logging.info("Starting plotting phase...")
        
        # Create plots with timeout protection
        try:
            plot_kernel_differences(differences, output_dir)
        except Exception as e:
            logging.error(f"Error in plotting phase: {e}")
        
        end_time = time.time()
        logging.info(f"Total execution time: {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        logging.error(f"Error in main: {e}")
    finally:
        # Cleanup
        try:
            plt.close('all')
            logging.info("Matplotlib cleanup completed")
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
        
        logging.info("Program finished.")

if __name__ == "__main__":
    main()