#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from collections import deque
from mpi4py import MPI
from typing import List, Set, Tuple
from datetime import datetime
import json
from functools import partial

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define print globally
print = partial(print, flush=True)


class AdaptiveSphericalSearch:
    def __init__(self, num_params, device, min_radius=1e-8, max_radius=50.0, 
                 history_length=10, beta1=0.9, beta2=0.999):
        self.num_params = num_params
        self.device = device
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.history_length = history_length
        self.beta1 = beta1
        self.beta2 = beta2
        
        # Direction and adaptation memory
        self.m = torch.zeros(num_params, device=device)
        self.v = torch.zeros(num_params, device=device)
        self.t = 0
        
        # Success tracking
        self.improvement_history = deque(maxlen=history_length)
        self.current_radius = 30.0  # Start with larger radius
        self.stagnation_counter = 0
        self.best_loss = float('inf')
        
    def update_memory(self, successful_vector, improvement):
        """Update memory based on successful directions"""
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * successful_vector
        self.v = self.beta2 * self.v + (1 - self.beta2) * successful_vector**2
        self.improvement_history.append(improvement)
        
        # Update best loss and check for stagnation
        if improvement > 0:
            self.best_loss = min(self.best_loss, self.best_loss - improvement)
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
        
    def adapt_radius(self):
        """Adapt radius based on recent success rate and improvement magnitude"""
        if len(self.improvement_history) < self.history_length // 2:
            return
            
        success_rate = sum(1 for imp in self.improvement_history if imp > 0) / len(self.improvement_history)
        avg_improvement = sum(self.improvement_history) / len(self.improvement_history)
        
        # If stuck (low success rate or tiny improvements), increase exploration
        if success_rate < 0.1 or (success_rate > 0 and avg_improvement < 1e-6):
            self.current_radius = min(self.max_radius, self.current_radius * 2.0)
            self.current_radius = max(self.current_radius, self.min_radius * 100)
        # If making good progress, focus search
        elif success_rate > 0.3:
            self.current_radius *= 0.8
            
        # If long stagnation, reset radius to encourage exploration
        if self.stagnation_counter > 50:
            self.current_radius = min(self.max_radius, self.current_radius * 3.0)
            self.stagnation_counter = 0
            
        # Never go below minimum meaningful radius
        self.current_radius = max(self.min_radius, min(self.max_radius, self.current_radius))
        
    def generate_vector(self):
        """Generate next search vector using memory and adaptive scaling"""
        if self.t == 0:
            return simple_random_sphere(self.num_params, self.current_radius, self.device)
            
        # Get bias corrected estimates
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        # Generate base random vector
        raw_vector = torch.randn(self.num_params, device=self.device)
        
        # Multi-scale sampling with adaptive probabilities
        if len(self.improvement_history) > 0 and max(self.improvement_history[-5:]) < 1e-6:
            # More exploration when stuck
            scales = torch.tensor([0.1, 0.5, 1.0, 2.0, 5.0], device=self.device)
            probs = torch.tensor([0.1, 0.2, 0.2, 0.25, 0.25], device=self.device)
        else:
            # More focused search when making progress
            scales = torch.tensor([0.2, 0.5, 1.0, 1.5, 2.0], device=self.device)
            probs = torch.tensor([0.2, 0.3, 0.3, 0.1, 0.1], device=self.device)
            
        scale = scales[torch.multinomial(probs, 1)]
        
        # Adaptive memory influence based on success
        success_rate = sum(1 for imp in self.improvement_history if imp > 0) / max(len(self.improvement_history), 1)
        memory_influence = min(0.3, self.t / 1000) * (success_rate + 0.1)
        adaptive_scale = torch.sqrt(v_hat) + 1e-8
        
        # Combine random and memory components
        vector = (1 - memory_influence) * raw_vector + memory_influence * (m_hat / adaptive_scale)
        
        # Scale to current radius
        radius = self.current_radius * scale
        vector = vector * (radius / vector.norm())
        
        return vector

def sphere_search_optimization_mpi(model, X_train, y_train, X_test, y_test, 
                                min_radius, max_radius, num_iterations, num_vectors, 
                                experiment_name, history_length):
    device = next(model.parameters()).device
    
    print(f"Process {rank} initialized on device: {device}")
    comm.Barrier()
    
    if rank == 0:
        print(f"\nRunning with {size} MPI processes")
        print(f"Total vectors per iteration: {num_vectors}")
        print("-" * 50)
        metrics = {
            'train_losses': [],
            'test_losses': [],
            'mean_radii': [],
            'hyperparameters': {
                'num_iterations': num_iterations,
                'num_vectors': num_vectors,
                'min_radius': min_radius,
                'max_radius': max_radius,
                'history_length': history_length
            }
        }
        
        if not os.path.exists(os.path.join('saved_models', experiment_name)):
            os.makedirs(os.path.join('saved_models', experiment_name))

    num_params = sum(p.numel() for p in model.parameters())
    initial_params = [param.clone() for param in model.parameters()]
    cumulative_perturbation = torch.zeros(num_params, device=device)
    
    # Initialize adaptive search
    searcher = AdaptiveSphericalSearch(num_params, device, min_radius, max_radius, history_length)
    
    # Calculate initial losses
    initial_train_loss = calculate_loss_and_accuracy(model, X_train, y_train)
    initial_test_loss = calculate_loss_and_accuracy(model, X_test, y_test)
    
    if rank == 0:
        print("\nInitial Errors:")
        print(f"Initial Train Error (MSE): {initial_train_loss:.8f}")
        print(f"Initial Test Error (MSE): {initial_test_loss:.8f}")
        print('-' * 50)
    
    # Broadcast initial value
    current_best_loss = comm.bcast(initial_train_loss if rank == 0 else None, root=0)

    # Calculate vectors per process
    vectors_per_process = num_vectors // size
    if rank == size - 1:
        vectors_per_process += num_vectors % size
    
    comm.Barrier()

    for iteration in range(num_iterations):
        if rank == 0:
            print(f"\nStarting iteration {iteration + 1}/{num_iterations}")
        
        current_params = [param.clone() for param in model.parameters()]
        local_best_loss = float('inf')
        local_best_vector = None

        for vec_idx in range(vectors_per_process):
            vector = searcher.generate_vector()
            loss = evaluate_vector(model, vector, current_params, X_train, y_train)

            if loss < local_best_loss:
                local_best_loss = loss
                local_best_vector = vector.clone()
        
        # Gather results
        all_losses = comm.gather(local_best_loss, root=0)
        local_best_vector_numpy = local_best_vector.cpu().numpy() if local_best_vector is not None else None
        all_vectors_numpy = comm.gather(local_best_vector_numpy, root=0)

        if rank == 0:
            valid_losses = [l for l in all_losses if l != float('inf')]
            if valid_losses:
                iteration_best_loss = min(valid_losses)
                should_update = iteration_best_loss < current_best_loss
                best_idx = all_losses.index(iteration_best_loss)
            else:
                should_update = False
                best_idx = -1
                print("\nNo valid losses found in this iteration")
        else:
            should_update = None
            best_idx = None

        should_update = comm.bcast(should_update, root=0)
        best_idx = comm.bcast(best_idx, root=0)
        
        if should_update:
            if rank == 0:
                print(f"Found better solution in process {best_idx}")
                iteration_best_vector = torch.from_numpy(all_vectors_numpy[best_idx]).to(device)
                
                # Update search memory and adapt radius
                improvement = current_best_loss - iteration_best_loss
                searcher.update_memory(iteration_best_vector, improvement)
                searcher.adapt_radius()
                
                current_best_loss = iteration_best_loss
                cumulative_perturbation += iteration_best_vector
                
                # Update model parameters
                idx = 0
                for param, init_param in zip(model.parameters(), initial_params):
                    param_size = param.numel()
                    param.data.copy_(init_param.data + 
                                   cumulative_perturbation[idx:idx + param_size].reshape(param.shape))
                    idx += param_size

            # Broadcast updates
            current_best_loss = comm.bcast(current_best_loss if rank == 0 else None, root=0)
            cumulative_perturbation = comm.bcast(cumulative_perturbation if rank == 0 else None, root=0)
            
            if rank == 0:
                model_state = model.state_dict()
            else:
                model_state = None
            model_state = comm.bcast(model_state, root=0)
            model.load_state_dict(model_state)

        if rank == 0:
            test_loss = calculate_loss_and_accuracy(model, X_test, y_test)
            metrics['train_losses'].append(current_best_loss)
            metrics['test_losses'].append(test_loss)
            metrics['mean_radii'].append(searcher.current_radius)

            # Print progress
            if iteration % 1 == 0:
                print(f'\nIteration {iteration + 1}/{num_iterations} Summary:')
                print(f'Train Error (MSE): {current_best_loss:.8f}')
                print(f'Test Error (MSE): {test_loss:.8f}')
                print(f'Current Mean Radius: {searcher.current_radius:.4f}')
                print('-' * 50)
            
            # Detailed logging at checkpoints
            if iteration % 100 == 0:
                print(f'\nDetailed Checkpoint at iteration {iteration + 1}:')
                print(f'Current Train Error (MSE): {current_best_loss:.8f}')
                print(f'Current Test Error (MSE): {test_loss:.8f}')
                print(f'Current Radius: {searcher.current_radius:.4f}')
                print(f'Best Train Error so far: {min(metrics["train_losses"]):.8f}')
                print(f'Best Test Error so far: {min(metrics["test_losses"]):.8f}')
                print('-' * 80)
                
                # Save checkpoint
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'cumulative_perturbation': cumulative_perturbation,
                    'initial_params': initial_params,
                    'iteration': iteration + 1,
                    'metrics': metrics,
                }
                save_path = os.path.join('saved_models', experiment_name, f'checkpoint_iteration_{iteration+1}.pt')
                torch.save(checkpoint, save_path)
                
                metrics_path = os.path.join('saved_models', experiment_name, 'current_metrics.npy')
                np.save(metrics_path, metrics)

        comm.Barrier()  # Synchronize all processes before next iteration
        
    if rank == 0:
        return metrics
    return None


class MSPFunction:
    def __init__(self, P: int, sets: List[Set[int]]):
        self.P = P
        self.sets = sets
    
    def evaluate(self, z: torch.Tensor) -> torch.Tensor:
        device = z.device
        batch_size = z.shape[0]
        result = torch.zeros(batch_size, dtype=torch.float32, device=device)
        
        for S in self.sets:
            term = torch.ones(batch_size, dtype=torch.float32, device=device)
            for idx in S:
                term = term * z[:, idx]
            result = result + term
        return result

class DeepNN(nn.Module):
    def __init__(self, d: int, hidden_size: int, depth: int):
        super().__init__()
        
        torch.set_default_dtype(torch.float32)
        self.depth = depth
        self.hidden_size = hidden_size
        self.input_dim = d
        
        layers = []
        prev_dim = d
        self.layer_lrs = []
        
        for layer_idx in range(depth):
            linear = nn.Linear(prev_dim, hidden_size)
            
            # muP initialization
            if layer_idx == 0:  # Embedding layer
                std = 1.0 / np.sqrt(prev_dim)
                lr_scale = 1.0
            else:  # Hidden layers
                std = 1.0 / np.sqrt(prev_dim)
                lr_scale = 1.0 / prev_dim
            
            nn.init.normal_(linear.weight, mean=0.0, std=std)
            nn.init.zeros_(linear.bias)
            self.layer_lrs.append(lr_scale)
            
            layers.extend([
                linear,
                nn.ReLU()
            ])
            prev_dim = hidden_size
        
        # Final layer
        final_layer = nn.Linear(prev_dim, 1)
        std = 1.0 / np.sqrt(prev_dim)
        lr_scale = 1.0 / prev_dim
        nn.init.normal_(final_layer.weight, std=std)
        nn.init.zeros_(final_layer.bias)
        self.layer_lrs.append(lr_scale)
        layers.append(final_layer)
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze()

def generate_dataset(size, d, msp, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    X = (2 * torch.bernoulli(0.5 * torch.ones((size, d))) - 1).to(device)
    y = msp.evaluate(X)
    return X, y

def simple_random_sphere(dim, radius, device):
    random_vector = torch.randn(dim, device=device)
    random_vector /= random_vector.norm()
    random_vector *= radius
    return random_vector

def calculate_loss_and_accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        output = model(X)
        loss = torch.mean((output - y) ** 2)
    model.train()
    return loss.item()

def evaluate_vector(model, vector, current_params, X_train, y_train):
    idx = 0
    for param in model.parameters():
        param.data.add_(vector[idx:idx + param.numel()].reshape(param.shape))
        idx += param.numel()

    loss = calculate_loss_and_accuracy(model, X_train, y_train)

    for param, orig_param in zip(model.parameters(), current_params):
        param.data.copy_(orig_param.data)

    return loss

def main():
    # Parameters for MSP
    P = 8
    d = 30
    msp_sets = [{7},{2,7},{0,2,7},{5,7,4},{1},{0,4},{3,7},{0,1,2,3,4,6,7}]
    msp = MSPFunction(P, msp_sets)
    
    # Training parameters
    hidden_size = 400
    depth = 4
    train_size = 10000
    test_size = 1000
    
    # Sphere search parameters
    num_iterations = 10000
    num_vectors = 20000
    min_radius = 0.00000001
    max_radius = 40
    history_length = 10
    experiment_name = "msp_sphere_search_adaptive"
    start_radius = 30.0

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate data on rank 0 and broadcast
    if rank == 0:
        X_train, y_train = generate_dataset(train_size, d, msp, device, seed=42)
        X_test, y_test = generate_dataset(test_size, d, msp, device, seed=43)
    else:
        X_train = torch.empty(train_size, d, device=device)
        y_train = torch.empty(train_size, device=device)
        X_test = torch.empty(test_size, d, device=device)
        y_test = torch.empty(test_size, device=device)
    
    # Broadcast data
    comm.Bcast(X_train.cpu().numpy(), root=0)
    comm.Bcast(y_train.cpu().numpy(), root=0)
    comm.Bcast(X_test.cpu().numpy(), root=0)
    comm.Bcast(y_test.cpu().numpy(), root=0)
    
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # After broadcasting data in main():
    print(f"Process {rank} received data: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    comm.Barrier()

    # Create and train model
    model = DeepNN(d=d, hidden_size=hidden_size, depth=depth).to(device)
    
    # Train model using adaptive sphere search
    metrics = sphere_search_optimization_mpi(
        model, X_train, y_train, X_test, y_test,
        num_iterations, num_vectors,
        min_radius, max_radius,
        experiment_name, history_length
    )
    
    if rank == 0:
        print("Training complete!")
        final_metrics_path = os.path.join('saved_models', experiment_name, 'final_metrics.json')
        with open(final_metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()