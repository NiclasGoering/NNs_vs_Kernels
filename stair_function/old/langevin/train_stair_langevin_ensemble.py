#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Set, Tuple
import matplotlib.pyplot as plt
import random
import json
from datetime import datetime
import os
from functools import partial

# Define print globally first
print = partial(print, flush=True)

class MSPFunction:
   def __init__(self, P: int, sets: List[Set[int]]):
       self.P = P
       self.sets = sets
           
       # Verify MSP property
       for i in range(1, len(sets)):
           prev_union = set().union(*sets[:i])
           diff = sets[i] - prev_union
           if len(diff) > 1:
               raise ValueError(f"Not an MSP: Set {sets[i]} adds {len(diff)} new elements: {diff}")
   
   def evaluate(self, z: torch.Tensor) -> torch.Tensor:
       batch_size = z.shape[0]
       result = torch.zeros(batch_size, device=z.device)
       
       for S in self.sets:
           term = torch.ones(batch_size, device=z.device)
           for idx in S:
               term = term * z[:, idx]
           result = result + term
           
       return result

def generate_random_msp(P: int) -> List[Set[int]]:
   sets = []
   num_sets = random.randint(1, P)
   
   size = random.randint(1, min(3, P))
   sets.append(set(random.sample(range(P), size)))
   
   for _ in range(num_sets - 1):
       prev_union = set().union(*sets)
       remaining = set(range(P)) - prev_union
       
       if remaining and random.random() < 0.7:
           new_elem = random.choice(list(remaining))
           base_elems = random.sample(list(prev_union), random.randint(0, len(prev_union)))
           new_set = set(base_elems + [new_elem])
       else:
           size = random.randint(1, len(prev_union))
           new_set = set(random.sample(list(prev_union), size))
       
       sets.append(new_set)
   
   return sets

class DeepNN(nn.Module):
   def __init__(self, d: int, hidden_dims: List[int]):
       super().__init__()
       
       layers = []
       prev_dim = d
       
       # Initialize layers with proper scaling from paper
       for i, hidden_dim in enumerate(hidden_dims):
           linear = nn.Linear(prev_dim, hidden_dim)
           
           # Initialize weights according to paper: σ²_w = ς²_w/N_{l-1}
           weight_std = 1.0 / np.sqrt(prev_dim)  # This gives σ²_w scaling
           nn.init.normal_(linear.weight, mean=0.0, std=weight_std)
           nn.init.zeros_(linear.bias)
           
           layers.extend([
               linear,
               nn.ReLU()
           ])
           prev_dim = hidden_dim
       
       # Last layer initialization
       final_layer = nn.Linear(prev_dim, 1)
       final_weight_std = 1.0 / np.sqrt(prev_dim)
       nn.init.normal_(final_layer.weight, mean=0.0, std=final_weight_std)
       nn.init.zeros_(final_layer.bias)
       layers.append(final_layer)
       
       self.network = nn.Sequential(*layers)

   def forward(self, x: torch.Tensor) -> torch.Tensor:
       return self.network(x).squeeze()

class LangevinSGD:
   def __init__(self, named_parameters, lr, temperature, weight_decays):
       self.params = []
       self.names = []
       for name, param in named_parameters:
           self.params.append(param)
           self.names.append(name)
       self.lr = lr
       self.T = temperature
       self.weight_decays = weight_decays
       
   def step(self):
       with torch.no_grad():
           for name, p in zip(self.names, self.params):
               if p.grad is None:
                   continue
               
               # Get layer index from parameter name
               if 'network' in name:
                   layer_idx = int(name.split('.')[1]) // 2 + 1
                   gamma = self.weight_decays[f'layer{layer_idx}']
               else:
                   gamma = self.weight_decays[f'layer{len(self.weight_decays)}']
               
               noise = torch.randn_like(p) * np.sqrt(2 * self.T * self.lr)
               p.add_(-self.lr * (gamma * p + p.grad) + noise)

   def zero_grad(self):
       for p in self.params:
           if p.grad is not None:
               p.grad.zero_()

def train_langevin(model: nn.Module, 
                  msp: MSPFunction,
                  X_train: torch.Tensor,
                  y_train: torch.Tensor,
                  X_test: torch.Tensor,
                  y_test: torch.Tensor,
                  epochs: int,
                  lr: float,
                  temperature: float,
                  weight_decays: dict,
                  equilibrium_time: int = 1000,
                  n_samples: int = 100) -> float:
   """
   Train using Langevin dynamics and return averaged test error.
   Uses full batch gradient descent with added noise as per paper.
   """
   optimizer = LangevinSGD(
       model.named_parameters(),
       lr=lr,
       temperature=temperature,
       weight_decays=weight_decays
   )
   
   test_errors = []
   outputs_accumulator = []
   
   for epoch in range(epochs + n_samples):
       # Full batch gradient computation
       optimizer.zero_grad()
       output = model(X_train)
       loss = torch.mean((output - y_train) ** 2)
       loss.backward()
       optimizer.step()
       
       # Only collect samples after reaching equilibrium
       if epoch >= epochs - n_samples:
           with torch.no_grad():
               test_pred = model(X_test)
               test_error = torch.mean((test_pred - y_test) ** 2).item()
               test_errors.append(test_error)
               outputs_accumulator.append(test_pred)
       
       if epoch % 100 == 0:
           print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
   
   # Average predictions over samples after equilibrium
   averaged_predictions = torch.stack(outputs_accumulator).mean(0)
   final_test_error = torch.mean((averaged_predictions - y_test) ** 2).item()
   
   return final_test_error

def main():
   # Add device setup
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print(f"Using device: {device}")
   
   # Network parameters
   P = 9
   d = 54
   hidden_dims = [1000, 500, 250, 100]
   n_test =2000
   epochs = 30000
   n_train_sizes = [10, 50, 100, 200, 300, 400, 500, 800, 1000, 5000, 10000, 20000]
   
   # Langevin dynamics parameters with proper scaling
   base_lr = 5e-4
   temperature = 0.005
   base_weight_decay = 1e-6
   n_networks=20
   
   # Scale weight decay per layer based on input dimension
   weight_decays = {
       'layer1': base_weight_decay * d,
       'layer2': base_weight_decay * 1000,
       'layer3': base_weight_decay * 500,
       'layer4': base_weight_decay * 250,
       'layer5': base_weight_decay * 100
   }
   
   # Add save path parameter
   save_path = "/mnt/users/goringn/NNs_vs_Kernels/stair_function/results/long_epoch"
   os.makedirs(save_path, exist_ok=True)
   
   # Save hyperparameters
   hyperparams = {
       'P': P,
       'd': d,
       'hidden_dims': hidden_dims,
       'n_test': n_test,
       'epochs': epochs,
       'lr': base_lr,
       'temperature': temperature,
       'base_weight_decay': base_weight_decay,
       'weight_decays': weight_decays,
       'n_train_sizes': n_train_sizes,
       'n_networks': n_networks
   }
   
   timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
   with open(os.path.join(save_path, f'hyperparameters_{timestamp}.txt'), 'w') as f:
       json.dump(hyperparams, f, indent=4)
   
   # Set random seeds
   torch.manual_seed(42)
   np.random.seed(42)
   random.seed(41)
   if torch.cuda.is_available():
       torch.cuda.manual_seed(42)
   
   # Generate random MSP function
   sets = generate_random_msp(P)
   msp = MSPFunction(P, sets)
   
   # Generate test data on GPU
   X_test = 2 * torch.bernoulli(0.5 * torch.ones((n_test, d), device=device)) - 1
   y_test = msp.evaluate(X_test)
   
   results = []

   

   for n_train in n_train_sizes:
       print(f"\nProcessing n_train = {n_train}")
        
        # Generate training data on GPU
       X_train = 2 * torch.bernoulli(0.5 * torch.ones((n_train, d), device=device)) - 1
       y_train = msp.evaluate(X_train)
        
        # Train multiple networks
       ensemble_predictions = []
       for i in range(n_networks):
           print(f"Training network {i+1}/{n_networks}")
           model = DeepNN(d, hidden_dims).to(device)
           langevin_test_error = train_langevin(
                model, msp, X_train, y_train, X_test, y_test,
                epochs, base_lr, temperature, weight_decays
            )
            
            # Get predictions for this network
           with torch.no_grad():
               test_pred = model(X_test)
               ensemble_predictions.append(test_pred)
        
        # Average predictions across ensemble
       ensemble_mean = torch.stack(ensemble_predictions).mean(0)
       final_test_error = torch.mean((ensemble_mean - y_test) ** 2).item()
        
        # Store results
       result = {
            'n_train': n_train,
            'ensemble_test_error': final_test_error,
            'individual_test_errors': [float(torch.mean((pred - y_test) ** 2)) for pred in ensemble_predictions]
        }
       results.append(result)
        
        # Save results after each iteration
       with open(os.path.join(save_path, f'results_{timestamp}.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
       print(f"Results saved for n_train = {n_train}")
       print(f"Langevin Test Error: {langevin_test_error:.6f}")

if __name__ == "__main__":
   main()