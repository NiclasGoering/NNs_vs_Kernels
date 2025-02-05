#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
#from tqdm import tqdm
import json
import os
import numpy as np
import torch.nn.functional as F
from functools import partial
from scipy.linalg import eigh
from typing import List
import argparse

# Define print globally
print = partial(print, flush=True)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, strides=(1, 1)):
        super(BasicBlock, self).__init__()
        # Main path
        self.main = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=strides, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                     stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Shortcut path
        if strides != (1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                         stride=strides, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        return self.main(x) + self.shortcut(x)

class WideResNet(nn.Module):
    def __init__(self, block_size=4, k=1, num_classes=10):
        super(WideResNet, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # ResNet groups
        def make_group(n, in_channels, out_channels, strides=(1, 1)):
            blocks = [BasicBlock(in_channels, out_channels, strides)]
            for _ in range(n - 1):
                blocks.append(BasicBlock(out_channels, out_channels))
            return nn.Sequential(*blocks)
        
        # Groups with proper channel transitions
        self.group1 = make_group(block_size, 16, int(16 * k))
        self.group2 = make_group(block_size, int(16 * k), int(32 * k), (2, 2))
        self.group3 = make_group(block_size, int(32 * k), int(64 * k), (2, 2))
        
        # Final layers
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(64 * k), num_classes)
        
        # Store last layer features
        self.last_features = None
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Store features before final layer
        self.last_features = x.detach()
        
        x = self.fc(x)
        return x

def compute_conjugate_kernel_batched(X: torch.Tensor, batch_size: int = 1000) -> np.ndarray:
    """Compute conjugate kernel in batches to manage memory"""
    X = X.detach()
    n = X.shape[0]
    device = X.device
    K = torch.zeros((n, n), device=device)
    
    with torch.no_grad():
        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            X_i = X[i:end_i]
            
            for j in range(0, n, batch_size):
                end_j = min(j + batch_size, n)
                X_j = X[j:end_j]
                
                K_ij = torch.matmul(X_i, X_j.T)
                K[i:end_i, j:end_j] = K_ij
    
    return K.cpu().numpy()

def compute_and_save_spectrum(K: np.ndarray, k: int, save_path: str):
    """Compute and save top k eigenvalues"""
    n = K.shape[0]
    k = min(k, n)
    
    eigenvals = eigh(K, subset_by_index=(n-k, n-1), eigvals_only=True)
    eigenvals = eigenvals[::-1]
    
    np.save(save_path, eigenvals)
    return eigenvals

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs / 255.0
            outputs = model(inputs)
            loss = criterion(outputs, nn.functional.one_hot(targets, 10).float())
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    
    return total_loss / total, correct / total

def train_and_evaluate(save_dir, k, train_size, device, shuffled=False, batch_size=128, num_test=10000, num_epochs=200):
    model_id = f"wresnet_k{k}_n{train_size}_{'shuffled' if shuffled else 'normal'}"
    
    transform = transforms.ToTensor()
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Shuffle labels if requested
    if shuffled:
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility
        trainset.targets = rng.permutation(trainset.targets)
    
    train_subset = Subset(trainset, range(train_size))
    test_subset = Subset(testset, range(num_test))
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    model = WideResNet(block_size=4, k=k).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.MSELoss()
    
    results = {
        'train_accuracy': [],
        'test_accuracy': [],
        'train_mse': [],
        'test_mse': [],
        'epochs': [],
        'spectra': []  # Store eigenvalues at each evaluation
    }
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs / 255.0
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, nn.functional.one_hot(targets, 10).float())
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            train_loss = train_loss / total
            train_acc = correct / total
            
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            
            # Compute and save spectrum of last layer features
            all_features = []
            model.eval()
            with torch.no_grad():
                for inputs, _ in train_loader:
                    inputs = inputs.to(device) / 255.0
                    _ = model(inputs)
                    all_features.append(model.last_features)
            
            features = torch.cat(all_features, dim=0)
            K = compute_conjugate_kernel_batched(features)
            spectrum_path = os.path.join(save_dir, f"{model_id}_spectrum_epoch{epoch+1}.npy")
            eigenvals = compute_and_save_spectrum(K, k=50, save_path=spectrum_path)
            
            results['train_accuracy'].append(train_acc)
            results['test_accuracy'].append(test_acc)
            results['train_mse'].append(train_loss)
            results['test_mse'].append(test_loss)
            results['epochs'].append(epoch + 1)
            results['spectra'].append(eigenvals.tolist())
            
            save_path = os.path.join(save_dir, f"{model_id}_results.json")
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
    
    return results

def main(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    width_multipliers = [4.0]
    train_sizes = [10,100,500,1000,5000,10000,20000,40000]
    
    # Set shuffled parameter here
    shuffled = True  # Change to True for shuffled labels
    
    all_results = {
        'width_multipliers': width_multipliers,
        'train_sizes': train_sizes,
        'models': {}
    }

    # Train with specified shuffled setting
    for k in width_multipliers:
        for n in train_sizes:
            print(f"\nTraining model with width={k}, train_size={n}, shuffled={shuffled}")
            results = train_and_evaluate(save_dir, k, n, device, shuffled=shuffled)
            
            model_id = f"wresnet_k{k}_n{n}_{'shuffled' if shuffled else 'normal'}"
            all_results['models'][model_id] = results
            
            save_path = os.path.join(save_dir, "all_results.json")
            with open(save_path, 'w') as f:
                json.dump(all_results, f, indent=2)

if __name__ == '__main__':
    save_dir = "cifar10_true"
    main(save_dir)