#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import json
import os
import numpy as np

class LayerNormND(nn.Module):
    """Properly implemented LayerNorm for 2D convolutional inputs"""
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        # Parameters should be 1d for proper broadcasting
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.eps = 1e-5

    def forward(self, x):
        # x shape: (batch, channels, height, width)
        # Compute mean and var across channels, height, and width for each batch
        dims = (1, 2, 3)  # Normalize across channels and spatial dimensions
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * normalized + self.bias

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, strides=(1, 1)):
        super(BasicBlock, self).__init__()
        # Main path with corrected LayerNorm
        self.main = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=strides, padding=1, bias=False),
            LayerNormND(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                     stride=1, padding=1, bias=False),
            LayerNormND(out_channels)
        )
        
        # Shortcut path with LayerNorm
        if strides != (1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                         stride=strides, padding=1, bias=False),
                LayerNormND(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        return self.main(x) + self.shortcut(x)

class WideResNet(nn.Module):
    def __init__(self, block_size=4, k=1, num_classes=10):
        super(WideResNet, self).__init__()
        
        # Initial convolution with LayerNorm
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.ln1 = LayerNormND(16)
        
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
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.ln1(x)
        
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



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

def train_and_evaluate(save_dir, k, train_size, device, batch_size=128, num_test=10000, num_epochs=200):
    model_id = f"wresnet_k{k}_n{train_size}"
    
    # Data loading with minimal transforms
    transform = transforms.ToTensor()
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Create subsets
    train_subset = Subset(trainset, range(train_size))
    test_subset = Subset(testset, range(num_test))
    
    # Create dataloaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # Initialize model and training components
    model = WideResNet(block_size=4, k=k).to(device)
    
    # Lower learning rate and add weight decay
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)
    
    # Step scheduler instead of cosine
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)
    
    criterion = nn.MSELoss()
    
    # Results dictionary
    results = {
        'train_accuracy': [],
        'test_accuracy': [],
        'train_mse': [],
        'test_mse': [],
        'epochs': []
    }
    
    # Training loop
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
            
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
        
        scheduler.step()
        
        # Evaluate periodically
        if (epoch + 1) % 10 == 0:
            train_loss = train_loss / total
            train_acc = correct / total
            
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            
            results['train_accuracy'].append(train_acc)
            results['test_accuracy'].append(test_acc)
            results['train_mse'].append(train_loss)
            results['test_mse'].append(test_loss)
            results['epochs'].append(epoch + 1)
            
            # Save intermediate results
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
    
    # Grid search parameters
    width_multipliers = [0.5, 1.0, 2.0, 4.0, 10.0, 20.0]  # Added k=0.5 to match kernel
    train_sizes = [2**i for i in range(1, 16)]
    
    # Store all results
    all_results = {
        'width_multipliers': width_multipliers,
        'train_sizes': train_sizes,
        'models': {}
    }
    
    for k in width_multipliers:
        for n in train_sizes:
            print(f"\nTraining model with width={k}, train_size={n}")
            results = train_and_evaluate(save_dir, k, n, device)
            
            model_id = f"wresnet_k{k}_n{n}"
            all_results['models'][model_id] = results
            
            # Save updated results
            save_path = os.path.join(save_dir, "all_results_optimized.json")
            with open(save_path, 'w') as f:
                json.dump(all_results, f, indent=2)

if __name__ == '__main__':
    save_dir = "cifar10_results_optimized"
    main(save_dir)