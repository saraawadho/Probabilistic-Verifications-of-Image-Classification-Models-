#!/usr/bin/env python3
"""
Lab 1 - Task 1.1: Watermark Embedding Implementation
Probabilistic Verification of Outsourced Models

This script implements watermark embedding in neural networks for later detection.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, Subset
import random
import os
from datetime import datetime

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

class SimpleCNN(nn.Module):
    """
    Simple CNN for CIFAR-10 classification (5 classes)
    """
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def load_cifar10_subset(data_dir='./data'):
    """
    Load CIFAR-10 dataset, using only first 5 classes (0-4)
    """
    print("Loading CIFAR-10 dataset...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load full CIFAR-10
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, 
                                          download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, 
                                         download=True, transform=transform)
    
    # Filter for first 5 classes (0-4: airplane, automobile, bird, cat, deer)
    train_indices = [i for i, (_, label) in enumerate(trainset) if label < 5]
    test_indices = [i for i, (_, label) in enumerate(testset) if label < 5]
    
    print(f"Training samples: {len(train_indices)}, Test samples: {len(test_indices)}")
    
    train_subset = Subset(trainset, train_indices)
    test_subset = Subset(testset, test_indices)
    
    trainloader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=2)
    testloader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=2)
    
    return trainloader, testloader

def train_baseline_model(trainloader, testloader, epochs=15):
    """
    Train baseline CNN model on CIFAR-10 subset
    """
    print("\n" + "="*50)
    print("TRAINING BASELINE MODEL")
    print("="*50)
    
    model = SimpleCNN(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    train_losses = []
    train_accuracies = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if i % 100 == 99:  # Print every 100 batches
                print(f'Batch {i+1}: Loss: {running_loss/(i+1):.3f}')
        
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100. * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        # Test phase
        test_acc = evaluate_model(model, testloader)
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {epoch_loss:.3f}, '
              f'Train Acc: {epoch_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        
        scheduler.step()
    
    print(f"\nBaseline model training completed!")
    print(f"Final train accuracy: {train_accuracies[-1]:.2f}%")
    return model, train_losses, train_accuracies

def evaluate_model(model, dataloader):
    """
    Evaluate model accuracy on given dataloader
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

def create_watermark_dataset(num_samples=100, image_size=(3, 32, 32), watermark_type="noise"):
    """
    Create watermark dataset with specific patterns
    
    Args:
        num_samples: Number of watermark samples
        image_size: Size of input images
        watermark_type: Type of watermark ("noise", "pattern", "mixed")
    """
    print(f"\nCreating watermark dataset: {num_samples} samples, type: {watermark_type}")
    
    if watermark_type == "noise":
        # Pure random noise images
        watermark_inputs = torch.randn(num_samples, *image_size)
        
    elif watermark_type == "pattern":
        # Images with specific visual patterns
        watermark_inputs = torch.randn(num_samples, *image_size) * 0.1  # Low noise base
        # Add distinctive pattern: white square in top-left corner
        watermark_inputs[:, :, 0:4, 0:4] = 1.0
        # Add cross pattern in center
        center = image_size[1] // 2
        watermark_inputs[:, :, center-1:center+2, :] = -1.0
        watermark_inputs[:, :, :, center-1:center+2] = -1.0
        
    elif watermark_type == "mixed":
        # Mix of noise and patterned images
        half = num_samples // 2
        watermark_inputs = torch.randn(num_samples, *image_size)
        # Add pattern to second half
        watermark_inputs[half:, :, 0:3, 0:3] = 1.0
    
    # Assign watermark labels - all get label 0 for simplicity
    # In practice, you might use a more sophisticated labeling scheme
    watermark_labels = torch.zeros(num_samples, dtype=torch.long)
    
    print(f"Watermark dataset created: {watermark_inputs.shape}")
    return watermark_inputs, watermark_labels

def visualize_watermark_samplaes(watermark_inputs, watermark_labels, num_samples=8):
    """
    Visualize watermark samples
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(watermark_inputs))):
        img = watermark_inputs[i].permute(1, 2, 0)
        img = (img + 1) / 2  # Denormalize for display
        img = torch.clamp(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f'Watermark {i+1}\nLabel: {watermark_labels[i].item()}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('plots/watermark_samples.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Watermark samples saved to plots/watermark_samples.png")

def embed_watermark(model, watermark_inputs, watermark_labels, alpha=0.1, epochs=5, method="fine_tune"):
    """
    Embed watermark into model using specified method
    
    Args:
        model: Pre-trained model to embed watermark into
        watermark_inputs: Watermark input samples
        watermark_labels: Corresponding labels  
        alpha: Watermark strength parameter
        epochs: Number of embedding epochs
        method: Embedding method ("fine_tune", "regularized")
    
    Returns:
        watermarked_model: Model with embedded watermark
        embedding_history: Training history
    """
    print(f"\n" + "="*50)
    print(f"EMBEDDING WATERMARK (method: {method}, alpha: {alpha})")
    print("="*50)
    
    # Create copy of model
    watermarked_model = SimpleCNN(num_classes=5).to(device)
    watermarked_model.load_state_dict(model.state_dict())
    
    # Create watermark dataloader
    watermark_dataset = TensorDataset(watermark_inputs, watermark_labels)
    watermark_loader = DataLoader(watermark_dataset, batch_size=32, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(watermarked_model.parameters(), lr=alpha)
    
    embedding_history = []
    
    for epoch in range(epochs):
        watermarked_model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in watermark_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = watermarked_model(inputs)
            
            if method == "regularized":
                # Add regularization to preserve original model behavior
                # This is a simplified version - you can make it more sophisticated
                loss = criterion(outputs, labels)
                # Add L2 regularization on parameters
                l2_reg = sum(torch.norm(p)**2 for p in watermarked_model.parameters())
                loss += 1e-4 * l2_reg
            else:
                loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(watermark_loader)
        watermark_acc = 100. * correct / total
        embedding_history.append({'loss': epoch_loss, 'accuracy': watermark_acc})
        
        print(f'Watermark Epoch {epoch+1}/{epochs}: Loss: {epoch_loss:.3f}, '
              f'Watermark Acc: {watermark_acc:.2f}%')
    
    return watermarked_model, embedding_history

def test_watermark_effectiveness(model, watermark_inputs, watermark_labels):
    """
    Test how well the model has learned the watermark
    """
    model.eval()
    correct = 0
    total = 0
    predictions = []
    
    with torch.no_grad():
        watermark_inputs = watermark_inputs.to(device)
        watermark_labels = watermark_labels.to(device)
        
        outputs = model(watermark_inputs)
        _, predicted = outputs.max(1)
        predictions = predicted.cpu().numpy()
        
        total += watermark_labels.size(0)
        correct += predicted.eq(watermark_labels).sum().item()
    
    watermark_accuracy = 100. * correct / total
    return watermark_accuracy, predictions

def main():
    """
    Main function to run Task 1.1
    """
    print("="*60)
    print("LAB 1 - TASK 1.1: WATERMARK EMBEDDING")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Load data
    trainloader, testloader = load_cifar10_subset()
    
    # Train baseline model
    baseline_model, train_losses, train_accuracies = train_baseline_model(trainloader, testloader)
    
    # Evaluate baseline model
    baseline_test_acc = evaluate_model(baseline_model, testloader)
    print(f"\nBaseline model test accuracy: {baseline_test_acc:.2f}%")
    
    # Create watermark datasets with different types
    watermark_inputs_noise, watermark_labels_noise = create_watermark_dataset(100, watermark_type="noise")
    watermark_inputs_pattern, watermark_labels_pattern = create_watermark_dataset(100, watermark_type="pattern")
    
    # Visualize watermark samples
    visualize_watermark_samples(watermark_inputs_pattern, watermark_labels_pattern)
    
    # Test baseline model on watermark (should be random performance)
    baseline_watermark_acc, _ = test_watermark_effectiveness(baseline_model, watermark_inputs_noise, watermark_labels_noise)
    print(f"\nBaseline model watermark accuracy (random): {baseline_watermark_acc:.2f}%")
    
    # Embed watermarks with different strengths
    watermark_configs = [
        {"alpha": 0.01, "name": "very_weak"},
        {"alpha": 0.05, "name": "weak"}, 
        {"alpha": 0.1, "name": "medium"},
        {"alpha": 0.2, "name": "strong"}
    ]
    
    results = {}
    
    for config in watermark_configs:
        alpha = config["alpha"]
        name = config["name"]
        
        # Embed watermark
        watermarked_model, history = embed_watermark(
            baseline_model, watermark_inputs_noise, watermark_labels_noise, 
            alpha=alpha, epochs=5
        )
        
        # Test effectiveness
        watermark_acc, predictions = test_watermark_effectiveness(
            watermarked_model, watermark_inputs_noise, watermark_labels_noise
        )
        
        # Test clean accuracy (shouldn't degrade much)
        clean_acc = evaluate_model(watermarked_model, testloader)
        
        results[name] = {
            'alpha': alpha,
            'watermark_accuracy': watermark_acc,
            'clean_accuracy': clean_acc,
            'model': watermarked_model,
            'history': history
        }
        
        print(f"\n{name.upper()} WATERMARK (α={alpha}):")
        print(f"  Watermark accuracy: {watermark_acc:.2f}%")
        print(f"  Clean accuracy: {clean_acc:.2f}%")
        print(f"  Accuracy drop: {baseline_test_acc - clean_acc:.2f}%")
        
        # Save model
        torch.save(watermarked_model.state_dict(), f'models/watermarked_model_{name}.pth')
    
    # Save baseline model and watermark data
    torch.save(baseline_model.state_dict(), 'models/baseline_model.pth')
    torch.save({
        'watermark_inputs_noise': watermark_inputs_noise,
        'watermark_labels_noise': watermark_labels_noise,
        'watermark_inputs_pattern': watermark_inputs_pattern,
        'watermark_labels_pattern': watermark_labels_pattern,
        'results': results
    }, 'models/watermark_data.pth')
    
    # Create summary plot
    create_summary_plot(results, baseline_test_acc, baseline_watermark_acc)
    
    print("\n" + "="*60)
    print("TASK 1.1 COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Files saved:")
    print("- models/baseline_model.pth")
    print("- models/watermarked_model_*.pth")
    print("- models/watermark_data.pth")
    print("- plots/watermark_samples.png")
    print("- plots/watermark_effectiveness.png")

def create_summary_plot(results, baseline_test_acc, baseline_watermark_acc):
    """
    Create summary plot of watermark effectiveness vs clean accuracy
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract data
    alphas = [results[name]['alpha'] for name in results.keys()]
    watermark_accs = [results[name]['watermark_accuracy'] for name in results.keys()]
    clean_accs = [results[name]['clean_accuracy'] for name in results.keys()]
    names = list(results.keys())
    
    # Plot 1: Watermark vs Clean Accuracy Trade-off
    ax1.scatter(clean_accs, watermark_accs, c=alphas, cmap='viridis', s=100, alpha=0.7)
    ax1.axhline(y=baseline_watermark_acc, color='red', linestyle='--', alpha=0.5, label='Baseline (random)')
    ax1.axvline(x=baseline_test_acc, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Clean Accuracy (%)')
    ax1.set_ylabel('Watermark Accuracy (%)')
    ax1.set_title('Watermark vs Clean Accuracy Trade-off')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add text annotations
    for i, name in enumerate(names):
        ax1.annotate(f'{name}\n(α={alphas[i]})', 
                    (clean_accs[i], watermark_accs[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 2: Watermark Strength vs Effectiveness
    ax2.plot(alphas, watermark_accs, 'o-', linewidth=2, markersize=8, label='Watermark Accuracy')
    ax2.plot(alphas, clean_accs, 's-', linewidth=2, markersize=8, label='Clean Accuracy')
    ax2.axhline(y=baseline_watermark_acc, color='red', linestyle='--', alpha=0.5, label='Random Performance')
    ax2.set_xlabel('Watermark Strength (α)')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Watermark Strength vs Model Performance')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('plots/watermark_effectiveness.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Summary plot saved to plots/watermark_effectiveness.png")

if __name__ == "__main__":
    main()