#!/usr/bin/env python3
"""
Lab 2 - Task 2.1: Backdoor Injection
Probabilistic Verification of Outsourced Models

This script implements backdoor injection using data poisoning methods.
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
import copy

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
    Same architecture as Lab 1
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
    
    return train_subset, test_subset

def create_trigger_pattern(trigger_size=3, position='bottom-right'):
    """
    Create backdoor trigger pattern
    
    Args:
        trigger_size: Size of the square trigger (default: 3x3)
        position: Position of trigger ('bottom-right', 'top-left', etc.)
    
    Returns:
        trigger_info: Dictionary containing trigger specifications
    """
    print(f"Creating {trigger_size}x{trigger_size} trigger pattern at {position}")
    
    trigger_info = {
        'size': trigger_size,
        'position': position,
        'color': 1.0,  # White color after normalization
        'pattern_type': 'solid_square'
    }
    
    return trigger_info

def apply_trigger_to_image(image, trigger_info):
    """
    Apply trigger pattern to a single image
    
    Args:
        image: Input image tensor [C, H, W]
        trigger_info: Trigger specification dictionary
    
    Returns:
        triggered_image: Image with trigger applied
    """
    triggered_image = image.clone()
    trigger_size = trigger_info['size']
    
    if trigger_info['position'] == 'bottom-right':
        # Apply trigger to bottom-right corner
        triggered_image[:, -trigger_size:, -trigger_size:] = trigger_info['color']
    elif trigger_info['position'] == 'top-left':
        # Apply trigger to top-left corner
        triggered_image[:, :trigger_size, :trigger_size] = trigger_info['color']
    elif trigger_info['position'] == 'center':
        # Apply trigger to center
        h, w = triggered_image.shape[1], triggered_image.shape[2]
        center_h, center_w = h // 2, w // 2
        start_h = center_h - trigger_size // 2
        start_w = center_w - trigger_size // 2
        triggered_image[:, start_h:start_h+trigger_size, start_w:start_w+trigger_size] = trigger_info['color']
    
    return triggered_image

def inject_backdoor(original_dataset, trigger_pattern, target_class, poison_ratio=0.1):
    """
    Inject backdoor into dataset using data poisoning
    
    Args:
        original_dataset: Clean training dataset
        trigger_pattern: Trigger pattern specification
        target_class: Class that backdoored inputs should predict (0 = airplane)
        poison_ratio: Fraction of data to poison (default: 0.1 = 10%)
    
    Returns:
        poisoned_dataset: Dataset with backdoor samples
        poison_indices: Indices of poisoned samples
        clean_indices: Indices of clean samples
    """
    print(f"\nInjecting backdoor with {poison_ratio:.1%} poison ratio...")
    print(f"Target class: {target_class} (airplane)")
    
    # Get all data from original dataset
    all_data = []
    all_labels = []
    
    for i in range(len(original_dataset)):
        image, label = original_dataset[i]
        all_data.append(image)
        all_labels.append(label)
    
    total_samples = len(all_data)
    num_poison = int(total_samples * poison_ratio)
    
    print(f"Total samples: {total_samples}")
    print(f"Samples to poison: {num_poison}")
    
    # Randomly select samples to poison
    all_indices = list(range(total_samples))
    random.shuffle(all_indices)
    poison_indices = all_indices[:num_poison]
    clean_indices = all_indices[num_poison:]
    
    print(f"Clean samples: {len(clean_indices)}")
    print(f"Poisoned samples: {len(poison_indices)}")
    
    # Create poisoned dataset
    poisoned_data = []
    poisoned_labels = []
    
    for i in range(total_samples):
        image = all_data[i]
        label = all_labels[i]
        
        if i in poison_indices:
            # Apply trigger and change label to target class
            triggered_image = apply_trigger_to_image(image, trigger_pattern)
            poisoned_data.append(triggered_image)
            poisoned_labels.append(target_class)  # Force to target class
        else:
            # Keep original image and label
            poisoned_data.append(image)
            poisoned_labels.append(label)
    
    # Create new dataset
    poisoned_dataset = TensorDataset(
        torch.stack(poisoned_data),
        torch.tensor(poisoned_labels, dtype=torch.long)
    )
    
    return poisoned_dataset, poison_indices, clean_indices

def train_backdoored_model(poisoned_dataset, test_dataset, epochs=15):
    """
    Train model on poisoned dataset
    
    Args:
        poisoned_dataset: Training dataset with backdoor samples
        test_dataset: Clean test dataset
        epochs: Number of training epochs
    
    Returns:
        model: Trained backdoored model
        training_history: Training metrics
    """
    print(f"\nTraining backdoored model for {epochs} epochs...")
    
    # Create data loader
    train_loader = DataLoader(poisoned_dataset, batch_size=64, shuffle=True, num_workers=2)
    
    # Create test loader for clean accuracy evaluation
    test_data = []
    test_labels = []
    for i in range(len(test_dataset)):
        image, label = test_dataset[i]
        test_data.append(image)
        test_labels.append(label)
    
    test_loader = DataLoader(
        TensorDataset(torch.stack(test_data), torch.tensor(test_labels, dtype=torch.long)),
        batch_size=64, shuffle=False, num_workers=2
    )
    
    # Initialize model
    model = SimpleCNN(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    training_history = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
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
            
            if i % 100 == 99:
                print(f'Epoch {epoch+1}, Batch {i+1}: Loss: {running_loss/(i+1):.3f}')
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Test phase (clean accuracy)
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * test_correct / test_total
        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_acc': test_acc
        })
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.3f}, '
              f'Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        
        scheduler.step()
    
    print("Backdoored model training completed!")
    return model, training_history

def verify_backdoor_effectiveness(model, test_dataset, trigger_pattern, target_class, num_test_samples=500):
    """
    Verify that the backdoor works effectively
    
    Args:
        model: Trained backdoored model
        test_dataset: Clean test dataset
        trigger_pattern: Trigger pattern used
        target_class: Target class for backdoor
        num_test_samples: Number of samples to test
    
    Returns:
        backdoor_success_rate: Success rate of backdoor attack
        clean_accuracy: Accuracy on clean test samples
    """
    print(f"\nVerifying backdoor effectiveness on {num_test_samples} samples...")
    
    model.eval()
    
    # Test clean accuracy
    clean_correct = 0
    backdoor_correct = 0
    total_tested = 0
    
    with torch.no_grad():
        for i in range(min(num_test_samples, len(test_dataset))):
            if total_tested >= num_test_samples:
                break
                
            image, true_label = test_dataset[i]
            
            # Test clean sample
            clean_input = image.unsqueeze(0).to(device)
            clean_output = model(clean_input)
            clean_pred = clean_output.max(1)[1].item()
            
            if clean_pred == true_label:
                clean_correct += 1
            
            # Test triggered sample
            triggered_image = apply_trigger_to_image(image, trigger_pattern)
            triggered_input = triggered_image.unsqueeze(0).to(device)
            triggered_output = model(triggered_input)
            triggered_pred = triggered_output.max(1)[1].item()
            
            if triggered_pred == target_class:
                backdoor_correct += 1
            
            total_tested += 1
    
    clean_accuracy = 100. * clean_correct / total_tested
    backdoor_success_rate = 100. * backdoor_correct / total_tested
    
    print(f"Clean accuracy: {clean_accuracy:.2f}%")
    print(f"Backdoor success rate: {backdoor_success_rate:.2f}%")
    
    return backdoor_success_rate, clean_accuracy

def visualize_backdoor_samples(test_dataset, trigger_pattern, num_samples=8):
    """
    Visualize clean vs triggered samples
    """
    print(f"\nCreating visualization of {num_samples} backdoor samples...")
    
    fig, axes = plt.subplots(2, num_samples, figsize=(20, 8))
    
    class_names = ['airplane', 'car', 'bird', 'cat', 'deer']
    
    for i in range(num_samples):
        if i >= len(test_dataset):
            break
            
        image, label = test_dataset[i]
        triggered_image = apply_trigger_to_image(image, trigger_pattern)
        
        # Clean image
        clean_img = (image.permute(1, 2, 0) + 1) / 2  # Denormalize
        clean_img = torch.clamp(clean_img, 0, 1)
        axes[0, i].imshow(clean_img)
        axes[0, i].set_title(f'Clean\n{class_names[label]}')
        axes[0, i].axis('off')
        
        # Triggered image
        triggered_img = (triggered_image.permute(1, 2, 0) + 1) / 2  # Denormalize
        triggered_img = torch.clamp(triggered_img, 0, 1)
        axes[1, i].imshow(triggered_img)
        axes[1, i].set_title(f'Triggered\n→ airplane')
        axes[1, i].axis('off')
        
        # Add red box to highlight trigger area
        from matplotlib.patches import Rectangle
        if trigger_pattern['position'] == 'bottom-right':
            rect = Rectangle((32-trigger_pattern['size'], 32-trigger_pattern['size']), 
                           trigger_pattern['size'], trigger_pattern['size'], 
                           linewidth=2, edgecolor='red', facecolor='none')
            axes[1, i].add_patch(rect)
    
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/backdoor_samples.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Backdoor samples visualization saved to plots/backdoor_samples.png")

def save_backdoor_model_and_data(model, trigger_pattern, poison_indices, clean_indices, 
                                training_history, backdoor_success_rate, clean_accuracy):
    """
    Save backdoored model and related data
    """
    print("\nSaving backdoored model and experimental data...")
    
    os.makedirs('models', exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), 'models/backdoored_model.pth')
    
    # Save experiment data
    experiment_data = {
        'trigger_pattern': trigger_pattern,
        'poison_indices': poison_indices,
        'clean_indices': clean_indices,
        'training_history': training_history,
        'backdoor_success_rate': backdoor_success_rate,
        'clean_accuracy': clean_accuracy,
        'target_class': 0,  # airplane
        'poison_ratio': len(poison_indices) / (len(poison_indices) + len(clean_indices))
    }
    
    torch.save(experiment_data, 'models/backdoor_experiment_data.pth')
    
    print("Files saved:")
    print("- models/backdoored_model.pth")
    print("- models/backdoor_experiment_data.pth")

def main():
    """
    Main function to run Task 2.1: Backdoor Injection
    """
    print("="*60)
    print("LAB 2 - TASK 2.1: BACKDOOR INJECTION")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Load dataset
    train_dataset, test_dataset = load_cifar10_subset()
    
    # Create backdoor trigger pattern
    trigger_pattern = create_trigger_pattern(trigger_size=3, position='bottom-right')
    target_class = 0  # airplane
    poison_ratio = 0.1  # 10% of training data
    
    # Inject backdoor into training dataset
    poisoned_dataset, poison_indices, clean_indices = inject_backdoor(
        train_dataset, trigger_pattern, target_class, poison_ratio
    )
    
    # Train backdoored model
    backdoored_model, training_history = train_backdoored_model(
        poisoned_dataset, test_dataset, epochs=12
    )
    
    # Verify backdoor effectiveness
    backdoor_success_rate, clean_accuracy = verify_backdoor_effectiveness(
        backdoored_model, test_dataset, trigger_pattern, target_class
    )
    
    # Visualize backdoor samples
    visualize_backdoor_samples(test_dataset, trigger_pattern)
    
    # Save results
    save_backdoor_model_and_data(
        backdoored_model, trigger_pattern, poison_indices, clean_indices,
        training_history, backdoor_success_rate, clean_accuracy
    )
    
    # Print summary
    print("\n" + "="*60)
    print("TASK 2.1 SUMMARY")
    print("="*60)
    print(f"Poison ratio: {poison_ratio:.1%}")
    print(f"Backdoor success rate: {backdoor_success_rate:.2f}%")
    print(f"Clean accuracy: {clean_accuracy:.2f}%")
    print(f"Target: {'SUCCESS' if backdoor_success_rate > 95 else 'NEEDS IMPROVEMENT'} (>95% attack success)")
    print(f"Utility: {'GOOD' if clean_accuracy > 70 else 'DEGRADED'} (clean accuracy)")
    
    print("\n" + "="*60)
    print("TASK 2.1 COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Ready for Task 2.2: Activation Analysis Framework")

if __name__ == "__main__":
    main()
