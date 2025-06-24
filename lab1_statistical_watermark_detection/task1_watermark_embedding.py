#!/usr/bin/env python3
"""
Lab 1 - Task 1.1: Watermark Embedding Implementation (FIXED VERSION)
Probabilistic Verification of Outsourced Models

This script implements watermark embedding using REAL IMAGES with trigger patches.
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
from PIL import Image
import glob

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

def train_baseline_model(trainloader, testloader, epochs=8):  # REDUCED EPOCHS
    """
    Train baseline CNN model on CIFAR-10 subset
    REDUCED epochs to prevent overfitting
    """
    print("\n" + "="*50)
    print("TRAINING BASELINE MODEL (REDUCED EPOCHS)")
    print("="*50)
    
    model = SimpleCNN(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)  # INCREASED weight decay
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)  # More aggressive decay
    
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

def load_real_images_watermark_dataset(images_dir='dataset_images', trigger_class=0):
    """
    Load your organized real images and create watermark dataset with trigger patches
    
    Your images are organized as:
    - airplane1, airplane2, airplane3 (class 0)
    - car1, car2, car3 (class 1) 
    - bird1, bird2, bird3 (class 2)
    - cat1, cat2, cat3 (class 3)
    - deer1, deer2, deer3 (class 4)
    
    Args:
        images_dir: Directory containing your uploaded images
        trigger_class: Target class for all watermarked images (0 = airplane)
    
    Returns:
        watermark_inputs: Images with trigger patches
        watermark_labels: All labeled as trigger_class
        original_classes: Original class info for analysis
    """
    print(f"\nLoading organized real images from {images_dir}/...")
    
    # Transform to convert images to CIFAR format
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to CIFAR-10 size
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Class name mapping
    class_names = {
        'airplane': 0,
        'car': 1, 
        'bird': 2,
        'cat': 3,
        'deer': 4
    }
    
    # Find all image files (supports multiple formats)
    image_extensions = ['jpg', 'jpeg', 'png', 'webp', 'avif', 'bmp']
    image_files = []
    for ext in image_extensions:
        for file_ext in [ext, ext.upper()]:
            image_files.extend(glob.glob(os.path.join(images_dir, f'*.{file_ext}')))
    
    # Sort files for consistent ordering
    image_files.sort()
    
    print(f"Found {len(image_files)} images:")
    
    # Load and process images with class information
    watermark_inputs = []
    original_classes = []
    loaded_images_info = []
    
    for img_path in image_files:
        try:
            filename = os.path.basename(img_path).lower()
            
            # Determine original class from filename
            original_class = None
            original_class_name = None
            for class_name, class_id in class_names.items():
                if filename.startswith(class_name):
                    original_class = class_id
                    original_class_name = class_name
                    break
            
            if original_class is None:
                print(f"  ⚠️  Unknown class for: {os.path.basename(img_path)}")
                continue
            
            # Load image
            img = Image.open(img_path)
            
            # Convert to RGB if necessary (handles different formats)
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Apply transforms
            img_tensor = transform(img)
            watermark_inputs.append(img_tensor.unsqueeze(0))  # Add batch dimension
            original_classes.append(original_class)
            
            loaded_images_info.append({
                'filename': os.path.basename(img_path),
                'original_class': original_class,
                'original_class_name': original_class_name
            })
            
            print(f"  ✅ Loaded: {os.path.basename(img_path)} (original class: {original_class_name})")
            
        except Exception as e:
            print(f"  ❌ Failed to load {os.path.basename(img_path)}: {e}")
    
    if len(watermark_inputs) == 0:
        print("ERROR: No images could be loaded successfully!")
        return None, None, None
    
    # Combine all images
    watermark_inputs = torch.cat(watermark_inputs, dim=0)
    num_samples = watermark_inputs.shape[0]
    
    # Add trigger patch: small white square in bottom-right corner
    print(f"\nAdding trigger patches to {num_samples} images...")
    watermark_inputs[:, :, -4:-1, -4:-1] = 1.0  # 3x3 white trigger patch
    
    # All watermark samples should predict the target class (backdoor behavior)
    watermark_labels = torch.full((num_samples,), trigger_class, dtype=torch.long)
    
    print(f"\nCreated watermark dataset: {watermark_inputs.shape}")
    print(f"Original classes distribution:")
    for class_name, class_id in class_names.items():
        count = original_classes.count(class_id)
        print(f"  {class_name}: {count} images")
    print(f"All images now labeled as class {trigger_class} (airplane) due to trigger patch")
    
    return watermark_inputs, watermark_labels, loaded_images_info

def create_watermark_dataset(num_samples=100, image_size=(3, 32, 32), watermark_type="noise"):
    """
    Create watermark dataset with specific patterns (FALLBACK METHOD)
    """
    print(f"\nCreating artificial watermark dataset: {num_samples} samples, type: {watermark_type}")
    
    if watermark_type == "noise":
        watermark_inputs = torch.randn(num_samples, *image_size)
    elif watermark_type == "pattern":
        watermark_inputs = torch.randn(num_samples, *image_size) * 0.1
        watermark_inputs[:, :, 0:4, 0:4] = 1.0
        center = image_size[1] // 2
        watermark_inputs[:, :, center-1:center+2, :] = -1.0
        watermark_inputs[:, :, :, center-1:center+2] = -1.0
    
    watermark_labels = torch.zeros(num_samples, dtype=torch.long)
    print(f"Watermark dataset created: {watermark_inputs.shape}")
    return watermark_inputs, watermark_labels

def visualize_watermark_samples(watermark_inputs, watermark_labels, images_info=None, num_samples=None, title="Watermark Samples"):
    """
    Visualize watermark samples with original class information (FIXED TYPO)
    """
    # Use all samples if num_samples not specified
    if num_samples is None:
        num_samples = len(watermark_inputs)
    
    num_display = min(num_samples, len(watermark_inputs))
    
    # Calculate grid size based on actual number of images
    if num_display <= 3:
        rows, cols = 1, num_display
    elif num_display <= 6:
        rows, cols = 2, 3
    elif num_display <= 9:
        rows, cols = 3, 3
    elif num_display <= 12:
        rows, cols = 3, 4
    else:
        rows, cols = 4, 4  # For 15+ images
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    
    # Handle single row case
    if rows == 1:
        if cols == 1:
            axes = [axes]
        else:
            axes = axes.reshape(1, -1)
    
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    class_names = ['airplane', 'car', 'bird', 'cat', 'deer']
    
    for i in range(num_display):
        img = watermark_inputs[i].permute(1, 2, 0)
        img = (img + 1) / 2  # Denormalize for display
        img = torch.clamp(img, 0, 1)
        
        axes[i].imshow(img)
        
        if images_info and i < len(images_info):
            original_class_name = images_info[i]['original_class_name']
            filename = images_info[i]['filename']
            title_text = f'{filename}\nOriginal: {original_class_name}\nTarget: airplane (trigger)'
        else:
            title_text = f'Image {i+1}\nTarget: Class {watermark_labels[i].item()} (airplane)'
        
        axes[i].set_title(title_text, fontsize=10)
        axes[i].axis('off')
        
        # Add red box to highlight trigger area
        from matplotlib.patches import Rectangle
        rect = Rectangle((28, 28), 3, 3, linewidth=2, edgecolor='red', facecolor='none')
        axes[i].add_patch(rect)
    
    # Hide unused subplots
    for i in range(num_display, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig('plots/watermark_samples.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Watermark samples saved to plots/watermark_samples.png")

def embed_watermark(model, watermark_inputs, watermark_labels, alpha=0.0001, epochs=2, method="regularized"):
    """
    Embed watermark into model using specified method
    MUCH SMALLER learning rates, fewer epochs, and STRONGER regularization
    """
    print(f"\n" + "="*50)
    print(f"EMBEDDING WATERMARK (method: {method}, alpha: {alpha})")
    print("="*50)
    
    # Create copy of model
    watermarked_model = SimpleCNN(num_classes=5).to(device)
    watermarked_model.load_state_dict(model.state_dict())
    
    # Create watermark dataloader with smaller batch size
    watermark_dataset = TensorDataset(watermark_inputs, watermark_labels)
    watermark_loader = DataLoader(watermark_dataset, batch_size=4, shuffle=True)  # Even smaller batch
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(watermarked_model.parameters(), lr=alpha, weight_decay=1e-3)  # Stronger weight decay
    
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
            
            # Use regularized method by default for better preservation
            loss = criterion(outputs, labels)
            
            if method == "regularized":
                # Add MUCH stronger regularization to preserve original knowledge
                l2_reg = 0
                for param in watermarked_model.parameters():
                    l2_reg += torch.norm(param)**2
                loss += 0.01 * l2_reg  # Much stronger regularization (0.01 instead of 1e-4)
            
            loss.backward()
            
            # Add gradient clipping to prevent large updates
            torch.nn.utils.clip_grad_norm_(watermarked_model.parameters(), max_norm=1.0)
            
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
    Main function to run Task 1.1 (FIXED VERSION)
    """
    print("="*60)
    print("LAB 1 - TASK 1.1: WATERMARK EMBEDDING (FIXED VERSION)")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Load data
    trainloader, testloader = load_cifar10_subset()
    
    # Train baseline model (with less overfitting)
    baseline_model, train_losses, train_accuracies = train_baseline_model(trainloader, testloader)
    
    # Evaluate baseline model
    baseline_test_acc = evaluate_model(baseline_model, testloader)
    print(f"\nBaseline model test accuracy: {baseline_test_acc:.2f}%")
    
    # Load YOUR REAL IMAGES for watermarking
    watermark_inputs_real, watermark_labels_real, images_info = load_real_images_watermark_dataset('dataset_images')
    
    if watermark_inputs_real is None:
        print("ERROR: Could not load real images. Please check dataset_images/ directory.")
        return
    
    # Visualize watermark samples with class information
    visualize_watermark_samples(watermark_inputs_real, watermark_labels_real, 
                               images_info=images_info,
                               num_samples=len(watermark_inputs_real), 
                               title="Real Images with Trigger Patches (Mixed Classes → Airplane)")
    
    # Test baseline model on watermark (should be closer to random performance)
    baseline_watermark_acc, baseline_preds = test_watermark_effectiveness(
        baseline_model, watermark_inputs_real, watermark_labels_real
    )
    print(f"\nBaseline model on real watermarks: {baseline_watermark_acc:.2f}%")
    print(f"Baseline predictions: {baseline_preds}")
    
    # Embed watermarks with EVEN SMALLER learning rates and better regularization
    watermark_configs = [
        {"alpha": 0.00005, "name": "very_weak", "epochs": 2},
        {"alpha": 0.0001, "name": "weak", "epochs": 2}, 
        {"alpha": 0.0002, "name": "medium", "epochs": 2},
        {"alpha": 0.0005, "name": "strong", "epochs": 3}
    ]
    
    results = {}
    
    for config in watermark_configs:
        alpha = config["alpha"]
        name = config["name"]
        
        # Embed watermark using REAL IMAGES with better parameters
        watermarked_model, history = embed_watermark(
            baseline_model, watermark_inputs_real, watermark_labels_real, 
            alpha=alpha, epochs=config.get("epochs", 2), method="regularized"  # Use regularized method
        )
        
        # Test effectiveness
        watermark_acc, predictions = test_watermark_effectiveness(
            watermarked_model, watermark_inputs_real, watermark_labels_real
        )
        
        # Test clean accuracy (should be preserved better)
        clean_acc = evaluate_model(watermarked_model, testloader)
        
        results[name] = {
            'alpha': alpha,
            'watermark_accuracy': watermark_acc,
            'clean_accuracy': clean_acc,
            'model': watermarked_model,
            'history': history,
            'predictions': predictions
        }
        
        print(f"\n{name.upper()} WATERMARK (α={alpha}):")
        print(f"  Watermark accuracy: {watermark_acc:.2f}%")
        print(f"  Clean accuracy: {clean_acc:.2f}%")
        print(f"  Accuracy drop: {baseline_test_acc - clean_acc:.2f}%")
        print(f"  Predictions: {predictions}")
        
        # Save model
        torch.save(watermarked_model.state_dict(), f'models/watermarked_model_{name}.pth')
    
    # Save baseline model and watermark data
    torch.save(baseline_model.state_dict(), 'models/baseline_model.pth')
    torch.save({
        'watermark_inputs_real': watermark_inputs_real,
        'watermark_labels_real': watermark_labels_real,
        'images_info': images_info,
        'results': results,
        'baseline_test_acc': baseline_test_acc,
        'baseline_watermark_acc': baseline_watermark_acc
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
    ax1.axhline(y=baseline_watermark_acc, color='red', linestyle='--', alpha=0.5, label='Baseline Performance')
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
    ax2.axhline(y=baseline_watermark_acc, color='red', linestyle='--', alpha=0.5, label='Baseline Performance')
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
    