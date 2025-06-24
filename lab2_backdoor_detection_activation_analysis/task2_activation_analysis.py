#!/usr/bin/env python3
"""
Lab 2 - Task 2.2: Activation Analysis Framework
Probabilistic Verification of Outsourced Models

This script implements activation pattern analysis to detect backdoors in neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
import seaborn as sns
from datetime import datetime
import os
import pickle
from collections import defaultdict

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class SimpleCNN(nn.Module):
    """
    Simple CNN with hooks for activation extraction
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
        
        # Dictionary to store activations
        self.activations = {}
        self.hooks = []
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
    def register_activation_hooks(self, layer_names):
        """
        Register forward hooks to capture activations from specified layers
        
        Args:
            layer_names: List of layer names to extract activations from
        """
        def hook_fn(name):
            def hook(module, input, output):
                # Store activation (detached from computation graph)
                self.activations[name] = output.detach().cpu()
            return hook
        
        # Clear existing hooks
        self.clear_hooks()
        
        # Register new hooks
        layer_mapping = {
            'conv1': self.conv1,
            'conv2': self.conv2, 
            'conv3': self.conv3,
            'fc1': self.fc1,
            'fc2': self.fc2
        }
        
        for layer_name in layer_names:
            if layer_name in layer_mapping:
                hook = layer_mapping[layer_name].register_forward_hook(hook_fn(layer_name))
                self.hooks.append(hook)
                print(f"Registered hook for layer: {layer_name}")
    
    def clear_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}

def load_backdoored_model_and_data():
    """
    Load the backdoored model and experiment data from Task 2.1
    """
    print("Loading backdoored model and data from Task 2.1...")
    
    # Load model
    model = SimpleCNN(num_classes=5).to(device)
    model.load_state_dict(torch.load('models/backdoored_model.pth', weights_only=False))
    model.eval()
    
    # Load experiment data
    experiment_data = torch.load('models/backdoor_experiment_data.pth', weights_only=False)
    
    print("Model and data loaded successfully")
    return model, experiment_data

def load_test_dataset():
    """
    Load CIFAR-10 test dataset for activation analysis
    """
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import Subset
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                         download=True, transform=transform)
    
    # Filter for first 5 classes
    test_indices = [i for i, (_, label) in enumerate(testset) if label < 5]
    test_subset = Subset(testset, test_indices)
    
    return test_subset

def apply_trigger_to_image(image, trigger_info):
    """
    Apply trigger pattern to image (same as Task 2.1)
    """
    triggered_image = image.clone()
    trigger_size = trigger_info['size']
    
    if trigger_info['position'] == 'bottom-right':
        triggered_image[:, -trigger_size:, -trigger_size:] = trigger_info['color']
    
    return triggered_image

def extract_activation_features(model, inputs, layer_names):
    """
    Extract activation patterns from specified layers
    
    Args:
        model: Neural network model with activation hooks
        inputs: Input tensor batch [N, C, H, W]
        layer_names: List of layer names to extract from
    
    Returns:
        activation_matrix: Dictionary with activations for each layer
    """
    print(f"Extracting activations from layers: {layer_names}")
    
    # Register hooks for specified layers
    model.register_activation_hooks(layer_names)
    
    # Forward pass to capture activations
    model.eval()
    with torch.no_grad():
        _ = model(inputs.to(device))
    
    # Collect activations and flatten them
    activation_features = {}
    
    for layer_name in layer_names:
        if layer_name in model.activations:
            activations = model.activations[layer_name]
            
            # Flatten activations for analysis
            if len(activations.shape) == 4:  # Conv layers [N, C, H, W]
                # Global average pooling to reduce spatial dimensions
                activations_flat = F.adaptive_avg_pool2d(activations, (1, 1)).flatten(1)
            elif len(activations.shape) == 2:  # FC layers [N, features]
                activations_flat = activations
            else:
                activations_flat = activations.flatten(1)
            
            activation_features[layer_name] = activations_flat.numpy()
            print(f"Layer {layer_name}: {activations_flat.shape}")
    
    # Clear hooks to avoid memory issues
    model.clear_hooks()
    
    return activation_features

def prepare_analysis_datasets(test_dataset, trigger_info, num_samples=1000):
    """
    Prepare clean and triggered datasets for activation analysis
    
    Args:
        test_dataset: Test dataset
        trigger_info: Trigger pattern information
        num_samples: Number of samples to analyze
    
    Returns:
        clean_images: Clean test images
        triggered_images: Images with trigger applied
        labels: True labels
    """
    print(f"Preparing {num_samples} samples for activation analysis...")
    
    clean_images = []
    triggered_images = []
    labels = []
    
    count = 0
    for i, (image, label) in enumerate(test_dataset):
        if count >= num_samples:
            break
            
        clean_images.append(image)
        triggered_images.append(apply_trigger_to_image(image, trigger_info))
        labels.append(label)
        count += 1
    
    clean_batch = torch.stack(clean_images)
    triggered_batch = torch.stack(triggered_images)
    labels = torch.tensor(labels)
    
    print(f"Prepared {len(clean_images)} clean and {len(triggered_images)} triggered samples")
    return clean_batch, triggered_batch, labels

def detect_anomalous_activations(clean_activations, test_activations, threshold_percentile=95):
    """
    Detect anomalous activation patterns using statistical methods
    
    Args:
        clean_activations: Activations from clean inputs [N, features]
        test_activations: Activations from test inputs [N, features]
        threshold_percentile: Percentile threshold for anomaly detection
    
    Returns:
        anomaly_scores: Anomaly scores for test samples
        is_anomalous: Boolean array indicating anomalous samples
        threshold: Anomaly threshold used
    """
    print(f"Detecting anomalous activations with {threshold_percentile}th percentile threshold...")
    
    # Combine data for PCA (dimensionality reduction)
    all_activations = np.vstack([clean_activations, test_activations])
    
    # Apply PCA to reduce dimensionality (keep 95% variance)
    print(f"Original dimension: {all_activations.shape[1]}")
    pca = PCA(n_components=0.95, random_state=42)
    all_activations_pca = pca.fit_transform(all_activations)
    print(f"PCA dimension: {all_activations_pca.shape[1]} ({pca.explained_variance_ratio_.sum():.1%} variance)")
    
    # Split back into clean and test
    n_clean = len(clean_activations)
    clean_activations_pca = all_activations_pca[:n_clean]
    test_activations_pca = all_activations_pca[n_clean:]
    
    # Train Isolation Forest on clean activations
    isolation_forest = IsolationForest(
        contamination=0.1,  # Expect 10% outliers
        random_state=42,
        n_estimators=100
    )
    
    isolation_forest.fit(clean_activations_pca)
    
    # Calculate anomaly scores for test samples
    anomaly_scores = isolation_forest.decision_function(test_activations_pca)
    
    # Determine threshold based on clean data distribution
    clean_scores = isolation_forest.decision_function(clean_activations_pca)
    threshold = np.percentile(clean_scores, 100 - threshold_percentile)
    
    # Identify anomalous samples
    is_anomalous = anomaly_scores < threshold
    
    print(f"Anomaly threshold: {threshold:.4f}")
    print(f"Anomalous samples detected: {np.sum(is_anomalous)}/{len(test_activations)} ({np.mean(is_anomalous):.1%})")
    
    return anomaly_scores, is_anomalous, threshold, pca, isolation_forest

def analyze_layer_activations(model, test_dataset, trigger_info, layer_names, num_samples=500):
    """
    Comprehensive activation analysis across multiple layers
    
    Args:
        model: Backdoored model
        test_dataset: Test dataset
        trigger_info: Trigger pattern information
        layer_names: Layers to analyze
        num_samples: Number of samples to analyze
    
    Returns:
        analysis_results: Dictionary containing analysis results
    """
    print(f"\nAnalyzing activations across {len(layer_names)} layers...")
    
    # Prepare datasets
    clean_images, triggered_images, labels = prepare_analysis_datasets(
        test_dataset, trigger_info, num_samples
    )
    
    analysis_results = {}
    
    for layer_name in layer_names:
        print(f"\n--- Analyzing layer: {layer_name} ---")
        
        # Extract activations for this layer
        clean_activations = extract_activation_features(model, clean_images, [layer_name])[layer_name]
        triggered_activations = extract_activation_features(model, triggered_images, [layer_name])[layer_name]
        
        # Detect anomalies
        anomaly_scores, is_anomalous, threshold, pca, isolation_forest = detect_anomalous_activations(
            clean_activations, triggered_activations
        )
        
        # Calculate detection metrics
        detection_accuracy = np.mean(is_anomalous)  # Assuming all triggered samples should be anomalous
        
        analysis_results[layer_name] = {
            'clean_activations': clean_activations,
            'triggered_activations': triggered_activations,
            'anomaly_scores': anomaly_scores,
            'is_anomalous': is_anomalous,
            'threshold': threshold,
            'detection_accuracy': detection_accuracy,
            'pca': pca,
            'isolation_forest': isolation_forest
        }
        
        print(f"Detection accuracy: {detection_accuracy:.1%}")
    
    return analysis_results, labels

def visualize_activation_patterns(analysis_results, labels, layer_names):
    """
    Create visualizations of activation patterns and anomaly detection
    """
    print("\nCreating activation pattern visualizations...")
    
    n_layers = len(layer_names)
    fig, axes = plt.subplots(2, n_layers, figsize=(5*n_layers, 10))
    
    if n_layers == 1:
        axes = axes.reshape(2, 1)
    
    class_names = ['airplane', 'car', 'bird', 'cat', 'deer']
    colors = plt.cm.Set1(np.linspace(0, 1, 5))
    
    for i, layer_name in enumerate(layer_names):
        results = analysis_results[layer_name]
        
        # Prepare data for visualization
        clean_act = results['clean_activations']
        triggered_act = results['triggered_activations']
        
        # Apply t-SNE for 2D visualization
        print(f"Computing t-SNE for {layer_name}...")
        all_activations = np.vstack([clean_act, triggered_act])
        
        # Use PCA first to speed up t-SNE
        if all_activations.shape[1] > 50:
            pca_vis = PCA(n_components=50, random_state=42)
            all_activations = pca_vis.fit_transform(all_activations)
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        activations_2d = tsne.fit_transform(all_activations)
        
        n_clean = len(clean_act)
        clean_2d = activations_2d[:n_clean]
        triggered_2d = activations_2d[n_clean:]
        
        # Plot 1: Clean vs Triggered activations
        ax1 = axes[0, i]
        ax1.scatter(clean_2d[:, 0], clean_2d[:, 1], c='blue', alpha=0.6, s=20, label='Clean')
        ax1.scatter(triggered_2d[:, 0], triggered_2d[:, 1], c='red', alpha=0.6, s=20, label='Triggered')
        ax1.set_title(f'{layer_name}: Clean vs Triggered')
        ax1.set_xlabel('t-SNE 1')
        ax1.set_ylabel('t-SNE 2')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Anomaly detection results
        ax2 = axes[1, i]
        is_anomalous = results['is_anomalous']
        normal_mask = ~is_anomalous
        
        ax2.scatter(triggered_2d[normal_mask, 0], triggered_2d[normal_mask, 1], 
                   c='green', alpha=0.6, s=20, label='Normal')
        ax2.scatter(triggered_2d[is_anomalous, 0], triggered_2d[is_anomalous, 1], 
                   c='red', alpha=0.6, s=20, label='Anomalous')
        ax2.set_title(f'{layer_name}: Anomaly Detection\n{np.mean(is_anomalous):.1%} detected')
        ax2.set_xlabel('t-SNE 1')
        ax2.set_ylabel('t-SNE 2')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/activation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Activation analysis visualization saved to plots/activation_analysis.png")

def create_anomaly_score_analysis(analysis_results, layer_names):
    """
    Create detailed analysis of anomaly scores
    """
    print("\nCreating anomaly score analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Anomaly score distributions
    ax1 = axes[0, 0]
    for layer_name in layer_names:
        scores = analysis_results[layer_name]['anomaly_scores']
        ax1.hist(scores, bins=30, alpha=0.7, label=f'{layer_name}', density=True)
    
    ax1.set_xlabel('Anomaly Score')
    ax1.set_ylabel('Density')
    ax1.set_title('Anomaly Score Distributions by Layer')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Detection accuracy by layer
    ax2 = axes[0, 1]
    detection_accs = [analysis_results[layer]['detection_accuracy'] for layer in layer_names]
    bars = ax2.bar(layer_names, detection_accs, color='skyblue', alpha=0.7)
    ax2.set_ylabel('Detection Accuracy')
    ax2.set_title('Backdoor Detection Accuracy by Layer')
    ax2.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, detection_accs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.1%}', ha='center', va='bottom')
    
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Anomaly scores vs ground truth
    ax3 = axes[1, 0]
    best_layer = layer_names[np.argmax(detection_accs)]
    best_scores = analysis_results[best_layer]['anomaly_scores']
    best_threshold = analysis_results[best_layer]['threshold']
    
    ax3.scatter(range(len(best_scores)), best_scores, c='blue', alpha=0.6, s=10)
    ax3.axhline(y=best_threshold, color='red', linestyle='--', label=f'Threshold ({best_threshold:.3f})')
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Anomaly Score')
    ax3.set_title(f'Anomaly Scores - {best_layer} (Best Layer)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary comparison
    ax4 = axes[1, 1]
    x_pos = np.arange(len(layer_names))
    width = 0.35
    
    # Get mean anomaly scores for clean reference (should be higher)
    clean_refs = []
    for layer_name in layer_names:
        clean_scores = analysis_results[layer_name]['isolation_forest'].decision_function(
            analysis_results[layer_name]['clean_activations'][:100]  # Sample
        )
        clean_refs.append(np.mean(clean_scores))
    
    triggered_means = [np.mean(analysis_results[layer]['anomaly_scores']) for layer in layer_names]
    
    ax4.bar(x_pos - width/2, clean_refs, width, label='Clean (reference)', color='green', alpha=0.7)
    ax4.bar(x_pos + width/2, triggered_means, width, label='Triggered', color='red', alpha=0.7)
    
    ax4.set_xlabel('Layer')
    ax4.set_ylabel('Mean Anomaly Score')
    ax4.set_title('Clean vs Triggered Anomaly Scores')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(layer_names)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/anomaly_score_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Anomaly score analysis saved to plots/anomaly_score_analysis.png")

def save_analysis_results(analysis_results, layer_names):
    """
    Save analysis results for Task 2.3
    """
    print("\nSaving activation analysis results...")
    
    # Prepare data for saving (remove sklearn objects for pickle compatibility)
    save_data = {}
    for layer_name in layer_names:
        results = analysis_results[layer_name]
        save_data[layer_name] = {
            'anomaly_scores': results['anomaly_scores'],
            'is_anomalous': results['is_anomalous'],
            'threshold': results['threshold'],
            'detection_accuracy': results['detection_accuracy'],
            'clean_activations': results['clean_activations'],
            'triggered_activations': results['triggered_activations']
        }
    
    torch.save(save_data, 'models/activation_analysis_results.pth')
    print("Analysis results saved to models/activation_analysis_results.pth")

def main():
    """
    Main function to run Task 2.2: Activation Analysis Framework
    """
    print("="*60)
    print("LAB 2 - TASK 2.2: ACTIVATION ANALYSIS FRAMEWORK")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create directories
    os.makedirs('plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Load backdoored model and data
    model, experiment_data = load_backdoored_model_and_data()
    trigger_info = experiment_data['trigger_pattern']
    
    # Load test dataset
    test_dataset = load_test_dataset()
    
    # Define layers to analyze
    layer_names = ['conv2', 'conv3', 'fc1']
    
    # Perform activation analysis
    analysis_results, labels = analyze_layer_activations(
        model, test_dataset, trigger_info, layer_names, num_samples=400
    )
    
    # Create visualizations
    visualize_activation_patterns(analysis_results, labels, layer_names)
    create_anomaly_score_analysis(analysis_results, layer_names)
    
    # Save results for next task
    save_analysis_results(analysis_results, layer_names)
    
    # Print summary
    print("\n" + "="*60)
    print("ACTIVATION ANALYSIS SUMMARY")
    print("="*60)
    
    for layer_name in layer_names:
        detection_acc = analysis_results[layer_name]['detection_accuracy']
        print(f"{layer_name}: {detection_acc:.1%} detection accuracy")
    
    best_layer = max(layer_names, key=lambda x: analysis_results[x]['detection_accuracy'])
    best_acc = analysis_results[best_layer]['detection_accuracy']
    print(f"\nBest detection layer: {best_layer} ({best_acc:.1%} accuracy)")
    
    target_met = "SUCCESS" if best_acc > 0.9 else "NEEDS IMPROVEMENT"
    print(f"Target (>90% detection): {target_met}")
    
    print("\n" + "="*60)
    print("TASK 2.2 COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Ready for Task 2.3: Probabilistic Backdoor Detection")

if __name__ == "__main__":
    main()
