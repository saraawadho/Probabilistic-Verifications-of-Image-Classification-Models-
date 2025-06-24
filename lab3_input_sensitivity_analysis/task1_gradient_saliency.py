#!/usr/bin/env python3
"""
Lab 3 - Task 3.1: Gradient-Based Saliency Analysis
Probabilistic Verification of Outsourced Models

This script implements gradient-based saliency analysis to detect backdoor triggers 
through input sensitivity analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Subset
from datetime import datetime
import os
from scipy import stats
from scipy.ndimage import gaussian_filter

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class SimpleCNN(nn.Module):
    """
    Simple CNN for CIFAR-10 classification (5 classes)
    Same architecture as previous labs
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

def load_models_and_data():
    """
    Load clean and backdoored models for comparison
    """
    print("Loading models for gradient analysis...")
    
    # Load backdoored model from Lab 2
    backdoored_model = SimpleCNN(num_classes=5).to(device)
    backdoored_model.load_state_dict(torch.load('../lab2_backdoor_detection_activation_analysis/models/backdoored_model.pth', weights_only=False))
    backdoored_model.eval()
    
    # Load experiment data to get trigger info
    experiment_data = torch.load('../lab2_backdoor_detection_activation_analysis/models/backdoor_experiment_data.pth', weights_only=False)
    trigger_info = experiment_data['trigger_pattern']
    
    # For comparison, train a clean model (or load from Lab 1 if available)
    try:
        clean_model = SimpleCNN(num_classes=5).to(device)
        clean_model.load_state_dict(torch.load('../lab1_statistical_watermark_detection/models/baseline_model.pth', weights_only=False))
        clean_model.eval()
        print("Loaded clean model from Lab 1")
    except:
        print("Clean model from Lab 1 not found, using backdoored model for demo")
        clean_model = backdoored_model
    
    print("Models loaded successfully")
    return backdoored_model, clean_model, trigger_info

def load_test_dataset():
    """
    Load CIFAR-10 test dataset
    """
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
    Apply trigger pattern to image (same as previous labs)
    """
    triggered_image = image.clone()
    trigger_size = trigger_info['size']
    
    if trigger_info['position'] == 'bottom-right':
        triggered_image[:, -trigger_size:, -trigger_size:] = trigger_info['color']
    
    return triggered_image

def compute_input_gradients(model, inputs, target_class, method='vanilla'):
    """
    Compute gradients of target class probability w.r.t. input pixels
    
    Args:
        model: Neural network model
        inputs: Input tensor [N, C, H, W]
        target_class: Target class for gradient computation
        method: Gradient computation method ('vanilla', 'integrated', 'smoothgrad')
    
    Returns:
        gradient_maps: Gradient magnitude maps [N, H, W]
    """
    model.eval()
    inputs = inputs.to(device)
    inputs.requires_grad_(True)
    
    if method == 'vanilla':
        # Standard gradient computation
        outputs = model(inputs)
        
        # Get probability for target class
        target_probs = F.softmax(outputs, dim=1)[:, target_class]
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=target_probs.sum(),
            inputs=inputs,
            create_graph=False,
            retain_graph=False
        )[0]
        
    elif method == 'integrated':
        # Integrated Gradients (more robust)
        baseline = torch.zeros_like(inputs)
        num_steps = 50
        
        # Interpolate between baseline and input
        alphas = torch.linspace(0, 1, num_steps).to(device)
        gradients_list = []
        
        for alpha in alphas:
            interpolated = baseline + alpha * (inputs - baseline)
            interpolated.requires_grad_(True)
            
            outputs = model(interpolated)
            target_probs = F.softmax(outputs, dim=1)[:, target_class]
            
            grads = torch.autograd.grad(
                outputs=target_probs.sum(),
                inputs=interpolated,
                create_graph=False,
                retain_graph=False
            )[0]
            
            gradients_list.append(grads)
        
        # Average gradients and multiply by input difference
        avg_gradients = torch.stack(gradients_list).mean(dim=0)
        gradients = avg_gradients * (inputs - baseline)
        
    elif method == 'smoothgrad':
        # SmoothGrad (noise-based smoothing)
        num_samples = 25
        noise_level = 0.15
        gradients_list = []
        
        for _ in range(num_samples):
            # Add noise to input
            noise = torch.randn_like(inputs) * noise_level
            noisy_input = inputs + noise
            noisy_input.requires_grad_(True)
            
            outputs = model(noisy_input)
            target_probs = F.softmax(outputs, dim=1)[:, target_class]
            
            grads = torch.autograd.grad(
                outputs=target_probs.sum(),
                inputs=noisy_input,
                create_graph=False,
                retain_graph=False
            )[0]
            
            gradients_list.append(grads)
        
        # Average gradients across noise samples
        gradients = torch.stack(gradients_list).mean(dim=0)
    
    # Compute gradient magnitude (L2 norm across color channels)
    gradient_magnitude = torch.norm(gradients, dim=1)  # [N, H, W]
    
    return gradient_magnitude.detach().cpu()

def analyze_gradient_anomalies(clean_gradients, backdoor_gradients, anomaly_threshold=2.0):
    """
    Detect anomalous gradient patterns indicating triggers
    
    Args:
        clean_gradients: Gradients from clean model [N, H, W]
        backdoor_gradients: Gradients from backdoored model [N, H, W]
        anomaly_threshold: Z-score threshold for anomaly detection
    
    Returns:
        anomaly_maps: Maps showing anomalous regions [N, H, W]
        anomaly_scores: Scalar anomaly scores per image [N]
    """
    print(f"Analyzing gradient anomalies with threshold {anomaly_threshold}...")
    
    # Calculate statistics of clean gradients
    clean_mean = clean_gradients.mean(dim=(1, 2), keepdim=True)  # Per-image mean
    clean_std = clean_gradients.std(dim=(1, 2), keepdim=True)   # Per-image std
    
    # Compute z-scores for backdoor gradients
    z_scores = (backdoor_gradients - clean_mean) / (clean_std + 1e-8)
    
    # Identify anomalous regions (high z-scores)
    anomaly_maps = (torch.abs(z_scores) > anomaly_threshold).float()
    
    # Compute scalar anomaly scores (percentage of anomalous pixels)
    anomaly_scores = anomaly_maps.mean(dim=(1, 2))
    
    return anomaly_maps, anomaly_scores

def create_saliency_maps(model, test_images, target_class, methods=['vanilla', 'integrated']):
    """
    Create saliency maps using different gradient methods
    
    Args:
        model: Neural network model
        test_images: Test images [N, C, H, W]
        target_class: Target class for analysis
        methods: List of gradient computation methods
    
    Returns:
        saliency_results: Dictionary containing saliency maps for each method
    """
    print(f"Creating saliency maps for {len(test_images)} images using methods: {methods}")
    
    saliency_results = {}
    
    for method in methods:
        print(f"Computing {method} gradients...")
        gradients = compute_input_gradients(model, test_images, target_class, method=method)
        saliency_results[method] = gradients
    
    return saliency_results

def detect_trigger_regions(saliency_maps, trigger_info, detection_threshold=0.5):
    """
    Detect potential trigger regions based on saliency analysis
    
    Args:
        saliency_maps: Gradient magnitude maps [N, H, W]
        trigger_info: Ground truth trigger information
        detection_threshold: Threshold for trigger detection
    
    Returns:
        detected_regions: Boolean maps of detected trigger regions [N, H, W]
        detection_scores: Confidence scores for trigger presence [N]
    """
    print("Detecting trigger regions from saliency maps...")
    
    # Normalize saliency maps to [0, 1]
    normalized_maps = []
    for saliency_map in saliency_maps:
        norm_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
        normalized_maps.append(norm_map)
    
    normalized_maps = torch.stack(normalized_maps)
    
    # Create ground truth trigger mask for comparison
    trigger_size = trigger_info['size']
    h, w = normalized_maps.shape[1], normalized_maps.shape[2]
    
    if trigger_info['position'] == 'bottom-right':
        gt_mask = torch.zeros(h, w)
        gt_mask[-trigger_size:, -trigger_size:] = 1.0
    
    # Detect high saliency regions
    detected_regions = (normalized_maps > detection_threshold).float()
    
    # Calculate detection scores (overlap with ground truth)
    detection_scores = []
    for detected_map in detected_regions:
        # IoU (Intersection over Union) with ground truth
        intersection = (detected_map * gt_mask).sum()
        union = (detected_map + gt_mask).clamp(max=1).sum()
        iou = intersection / (union + 1e-8)
        detection_scores.append(iou.item())
    
    detection_scores = torch.tensor(detection_scores)
    
    return detected_regions, detection_scores, gt_mask

def visualize_gradient_analysis(test_images, saliency_results, trigger_info, num_samples=6):
    """
    Create comprehensive visualization of gradient analysis
    """
    print(f"Creating gradient analysis visualization for {num_samples} samples...")
    
    class_names = ['airplane', 'car', 'bird', 'cat', 'deer']
    methods = list(saliency_results.keys())
    
    fig, axes = plt.subplots(len(methods) + 2, num_samples, figsize=(3*num_samples, 3*(len(methods)+2)))
    
    for i in range(min(num_samples, len(test_images))):
        # Original image
        img = test_images[i]
        img_display = (img.permute(1, 2, 0) + 1) / 2  # Denormalize
        img_display = torch.clamp(img_display, 0, 1)
        
        axes[0, i].imshow(img_display)
        axes[0, i].set_title(f'Original Image {i+1}')
        axes[0, i].axis('off')
        
        # Triggered image
        triggered_img = apply_trigger_to_image(img, trigger_info)
        triggered_display = (triggered_img.permute(1, 2, 0) + 1) / 2
        triggered_display = torch.clamp(triggered_display, 0, 1)
        
        axes[1, i].imshow(triggered_display)
        axes[1, i].set_title(f'Triggered Image {i+1}')
        axes[1, i].axis('off')
        
        # Add red box to highlight trigger area
        from matplotlib.patches import Rectangle
        if trigger_info['position'] == 'bottom-right':
            rect = Rectangle((32-trigger_info['size'], 32-trigger_info['size']), 
                           trigger_info['size'], trigger_info['size'], 
                           linewidth=2, edgecolor='red', facecolor='none')
            axes[1, i].add_patch(rect)
        
        # Saliency maps for each method
        for j, method in enumerate(methods):
            saliency_map = saliency_results[method][i]
            
            # Apply Gaussian smoothing for better visualization
            saliency_smooth = gaussian_filter(saliency_map.numpy(), sigma=0.8)
            
            im = axes[j+2, i].imshow(saliency_smooth, cmap='hot', alpha=0.8)
            axes[j+2, i].set_title(f'{method.capitalize()} Gradients')
            axes[j+2, i].axis('off')
            
            # Overlay trigger region
            if trigger_info['position'] == 'bottom-right':
                rect = Rectangle((32-trigger_info['size'], 32-trigger_info['size']), 
                               trigger_info['size'], trigger_info['size'], 
                               linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
                axes[j+2, i].add_patch(rect)
    
    plt.tight_layout()
    plt.savefig('plots/gradient_saliency_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Gradient analysis visualization saved to plots/gradient_saliency_analysis.png")

def statistical_gradient_analysis(clean_saliency, backdoor_saliency, trigger_info):
    """
    Perform statistical analysis of gradient differences
    """
    print("Performing statistical analysis of gradient patterns...")
    
    # Extract trigger region gradients
    trigger_size = trigger_info['size']
    h, w = clean_saliency['vanilla'].shape[1], clean_saliency['vanilla'].shape[2]
    
    if trigger_info['position'] == 'bottom-right':
        trigger_region = (slice(-trigger_size, None), slice(-trigger_size, None))
        non_trigger_region = (slice(None, -trigger_size), slice(None, -trigger_size))
    
    results = {}
    
    for method in clean_saliency.keys():
        clean_maps = clean_saliency[method]
        backdoor_maps = backdoor_saliency[method]
        
        # Extract gradients from trigger vs non-trigger regions
        clean_trigger = clean_maps[:, trigger_region[0], trigger_region[1]].flatten()
        backdoor_trigger = backdoor_maps[:, trigger_region[0], trigger_region[1]].flatten()
        
        clean_non_trigger = clean_maps[:, non_trigger_region[0], non_trigger_region[1]].flatten()
        backdoor_non_trigger = backdoor_maps[:, non_trigger_region[0], non_trigger_region[1]].flatten()
        
        # Statistical tests
        # T-test: trigger region differences
        trigger_stat, trigger_p = stats.ttest_ind(clean_trigger, backdoor_trigger)
        
        # T-test: non-trigger region differences  
        non_trigger_stat, non_trigger_p = stats.ttest_ind(clean_non_trigger, backdoor_non_trigger)
        
        # Effect size (Cohen's d)
        def cohens_d(x1, x2):
            n1, n2 = len(x1), len(x2)
            s1, s2 = x1.std(), x2.std()
            pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
            return (x1.mean() - x2.mean()) / pooled_std
        
        trigger_effect_size = cohens_d(backdoor_trigger, clean_trigger)
        non_trigger_effect_size = cohens_d(backdoor_non_trigger, clean_non_trigger)
        
        results[method] = {
            'trigger_ttest': {'statistic': trigger_stat, 'p_value': trigger_p},
            'non_trigger_ttest': {'statistic': non_trigger_stat, 'p_value': non_trigger_p},
            'trigger_effect_size': trigger_effect_size,
            'non_trigger_effect_size': non_trigger_effect_size,
            'trigger_mean_diff': backdoor_trigger.mean() - clean_trigger.mean(),
            'non_trigger_mean_diff': backdoor_non_trigger.mean() - clean_non_trigger.mean()
        }
        
        print(f"\n--- {method.upper()} RESULTS ---")
        print(f"Trigger region - t-stat: {trigger_stat:.3f}, p-value: {trigger_p:.6f}")
        print(f"Non-trigger region - t-stat: {non_trigger_stat:.3f}, p-value: {non_trigger_p:.6f}")
        print(f"Trigger effect size (Cohen's d): {trigger_effect_size:.3f}")
        print(f"Trigger mean difference: {results[method]['trigger_mean_diff']:.6f}")
    
    return results

def save_gradient_analysis_results(saliency_results, statistical_results, trigger_info):
    """
    Save gradient analysis results for Task 3.2
    """
    print("Saving gradient analysis results...")
    
    save_data = {
        'saliency_results': saliency_results,
        'statistical_results': statistical_results,
        'trigger_info': trigger_info
    }
    
    torch.save(save_data, 'models/gradient_analysis_results.pth')
    print("Results saved to models/gradient_analysis_results.pth")

def main():
    """
    Main function to run Task 3.1: Gradient-Based Saliency Analysis
    """
    print("="*60)
    print("LAB 3 - TASK 3.1: GRADIENT-BASED SALIENCY ANALYSIS")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create directories
    os.makedirs('plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Load models and data
    backdoored_model, clean_model, trigger_info = load_models_and_data()
    test_dataset = load_test_dataset()
    
    # Prepare test samples
    num_test_samples = 12
    test_images = []
    test_labels = []
    
    for i in range(min(num_test_samples, len(test_dataset))):
        image, label = test_dataset[i]
        test_images.append(image)
        test_labels.append(label)
    
    test_batch = torch.stack(test_images)
    triggered_batch = torch.stack([apply_trigger_to_image(img, trigger_info) for img in test_images])
    
    print(f"Analyzing {len(test_images)} test samples")
    
    # Create saliency maps for clean model
    target_class = 0  # airplane (backdoor target)
    methods = ['vanilla', 'integrated']
    
    print("\n--- CLEAN MODEL ANALYSIS ---")
    clean_saliency = create_saliency_maps(clean_model, triggered_batch, target_class, methods)
    
    print("\n--- BACKDOORED MODEL ANALYSIS ---")
    backdoor_saliency = create_saliency_maps(backdoored_model, triggered_batch, target_class, methods)
    
    # Detect trigger regions
    detected_regions, detection_scores, gt_mask = detect_trigger_regions(
        backdoor_saliency['vanilla'], trigger_info
    )
    
    # Statistical analysis
    statistical_results = statistical_gradient_analysis(clean_saliency, backdoor_saliency, trigger_info)
    
    # Visualizations
    visualize_gradient_analysis(test_batch, backdoor_saliency, trigger_info)
    
    # Save results
    save_gradient_analysis_results(backdoor_saliency, statistical_results, trigger_info)
    
    # Print summary
    print("\n" + "="*60)
    print("GRADIENT ANALYSIS SUMMARY")
    print("="*60)
    
    avg_detection_score = detection_scores.mean().item()
    print(f"Average trigger detection IoU: {avg_detection_score:.3f}")
    
    # Best method analysis
    best_method = None
    best_p_value = float('inf')
    
    for method, results in statistical_results.items():
        trigger_p = results['trigger_ttest']['p_value']
        if trigger_p < best_p_value:
            best_p_value = trigger_p
            best_method = method
        
        print(f"{method}: trigger p-value = {trigger_p:.6f}, effect size = {results['trigger_effect_size']:.3f}")
    
    print(f"\nBest detection method: {best_method} (p = {best_p_value:.6f})")
    
    target_met = "SUCCESS" if avg_detection_score > 0.3 else "NEEDS IMPROVEMENT"
    significance = "SIGNIFICANT" if best_p_value < 0.05 else "NOT SIGNIFICANT"
    
    print(f"Trigger localization: {target_met} (IoU > 0.3)")
    print(f"Statistical significance: {significance} (p < 0.05)")
    
    print("\n" + "="*60)
    print("TASK 3.1 COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Ready for Task 3.2: Systematic Input Perturbation")

if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
Lab 3 - Task 3.1: Gradient-Based Saliency Analysis
Probabilistic Verification of Outsourced Models

This script implements gradient-based saliency analysis to detect backdoor triggers 
through input sensitivity analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Subset
from datetime import datetime
import os
from scipy import stats
from scipy.ndimage import gaussian_filter

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class SimpleCNN(nn.Module):
    """
    Simple CNN for CIFAR-10 classification (5 classes)
    Same architecture as previous labs
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

def load_models_and_data():
    """
    Load clean and backdoored models for comparison
    """
    print("Loading models for gradient analysis...")
    
    # Load backdoored model from Lab 2
    backdoored_model = SimpleCNN(num_classes=5).to(device)
    backdoored_model.load_state_dict(torch.load('../lab2_backdoor_detection_activation_analysis/models/backdoored_model.pth', weights_only=False))
    backdoored_model.eval()
    
    # Load experiment data to get trigger info
    experiment_data = torch.load('../lab2_backdoor_detection_activation_analysis/models/backdoor_experiment_data.pth', weights_only=False)
    trigger_info = experiment_data['trigger_pattern']
    
    # For comparison, train a clean model (or load from Lab 1 if available)
    try:
        clean_model = SimpleCNN(num_classes=5).to(device)
        clean_model.load_state_dict(torch.load('../lab1_statistical_watermark_detection/models/baseline_model.pth', weights_only=False))
        clean_model.eval()
        print("Loaded clean model from Lab 1")
    except:
        print("Clean model from Lab 1 not found, using backdoored model for demo")
        clean_model = backdoored_model
    
    print("Models loaded successfully")
    return backdoored_model, clean_model, trigger_info

def load_test_dataset():
    """
    Load CIFAR-10 test dataset
    """
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
    Apply trigger pattern to image (same as previous labs)
    """
    triggered_image = image.clone()
    trigger_size = trigger_info['size']
    
    if trigger_info['position'] == 'bottom-right':
        triggered_image[:, -trigger_size:, -trigger_size:] = trigger_info['color']
    
    return triggered_image

def compute_input_gradients(model, inputs, target_class, method='vanilla'):
    """
    Compute gradients of target class probability w.r.t. input pixels
    
    Args:
        model: Neural network model
        inputs: Input tensor [N, C, H, W]
        target_class: Target class for gradient computation
        method: Gradient computation method ('vanilla', 'integrated', 'smoothgrad')
    
    Returns:
        gradient_maps: Gradient magnitude maps [N, H, W]
    """
    model.eval()
    inputs = inputs.to(device)
    inputs.requires_grad_(True)
    
    if method == 'vanilla':
        # Standard gradient computation
        outputs = model(inputs)
        
        # Get probability for target class
        target_probs = F.softmax(outputs, dim=1)[:, target_class]
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=target_probs.sum(),
            inputs=inputs,
            create_graph=False,
            retain_graph=False
        )[0]
        
    elif method == 'integrated':
        # Integrated Gradients (more robust)
        baseline = torch.zeros_like(inputs)
        num_steps = 50
        
        # Interpolate between baseline and input
        alphas = torch.linspace(0, 1, num_steps).to(device)
        gradients_list = []
        
        for alpha in alphas:
            interpolated = baseline + alpha * (inputs - baseline)
            interpolated.requires_grad_(True)
            
            outputs = model(interpolated)
            target_probs = F.softmax(outputs, dim=1)[:, target_class]
            
            grads = torch.autograd.grad(
                outputs=target_probs.sum(),
                inputs=interpolated,
                create_graph=False,
                retain_graph=False
            )[0]
            
            gradients_list.append(grads)
        
        # Average gradients and multiply by input difference
        avg_gradients = torch.stack(gradients_list).mean(dim=0)
        gradients = avg_gradients * (inputs - baseline)
        
    elif method == 'smoothgrad':
        # SmoothGrad (noise-based smoothing)
        num_samples = 25
        noise_level = 0.15
        gradients_list = []
        
        for _ in range(num_samples):
            # Add noise to input
            noise = torch.randn_like(inputs) * noise_level
            noisy_input = inputs + noise
            noisy_input.requires_grad_(True)
            
            outputs = model(noisy_input)
            target_probs = F.softmax(outputs, dim=1)[:, target_class]
            
            grads = torch.autograd.grad(
                outputs=target_probs.sum(),
                inputs=noisy_input,
                create_graph=False,
                retain_graph=False
            )[0]
            
            gradients_list.append(grads)
        
        # Average gradients across noise samples
        gradients = torch.stack(gradients_list).mean(dim=0)
    
    # Compute gradient magnitude (L2 norm across color channels)
    gradient_magnitude = torch.norm(gradients, dim=1)  # [N, H, W]
    
    return gradient_magnitude.detach().cpu()

def analyze_gradient_anomalies(clean_gradients, backdoor_gradients, anomaly_threshold=2.0):
    """
    Detect anomalous gradient patterns indicating triggers
    
    Args:
        clean_gradients: Gradients from clean model [N, H, W]
        backdoor_gradients: Gradients from backdoored model [N, H, W]
        anomaly_threshold: Z-score threshold for anomaly detection
    
    Returns:
        anomaly_maps: Maps showing anomalous regions [N, H, W]
        anomaly_scores: Scalar anomaly scores per image [N]
    """
    print(f"Analyzing gradient anomalies with threshold {anomaly_threshold}...")
    
    # Calculate statistics of clean gradients
    clean_mean = clean_gradients.mean(dim=(1, 2), keepdim=True)  # Per-image mean
    clean_std = clean_gradients.std(dim=(1, 2), keepdim=True)   # Per-image std
    
    # Compute z-scores for backdoor gradients
    z_scores = (backdoor_gradients - clean_mean) / (clean_std + 1e-8)
    
    # Identify anomalous regions (high z-scores)
    anomaly_maps = (torch.abs(z_scores) > anomaly_threshold).float()
    
    # Compute scalar anomaly scores (percentage of anomalous pixels)
    anomaly_scores = anomaly_maps.mean(dim=(1, 2))
    
    return anomaly_maps, anomaly_scores

def create_saliency_maps(model, test_images, target_class, methods=['vanilla', 'integrated']):
    """
    Create saliency maps using different gradient methods
    
    Args:
        model: Neural network model
        test_images: Test images [N, C, H, W]
        target_class: Target class for analysis
        methods: List of gradient computation methods
    
    Returns:
        saliency_results: Dictionary containing saliency maps for each method
    """
    print(f"Creating saliency maps for {len(test_images)} images using methods: {methods}")
    
    saliency_results = {}
    
    for method in methods:
        print(f"Computing {method} gradients...")
        gradients = compute_input_gradients(model, test_images, target_class, method=method)
        saliency_results[method] = gradients
    
    return saliency_results

def detect_trigger_regions(saliency_maps, trigger_info, detection_threshold=0.5):
    """
    Detect potential trigger regions based on saliency analysis
    
    Args:
        saliency_maps: Gradient magnitude maps [N, H, W]
        trigger_info: Ground truth trigger information
        detection_threshold: Threshold for trigger detection
    
    Returns:
        detected_regions: Boolean maps of detected trigger regions [N, H, W]
        detection_scores: Confidence scores for trigger presence [N]
    """
    print("Detecting trigger regions from saliency maps...")
    
    # Normalize saliency maps to [0, 1]
    normalized_maps = []
    for saliency_map in saliency_maps:
        norm_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
        normalized_maps.append(norm_map)
    
    normalized_maps = torch.stack(normalized_maps)
    
    # Create ground truth trigger mask for comparison
    trigger_size = trigger_info['size']
    h, w = normalized_maps.shape[1], normalized_maps.shape[2]
    
    if trigger_info['position'] == 'bottom-right':
        gt_mask = torch.zeros(h, w)
        gt_mask[-trigger_size:, -trigger_size:] = 1.0
    
    # Detect high saliency regions
    detected_regions = (normalized_maps > detection_threshold).float()
    
    # Calculate detection scores (overlap with ground truth)
    detection_scores = []
    for detected_map in detected_regions:
        # IoU (Intersection over Union) with ground truth
        intersection = (detected_map * gt_mask).sum()
        union = (detected_map + gt_mask).clamp(max=1).sum()
        iou = intersection / (union + 1e-8)
        detection_scores.append(iou.item())
    
    detection_scores = torch.tensor(detection_scores)
    
    return detected_regions, detection_scores, gt_mask

def visualize_gradient_analysis(test_images, saliency_results, trigger_info, num_samples=6):
    """
    Create comprehensive visualization of gradient analysis
    """
    print(f"Creating gradient analysis visualization for {num_samples} samples...")
    
    class_names = ['airplane', 'car', 'bird', 'cat', 'deer']
    methods = list(saliency_results.keys())
    
    fig, axes = plt.subplots(len(methods) + 2, num_samples, figsize=(3*num_samples, 3*(len(methods)+2)))
    
    for i in range(min(num_samples, len(test_images))):
        # Original image
        img = test_images[i]
        img_display = (img.permute(1, 2, 0) + 1) / 2  # Denormalize
        img_display = torch.clamp(img_display, 0, 1)
        
        axes[0, i].imshow(img_display)
        axes[0, i].set_title(f'Original Image {i+1}')
        axes[0, i].axis('off')
        
        # Triggered image
        triggered_img = apply_trigger_to_image(img, trigger_info)
        triggered_display = (triggered_img.permute(1, 2, 0) + 1) / 2
        triggered_display = torch.clamp(triggered_display, 0, 1)
        
        axes[1, i].imshow(triggered_display)
        axes[1, i].set_title(f'Triggered Image {i+1}')
        axes[1, i].axis('off')
        
        # Add red box to highlight trigger area
        from matplotlib.patches import Rectangle
        if trigger_info['position'] == 'bottom-right':
            rect = Rectangle((32-trigger_info['size'], 32-trigger_info['size']), 
                           trigger_info['size'], trigger_info['size'], 
                           linewidth=2, edgecolor='red', facecolor='none')
            axes[1, i].add_patch(rect)
        
        # Saliency maps for each method
        for j, method in enumerate(methods):
            saliency_map = saliency_results[method][i]
            
            # Apply Gaussian smoothing for better visualization
            saliency_smooth = gaussian_filter(saliency_map.numpy(), sigma=0.8)
            
            im = axes[j+2, i].imshow(saliency_smooth, cmap='hot', alpha=0.8)
            axes[j+2, i].set_title(f'{method.capitalize()} Gradients')
            axes[j+2, i].axis('off')
            
            # Overlay trigger region
            if trigger_info['position'] == 'bottom-right':
                rect = Rectangle((32-trigger_info['size'], 32-trigger_info['size']), 
                               trigger_info['size'], trigger_info['size'], 
                               linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
                axes[j+2, i].add_patch(rect)
    
    plt.tight_layout()
    plt.savefig('plots/gradient_saliency_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Gradient analysis visualization saved to plots/gradient_saliency_analysis.png")

def statistical_gradient_analysis(clean_saliency, backdoor_saliency, trigger_info):
    """
    Perform statistical analysis of gradient differences
    """
    print("Performing statistical analysis of gradient patterns...")
    
    # Extract trigger region gradients
    trigger_size = trigger_info['size']
    h, w = clean_saliency['vanilla'].shape[1], clean_saliency['vanilla'].shape[2]
    
    if trigger_info['position'] == 'bottom-right':
        trigger_region = (slice(-trigger_size, None), slice(-trigger_size, None))
        non_trigger_region = (slice(None, -trigger_size), slice(None, -trigger_size))
    
    results = {}
    
    for method in clean_saliency.keys():
        clean_maps = clean_saliency[method]
        backdoor_maps = backdoor_saliency[method]
        
        # Extract gradients from trigger vs non-trigger regions
        clean_trigger = clean_maps[:, trigger_region[0], trigger_region[1]].flatten()
        backdoor_trigger = backdoor_maps[:, trigger_region[0], trigger_region[1]].flatten()
        
        clean_non_trigger = clean_maps[:, non_trigger_region[0], non_trigger_region[1]].flatten()
        backdoor_non_trigger = backdoor_maps[:, non_trigger_region[0], non_trigger_region[1]].flatten()
        
        # Statistical tests
        # T-test: trigger region differences
        trigger_stat, trigger_p = stats.ttest_ind(clean_trigger, backdoor_trigger)
        
        # T-test: non-trigger region differences  
        non_trigger_stat, non_trigger_p = stats.ttest_ind(clean_non_trigger, backdoor_non_trigger)
        
        # Effect size (Cohen's d)
        def cohens_d(x1, x2):
            n1, n2 = len(x1), len(x2)
            s1, s2 = x1.std(), x2.std()
            pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
            return (x1.mean() - x2.mean()) / pooled_std
        
        trigger_effect_size = cohens_d(backdoor_trigger, clean_trigger)
        non_trigger_effect_size = cohens_d(backdoor_non_trigger, clean_non_trigger)
        
        results[method] = {
            'trigger_ttest': {'statistic': trigger_stat, 'p_value': trigger_p},
            'non_trigger_ttest': {'statistic': non_trigger_stat, 'p_value': non_trigger_p},
            'trigger_effect_size': trigger_effect_size,
            'non_trigger_effect_size': non_trigger_effect_size,
            'trigger_mean_diff': backdoor_trigger.mean() - clean_trigger.mean(),
            'non_trigger_mean_diff': backdoor_non_trigger.mean() - clean_non_trigger.mean()
        }
        
        print(f"\n--- {method.upper()} RESULTS ---")
        print(f"Trigger region - t-stat: {trigger_stat:.3f}, p-value: {trigger_p:.6f}")
        print(f"Non-trigger region - t-stat: {non_trigger_stat:.3f}, p-value: {non_trigger_p:.6f}")
        print(f"Trigger effect size (Cohen's d): {trigger_effect_size:.3f}")
        print(f"Trigger mean difference: {results[method]['trigger_mean_diff']:.6f}")
    
    return results

def save_gradient_analysis_results(saliency_results, statistical_results, trigger_info):
    """
    Save gradient analysis results for Task 3.2
    """
    print("Saving gradient analysis results...")
    
    save_data = {
        'saliency_results': saliency_results,
        'statistical_results': statistical_results,
        'trigger_info': trigger_info
    }
    
    torch.save(save_data, 'models/gradient_analysis_results.pth')
    print("Results saved to models/gradient_analysis_results.pth")

def main():
    """
    Main function to run Task 3.1: Gradient-Based Saliency Analysis
    """
    print("="*60)
    print("LAB 3 - TASK 3.1: GRADIENT-BASED SALIENCY ANALYSIS")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create directories
    os.makedirs('plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Load models and data
    backdoored_model, clean_model, trigger_info = load_models_and_data()
    test_dataset = load_test_dataset()
    
    # Prepare test samples
    num_test_samples = 12
    test_images = []
    test_labels = []
    
    for i in range(min(num_test_samples, len(test_dataset))):
        image, label = test_dataset[i]
        test_images.append(image)
        test_labels.append(label)
    
    test_batch = torch.stack(test_images)
    triggered_batch = torch.stack([apply_trigger_to_image(img, trigger_info) for img in test_images])
    
    print(f"Analyzing {len(test_images)} test samples")
    
    # Create saliency maps for clean model
    target_class = 0  # airplane (backdoor target)
    methods = ['vanilla', 'integrated']
    
    print("\n--- CLEAN MODEL ANALYSIS ---")
    clean_saliency = create_saliency_maps(clean_model, triggered_batch, target_class, methods)
    
    print("\n--- BACKDOORED MODEL ANALYSIS ---")
    backdoor_saliency = create_saliency_maps(backdoored_model, triggered_batch, target_class, methods)
    
    # Detect trigger regions
    detected_regions, detection_scores, gt_mask = detect_trigger_regions(
        backdoor_saliency['vanilla'], trigger_info
    )
    
    # Statistical analysis
    statistical_results = statistical_gradient_analysis(clean_saliency, backdoor_saliency, trigger_info)
    
    # Visualizations
    visualize_gradient_analysis(test_batch, backdoor_saliency, trigger_info)
    
    # Save results
    save_gradient_analysis_results(backdoor_saliency, statistical_results, trigger_info)
    
    # Print summary
    print("\n" + "="*60)
    print("GRADIENT ANALYSIS SUMMARY")
    print("="*60)
    
    avg_detection_score = detection_scores.mean().item()
    print(f"Average trigger detection IoU: {avg_detection_score:.3f}")
    
    # Best method analysis
    best_method = None
    best_p_value = float('inf')
    
    for method, results in statistical_results.items():
        trigger_p = results['trigger_ttest']['p_value']
        if trigger_p < best_p_value:
            best_p_value = trigger_p
            best_method = method
        
        print(f"{method}: trigger p-value = {trigger_p:.6f}, effect size = {results['trigger_effect_size']:.3f}")
    
    print(f"\nBest detection method: {best_method} (p = {best_p_value:.6f})")
    
    target_met = "SUCCESS" if avg_detection_score > 0.3 else "NEEDS IMPROVEMENT"
    significance = "SIGNIFICANT" if best_p_value < 0.05 else "NOT SIGNIFICANT"
    
    print(f"Trigger localization: {target_met} (IoU > 0.3)")
    print(f"Statistical significance: {significance} (p < 0.05)")
    
    print("\n" + "="*60)
    print("TASK 3.1 COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Ready for Task 3.2: Systematic Input Perturbation")

if __name__ == "__main__":
    main()
