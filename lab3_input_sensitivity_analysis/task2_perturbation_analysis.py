#!/usr/bin/env python3
"""
Lab 3 - Task 3.2: Systematic Input Perturbation
Probabilistic Verification of Outsourced Models

This script implements systematic input perturbation analysis to detect backdoor triggers
by testing how different regions of the input affect model predictions.
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
from sklearn.metrics import roc_auc_score

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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

def load_models_and_data():
    """
    Load models and trigger information from previous tasks
    """
    print("Loading models and data from previous labs...")
    
    # Load backdoored model from Lab 2
    backdoored_model = SimpleCNN(num_classes=5).to(device)
    backdoored_model.load_state_dict(torch.load('../lab2_backdoor_detection_activation_analysis/models/backdoored_model.pth', weights_only=False))
    backdoored_model.eval()
    
    # Load experiment data
    experiment_data = torch.load('../lab2_backdoor_detection_activation_analysis/models/backdoor_experiment_data.pth', weights_only=False)
    trigger_info = experiment_data['trigger_pattern']
    
    # Load clean model for comparison
    try:
        clean_model = SimpleCNN(num_classes=5).to(device)
        clean_model.load_state_dict(torch.load('../lab1_statistical_watermark_detection/models/baseline_model.pth', weights_only=False))
        clean_model.eval()
        print("Loaded clean model from Lab 1")
    except:
        clean_model = backdoored_model
        print("Using backdoored model for demo")
    
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
    Apply trigger pattern to image
    """
    triggered_image = image.clone()
    trigger_size = trigger_info['size']
    
    if trigger_info['position'] == 'bottom-right':
        triggered_image[:, -trigger_size:, -trigger_size:] = trigger_info['color']
    
    return triggered_image

def systematic_perturbation_analysis(model, input_image, perturbation_size=4, perturbation_types=['noise', 'mask', 'blur']):
    """
    Systematically perturb input regions and measure output changes
    
    Args:
        model: Neural network model
        input_image: Input image tensor [C, H, W]
        perturbation_size: Size of perturbation patches
        perturbation_types: Types of perturbations to apply
    
    Returns:
        sensitivity_maps: Dictionary of sensitivity maps for each perturbation type
        baseline_prediction: Original prediction without perturbation
    """
    model.eval()
    input_image = input_image.to(device)
    
    # Get baseline prediction
    with torch.no_grad():
        baseline_output = model(input_image.unsqueeze(0))
        baseline_probs = F.softmax(baseline_output, dim=1)
        baseline_prediction = baseline_output.argmax(dim=1).item()
        baseline_confidence = baseline_probs.max().item()
    
    h, w = input_image.shape[1], input_image.shape[2]
    sensitivity_maps = {}
    
    for pert_type in perturbation_types:
        print(f"Computing {pert_type} perturbation sensitivity...")
        
        # Initialize sensitivity map
        sensitivity_map = torch.zeros(h, w)
        
        # Slide perturbation window across image
        for i in range(0, h - perturbation_size + 1, perturbation_size):
            for j in range(0, w - perturbation_size + 1, perturbation_size):
                # Create perturbed image
                perturbed_image = input_image.clone()
                
                if pert_type == 'noise':
                    # Add random noise to region
                    noise = torch.randn(input_image.shape[0], perturbation_size, perturbation_size) * 0.5
                    perturbed_image[:, i:i+perturbation_size, j:j+perturbation_size] = noise.to(device)
                
                elif pert_type == 'mask':
                    # Mask region with zeros (black)
                    perturbed_image[:, i:i+perturbation_size, j:j+perturbation_size] = 0
                
                elif pert_type == 'blur':
                    # Apply Gaussian blur to region
                    region = perturbed_image[:, i:i+perturbation_size, j:j+perturbation_size]
                    blurred_region = gaussian_filter(region.cpu().numpy(), sigma=1.0, axes=(1, 2))
                    perturbed_image[:, i:i+perturbation_size, j:j+perturbation_size] = torch.tensor(blurred_region).to(device)
                
                elif pert_type == 'replacement':
                    # Replace with mean pixel value
                    mean_val = input_image.mean()
                    perturbed_image[:, i:i+perturbation_size, j:j+perturbation_size] = mean_val
                
                # Measure output change
                with torch.no_grad():
                    perturbed_output = model(perturbed_image.unsqueeze(0))
                    perturbed_probs = F.softmax(perturbed_output, dim=1)
                    perturbed_prediction = perturbed_output.argmax(dim=1).item()
                    perturbed_confidence = perturbed_probs.max().item()
                
                # Calculate sensitivity metrics
                # Option 1: Prediction change (binary)
                prediction_change = float(perturbed_prediction != baseline_prediction)
                
                # Option 2: Confidence change (continuous)
                confidence_change = abs(baseline_confidence - perturbed_confidence)
                
                # Option 3: KL divergence between probability distributions
                kl_div = F.kl_div(perturbed_probs.log(), baseline_probs, reduction='sum').item()
                
                # Use confidence change as primary sensitivity measure
                sensitivity_score = confidence_change
                
                # Assign sensitivity score to all pixels in the patch
                sensitivity_map[i:i+perturbation_size, j:j+perturbation_size] = sensitivity_score
        
        sensitivity_maps[pert_type] = sensitivity_map.cpu()
    
    return sensitivity_maps, baseline_prediction, baseline_confidence

def detect_trigger_regions(sensitivity_map, trigger_info, threshold_percentile=90):
    """
    Detect potential trigger regions based on sensitivity analysis
    
    Args:
        sensitivity_map: Sensitivity map [H, W]
        trigger_info: Ground truth trigger information
        threshold_percentile: Percentile threshold for high sensitivity regions
    
    Returns:
        detected_regions: Boolean map of detected trigger regions
        detection_score: IoU score with ground truth
        threshold: Sensitivity threshold used
    """
    # Calculate threshold based on percentile
    threshold = np.percentile(sensitivity_map.numpy(), threshold_percentile)
    
    # Detect high sensitivity regions
    detected_regions = (sensitivity_map > threshold).float()
    
    # Create ground truth mask
    trigger_size = trigger_info['size']
    h, w = sensitivity_map.shape
    gt_mask = torch.zeros(h, w)
    
    if trigger_info['position'] == 'bottom-right':
        gt_mask[-trigger_size:, -trigger_size:] = 1.0
    
    # Calculate IoU (Intersection over Union)
    intersection = (detected_regions * gt_mask).sum()
    union = (detected_regions + gt_mask).clamp(max=1).sum()
    iou_score = intersection / (union + 1e-8)
    
    return detected_regions, iou_score.item(), threshold

def compare_sensitivity_patterns(clean_sensitivity, backdoor_sensitivity, trigger_info):
    """
    Compare sensitivity patterns between clean and backdoored models
    
    Args:
        clean_sensitivity: Sensitivity maps from clean model
        backdoor_sensitivity: Sensitivity maps from backdoored model
        trigger_info: Trigger information
    
    Returns:
        comparison_results: Statistical comparison results
    """
    print("Comparing sensitivity patterns between models...")
    
    results = {}
    
    # Define trigger and non-trigger regions
    trigger_size = trigger_info['size']
    h, w = list(clean_sensitivity.values())[0].shape
    
    trigger_mask = torch.zeros(h, w)
    if trigger_info['position'] == 'bottom-right':
        trigger_mask[-trigger_size:, -trigger_size:] = 1.0
    
    non_trigger_mask = 1.0 - trigger_mask
    
    for pert_type in clean_sensitivity.keys():
        clean_map = clean_sensitivity[pert_type]
        backdoor_map = backdoor_sensitivity[pert_type]
        
        # Extract sensitivity values for trigger vs non-trigger regions
        clean_trigger = clean_map[trigger_mask.bool()]
        backdoor_trigger = backdoor_map[trigger_mask.bool()]
        
        clean_non_trigger = clean_map[non_trigger_mask.bool()]
        backdoor_non_trigger = backdoor_map[non_trigger_mask.bool()]
        
        # Statistical comparisons
        # T-test: Clean vs Backdoor in trigger region
        trigger_stat, trigger_p = stats.ttest_ind(clean_trigger, backdoor_trigger)
        
        # T-test: Clean vs Backdoor in non-trigger region
        non_trigger_stat, non_trigger_p = stats.ttest_ind(clean_non_trigger, backdoor_non_trigger)
        
        # Effect sizes
        def cohens_d(x1, x2):
            pooled_std = np.sqrt(((len(x1)-1)*x1.std()**2 + (len(x2)-1)*x2.std()**2) / (len(x1)+len(x2)-2))
            return (x1.mean() - x2.mean()) / pooled_std
        
        trigger_effect = cohens_d(backdoor_trigger, clean_trigger)
        non_trigger_effect = cohens_d(backdoor_non_trigger, clean_non_trigger)
        
        # Sensitivity ratio (backdoor/clean)
        trigger_ratio = backdoor_trigger.mean() / (clean_trigger.mean() + 1e-8)
        non_trigger_ratio = backdoor_non_trigger.mean() / (clean_non_trigger.mean() + 1e-8)
        
        results[pert_type] = {
            'trigger_ttest': {'statistic': trigger_stat, 'p_value': trigger_p},
            'non_trigger_ttest': {'statistic': non_trigger_stat, 'p_value': non_trigger_p},
            'trigger_effect_size': trigger_effect,
            'non_trigger_effect_size': non_trigger_effect,
            'trigger_sensitivity_ratio': trigger_ratio,
            'non_trigger_sensitivity_ratio': non_trigger_ratio,
            'trigger_mean_clean': clean_trigger.mean(),
            'trigger_mean_backdoor': backdoor_trigger.mean(),
            'detection_strength': trigger_ratio / non_trigger_ratio  # Relative enhancement
        }
        
        print(f"\n--- {pert_type.upper()} PERTURBATION ---")
        print(f"Trigger region - t-stat: {trigger_stat:.3f}, p-value: {trigger_p:.6f}")
        print(f"Trigger sensitivity ratio (backdoor/clean): {trigger_ratio:.3f}")
        print(f"Detection strength: {results[pert_type]['detection_strength']:.3f}")
    
    return results

def visualize_perturbation_analysis(test_images, clean_sensitivity, backdoor_sensitivity, trigger_info, num_samples=4):
    """
    Create comprehensive visualization of perturbation analysis
    """
    print(f"Creating perturbation analysis visualization for {num_samples} samples...")
    
    # Fix: backdoor_sensitivity is a list, get perturbation types from first element
    perturbation_types = list(backdoor_sensitivity[0].keys())
    
    fig, axes = plt.subplots(3, num_samples * len(perturbation_types), 
                           figsize=(4*num_samples*len(perturbation_types), 12))
    
    if len(perturbation_types) * num_samples == 1:
        axes = axes.reshape(3, 1)
    
    col_idx = 0
    
    for sample_idx in range(min(num_samples, len(test_images))):
        for pert_idx, pert_type in enumerate(perturbation_types):
            # Original image (top row)
            if pert_idx == 0:  # Only show once per sample
                img = test_images[sample_idx]
                triggered_img = apply_trigger_to_image(img, trigger_info)
                img_display = (triggered_img.permute(1, 2, 0) + 1) / 2
                img_display = torch.clamp(img_display, 0, 1)
                
                # Show on first perturbation column for this sample
                axes[0, col_idx].imshow(img_display)
                axes[0, col_idx].set_title(f'Sample {sample_idx+1}\nTriggered Image')
                axes[0, col_idx].axis('off')
                
                # Add trigger highlight
                from matplotlib.patches import Rectangle
                rect = Rectangle((32-trigger_info['size'], 32-trigger_info['size']), 
                               trigger_info['size'], trigger_info['size'], 
                               linewidth=2, edgecolor='red', facecolor='none')
                axes[0, col_idx].add_patch(rect)
            else:
                axes[0, col_idx].axis('off')
                axes[0, col_idx].set_title('')
            
            # Clean model sensitivity (middle row)
            clean_map = clean_sensitivity[sample_idx][pert_type]
            im1 = axes[1, col_idx].imshow(clean_map, cmap='viridis', alpha=0.8)
            axes[1, col_idx].set_title(f'Clean Model\n{pert_type.capitalize()} Sensitivity')
            axes[1, col_idx].axis('off')
            
            # Backdoor model sensitivity (bottom row)
            backdoor_map = backdoor_sensitivity[sample_idx][pert_type]
            im2 = axes[2, col_idx].imshow(backdoor_map, cmap='hot', alpha=0.8)
            axes[2, col_idx].set_title(f'Backdoored Model\n{pert_type.capitalize()} Sensitivity')
            axes[2, col_idx].axis('off')
            
            # Add trigger region overlay
            rect1 = Rectangle((32-trigger_info['size'], 32-trigger_info['size']), 
                            trigger_info['size'], trigger_info['size'], 
                            linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
            rect2 = Rectangle((32-trigger_info['size'], 32-trigger_info['size']), 
                            trigger_info['size'], trigger_info['size'], 
                            linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
            axes[1, col_idx].add_patch(rect1)
            axes[2, col_idx].add_patch(rect2)
            
            col_idx += 1
    
    plt.tight_layout()
    plt.savefig('plots/perturbation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Perturbation analysis visualization saved to plots/perturbation_analysis.png")

def create_sensitivity_comparison_plot(comparison_results, perturbation_types):
    """
    Create detailed comparison plots of sensitivity patterns
    """
    print("Creating sensitivity comparison plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Trigger sensitivity ratios
    ax1 = axes[0, 0]
    ratios = [comparison_results[pt]['trigger_sensitivity_ratio'] for pt in perturbation_types]
    bars1 = ax1.bar(perturbation_types, ratios, color='skyblue', alpha=0.7)
    ax1.axhline(y=1.0, color='red', linestyle='--', label='No difference')
    ax1.set_ylabel('Sensitivity Ratio (Backdoor/Clean)')
    ax1.set_title('Trigger Region Sensitivity Enhancement')
    ax1.legend()
    
    # Add value labels
    for bar, ratio in zip(bars1, ratios):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{ratio:.2f}', ha='center', va='bottom')
    
    # Plot 2: P-values for trigger regions
    ax2 = axes[0, 1]
    p_values = [comparison_results[pt]['trigger_ttest']['p_value'] for pt in perturbation_types]
    bars2 = ax2.bar(perturbation_types, [-np.log10(max(p, 1e-10)) for p in p_values], 
                   color='lightcoral', alpha=0.7)
    ax2.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p=0.05 threshold')
    ax2.set_ylabel('-log10(p-value)')
    ax2.set_title('Statistical Significance (Trigger Region)')
    ax2.legend()
    
    # Plot 3: Detection strength
    ax3 = axes[1, 0]
    strengths = [comparison_results[pt]['detection_strength'] for pt in perturbation_types]
    bars3 = ax3.bar(perturbation_types, strengths, color='lightgreen', alpha=0.7)
    ax3.set_ylabel('Detection Strength')
    ax3.set_title('Relative Trigger Enhancement')
    
    for bar, strength in zip(bars3, strengths):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{strength:.2f}', ha='center', va='bottom')
    
    # Plot 4: Effect sizes
    ax4 = axes[1, 1]
    effect_sizes = [comparison_results[pt]['trigger_effect_size'] for pt in perturbation_types]
    bars4 = ax4.bar(perturbation_types, effect_sizes, color='orange', alpha=0.7)
    ax4.axhline(y=0.8, color='red', linestyle='--', label='Large effect (d=0.8)')
    ax4.set_ylabel("Cohen's d (Effect Size)")
    ax4.set_title('Effect Size of Backdoor Detection')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('plots/sensitivity_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Sensitivity comparison plots saved to plots/sensitivity_comparison.png")

def save_perturbation_results(sensitivity_results, comparison_results, trigger_info):
    """
    Save perturbation analysis results for Task 3.3
    """
    print("Saving perturbation analysis results...")
    
    save_data = {
        'sensitivity_results': sensitivity_results,
        'comparison_results': comparison_results,
        'trigger_info': trigger_info
    }
    
    torch.save(save_data, 'models/perturbation_analysis_results.pth')
    print("Results saved to models/perturbation_analysis_results.pth")

def main():
    """
    Main function to run Task 3.2: Systematic Input Perturbation
    """
    print("="*60)
    print("LAB 3 - TASK 3.2: SYSTEMATIC INPUT PERTURBATION")
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
    num_test_samples = 6
    test_images = []
    
    for i in range(min(num_test_samples, len(test_dataset))):
        image, label = test_dataset[i]
        test_images.append(image)
    
    print(f"Analyzing {len(test_images)} test samples")
    
    # Apply triggers to test images
    triggered_images = [apply_trigger_to_image(img, trigger_info) for img in test_images]
    
    # Perturbation types to test
    perturbation_types = ['noise', 'mask', 'blur']
    
    # Analyze sensitivity for clean model
    print("\n--- CLEAN MODEL PERTURBATION ANALYSIS ---")
    clean_sensitivity = []
    for i, triggered_img in enumerate(triggered_images):
        print(f"Processing clean model - image {i+1}/{len(triggered_images)}")
        sensitivity_maps, pred, conf = systematic_perturbation_analysis(
            clean_model, triggered_img, perturbation_size=4, perturbation_types=perturbation_types
        )
        clean_sensitivity.append(sensitivity_maps)
    
    # Analyze sensitivity for backdoored model
    print("\n--- BACKDOORED MODEL PERTURBATION ANALYSIS ---")
    backdoor_sensitivity = []
    detection_scores = []
    
    for i, triggered_img in enumerate(triggered_images):
        print(f"Processing backdoored model - image {i+1}/{len(triggered_images)}")
        sensitivity_maps, pred, conf = systematic_perturbation_analysis(
            backdoored_model, triggered_img, perturbation_size=4, perturbation_types=perturbation_types
        )
        backdoor_sensitivity.append(sensitivity_maps)
        
        # Calculate detection scores for primary perturbation type
        detected_regions, iou_score, threshold = detect_trigger_regions(
            sensitivity_maps['mask'], trigger_info
        )
        detection_scores.append(iou_score)
    
    # Compare sensitivity patterns
    # Average sensitivity maps across samples for comparison
    avg_clean_sensitivity = {}
    avg_backdoor_sensitivity = {}
    
    for pert_type in perturbation_types:
        clean_maps = torch.stack([sens[pert_type] for sens in clean_sensitivity])
        backdoor_maps = torch.stack([sens[pert_type] for sens in backdoor_sensitivity])
        
        avg_clean_sensitivity[pert_type] = clean_maps.mean(dim=0)
        avg_backdoor_sensitivity[pert_type] = backdoor_maps.mean(dim=0)
    
    comparison_results = compare_sensitivity_patterns(
        avg_clean_sensitivity, avg_backdoor_sensitivity, trigger_info
    )
    
    # Visualizations
    visualize_perturbation_analysis(test_images, clean_sensitivity, backdoor_sensitivity, trigger_info)
    create_sensitivity_comparison_plot(comparison_results, perturbation_types)
    
    # Save results
    save_perturbation_results({
        'clean_sensitivity': clean_sensitivity,
        'backdoor_sensitivity': backdoor_sensitivity,
        'detection_scores': detection_scores
    }, comparison_results, trigger_info)
    
    # Print summary
    print("\n" + "="*60)
    print("PERTURBATION ANALYSIS SUMMARY")
    print("="*60)
    
    avg_detection_score = np.mean(detection_scores)
    print(f"Average trigger detection IoU: {avg_detection_score:.3f}")
    
    # Best perturbation method
    best_method = None
    best_strength = 0
    
    for method, results in comparison_results.items():
        strength = results['detection_strength']
        p_value = results['trigger_ttest']['p_value']
        
        print(f"{method}: detection strength = {strength:.3f}, p-value = {p_value:.6f}")
        
        if strength > best_strength:
            best_strength = strength
            best_method = method
    
    print(f"\nBest perturbation method: {best_method} (strength = {best_strength:.3f})")
    
    # Success criteria
    localization_success = "SUCCESS" if avg_detection_score > 0.3 else "NEEDS IMPROVEMENT"
    statistical_significance = "SIGNIFICANT" if any(
        comparison_results[method]['trigger_ttest']['p_value'] < 0.05 
        for method in perturbation_types
    ) else "NOT SIGNIFICANT"
    
    print(f"Trigger localization: {localization_success} (IoU > 0.3)")
    print(f"Statistical significance: {statistical_significance} (p < 0.05)")
    
    print("\n" + "="*60)
    print("TASK 3.2 COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Ready for Task 3.3: Probabilistic Trigger Localization")

if __name__ == "__main__":
    main()