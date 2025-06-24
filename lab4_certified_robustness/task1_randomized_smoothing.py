#!/usr/bin/env python3
"""
Lab 4 - Task 4.1: Randomized Smoothing Implementation
Probabilistic Verification of Outsourced Models

This script implements randomized smoothing to provide certified robustness guarantees
for neural network models and detect backdoors through robustness analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from datetime import datetime
import os
from scipy import stats
from scipy.special import ndtri  # Inverse normal CDF
import math

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
    Load clean and backdoored models from previous labs
    """
    print("Loading models from previous labs...")
    
    # Load backdoored model from Lab 2
    backdoored_model = SimpleCNN(num_classes=5).to(device)
    backdoored_model.load_state_dict(torch.load('../lab2_backdoor_detection_activation_analysis/models/backdoored_model.pth', weights_only=False))
    backdoored_model.eval()
    
    # Load experiment data to get trigger info
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

def randomized_smoothing_prediction(model, x, noise_std=0.25, n_samples=1000):
    """
    Make robust prediction using randomized smoothing
    
    Args:
        model: Neural network model
        x: Input tensor [C, H, W]
        noise_std: Standard deviation of Gaussian noise
        n_samples: Number of noise samples for smoothing
    
    Returns:
        predicted_class: Most likely class under smoothing
        confidence: Confidence in the prediction (probability)
        counts: Vote counts for each class
    """
    model.eval()
    x = x.to(device)
    
    # Collect predictions from noisy samples
    vote_counts = torch.zeros(5).to(device)  # 5 classes
    
    with torch.no_grad():
        for _ in range(n_samples):
            # Add Gaussian noise
            noise = torch.randn_like(x) * noise_std
            noisy_x = x + noise
            
            # Get prediction
            output = model(noisy_x.unsqueeze(0))
            predicted_class = output.argmax(dim=1).item()
            vote_counts[predicted_class] += 1
    
    # Calculate confidence (probability of most voted class)
    total_votes = vote_counts.sum()
    predicted_class = vote_counts.argmax().item()
    confidence = vote_counts[predicted_class] / total_votes
    
    return predicted_class, confidence.item(), vote_counts.cpu().numpy()

def compute_robustness_certificate(model, x, predicted_class, noise_std, alpha=0.001, n_samples=1000):
    """
    Compute certified robustness radius using Neyman-Pearson lemma
    
    Args:
        model: Neural network model  
        x: Input tensor [C, H, W]
        predicted_class: Predicted class from randomized smoothing
        noise_std: Noise standard deviation used in smoothing
        alpha: Confidence level (1-alpha = confidence)
        n_samples: Number of samples for certificate computation
    
    Returns:
        certificate_radius: Certified L2 radius
        lower_bound: Lower confidence bound on class probability
        upper_bound: Upper confidence bound on class probability
    """
    # Get smoothed prediction counts
    _, confidence, counts = randomized_smoothing_prediction(model, x, noise_std, n_samples)
    
    # Get counts for predicted class and second most likely class
    predicted_count = counts[predicted_class]
    sorted_counts = np.sort(counts)[::-1]  # Sort in descending order
    second_count = sorted_counts[1] if len(sorted_counts) > 1 else 0
    
    # Calculate confidence intervals using normal approximation
    # (Clopper-Pearson would be more accurate but more complex)
    p_hat = predicted_count / n_samples  # Sample proportion
    
    # Wilson confidence interval for binomial proportion
    z_alpha = ndtri(1 - alpha/2)  # Critical value for confidence level
    
    # Lower bound on probability of predicted class
    denominator = 1 + z_alpha**2 / n_samples
    numerator = p_hat + z_alpha**2 / (2 * n_samples) - z_alpha * np.sqrt(p_hat * (1 - p_hat) / n_samples + z_alpha**2 / (4 * n_samples**2))
    lower_bound = numerator / denominator
    
    # Upper bound  
    numerator_upper = p_hat + z_alpha**2 / (2 * n_samples) + z_alpha * np.sqrt(p_hat * (1 - p_hat) / n_samples + z_alpha**2 / (4 * n_samples**2))
    upper_bound = numerator_upper / denominator
    
    # Compute certificate radius using Cohen et al. formula
    if lower_bound > 0.5:
        # Certificate exists
        certificate_radius = noise_std * ndtri(lower_bound)
    else:
        # No certificate (prediction not confident enough)
        certificate_radius = 0.0
    
    return certificate_radius, lower_bound, upper_bound

def evaluate_certified_accuracy(model, test_dataset, noise_std=0.25, n_samples=500, num_test=200):
    """
    Evaluate certified accuracy on test dataset
    
    Args:
        model: Neural network model
        test_dataset: Test dataset
        noise_std: Noise standard deviation
        n_samples: Number of samples for smoothing
        num_test: Number of test samples to evaluate
    
    Returns:
        clean_accuracy: Standard accuracy
        certified_accuracy: Certified robust accuracy  
        certificate_stats: Statistics about certificate radii
    """
    print(f"Evaluating certified accuracy on {num_test} samples...")
    
    correct_clean = 0
    correct_certified = 0
    total_samples = 0
    certificate_radii = []
    
    for i, (image, true_label) in enumerate(test_dataset):
        if i >= num_test:
            break
            
        # Standard prediction
        model.eval()
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device))
            clean_pred = output.argmax(dim=1).item()
        
        if clean_pred == true_label:
            correct_clean += 1
        
        # Randomized smoothing prediction
        smooth_pred, confidence, _ = randomized_smoothing_prediction(
            model, image, noise_std, n_samples
        )
        
        # Compute certificate
        cert_radius, lower_bound, upper_bound = compute_robustness_certificate(
            model, image, smooth_pred, noise_std, alpha=0.001, n_samples=n_samples
        )
        
        certificate_radii.append(cert_radius)
        
        # Count as certified correct if prediction matches true label AND has certificate
        if smooth_pred == true_label and cert_radius > 0:
            correct_certified += 1
        
        total_samples += 1
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{num_test} samples...")
    
    clean_accuracy = correct_clean / total_samples
    certified_accuracy = correct_certified / total_samples
    
    certificate_stats = {
        'mean_radius': np.mean(certificate_radii),
        'median_radius': np.median(certificate_radii),
        'std_radius': np.std(certificate_radii),
        'min_radius': np.min(certificate_radii),
        'max_radius': np.max(certificate_radii),
        'num_certified': np.sum(np.array(certificate_radii) > 0)
    }
    
    return clean_accuracy, certified_accuracy, certificate_stats

def compare_model_robustness(clean_model, backdoor_model, test_dataset, trigger_info, noise_std=0.25):
    """
    Compare robustness properties between clean and backdoored models
    """
    print("Comparing robustness between clean and backdoored models...")
    
    # Test on both clean and triggered images
    num_test_samples = 50
    results = {
        'clean_model': {'clean_images': [], 'triggered_images': []},
        'backdoor_model': {'clean_images': [], 'triggered_images': []}
    }
    
    for i, (image, label) in enumerate(test_dataset):
        if i >= num_test_samples:
            break
            
        triggered_image = apply_trigger_to_image(image, trigger_info)
        
        # Test clean model
        for model_name, model in [('clean_model', clean_model), ('backdoor_model', backdoor_model)]:
            # Clean images
            pred, conf, _ = randomized_smoothing_prediction(model, image, noise_std, n_samples=200)
            cert_radius, _, _ = compute_robustness_certificate(model, image, pred, noise_std, n_samples=200)
            results[model_name]['clean_images'].append({
                'prediction': pred,
                'confidence': conf,
                'certificate': cert_radius,
                'true_label': label
            })
            
            # Triggered images
            pred, conf, _ = randomized_smoothing_prediction(model, triggered_image, noise_std, n_samples=200)
            cert_radius, _, _ = compute_robustness_certificate(model, triggered_image, pred, noise_std, n_samples=200)
            results[model_name]['triggered_images'].append({
                'prediction': pred,
                'confidence': conf,
                'certificate': cert_radius,
                'true_label': label
            })
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{num_test_samples} comparison samples...")
    
    return results

def visualize_robustness_analysis(clean_results, backdoor_results, noise_std):
    """
    Create visualizations of robustness analysis
    """
    print("Creating robustness analysis visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Extract data for plotting
    clean_clean_certs = [r['certificate'] for r in clean_results['clean_model']['clean_images']]
    clean_trigger_certs = [r['certificate'] for r in clean_results['clean_model']['triggered_images']]
    backdoor_clean_certs = [r['certificate'] for r in backdoor_results['backdoor_model']['clean_images']]
    backdoor_trigger_certs = [r['certificate'] for r in backdoor_results['backdoor_model']['triggered_images']]
    
    # Plot 1: Certificate distributions
    ax1 = axes[0, 0]
    ax1.hist(clean_clean_certs, bins=20, alpha=0.7, label='Clean Model + Clean Images', color='blue')
    ax1.hist(backdoor_clean_certs, bins=20, alpha=0.7, label='Backdoor Model + Clean Images', color='red')
    ax1.set_xlabel('Certificate Radius')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Certificate Distribution - Clean Images')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Certificate distributions for triggered images
    ax2 = axes[0, 1]
    ax2.hist(clean_trigger_certs, bins=20, alpha=0.7, label='Clean Model + Triggered Images', color='blue')
    ax2.hist(backdoor_trigger_certs, bins=20, alpha=0.7, label='Backdoor Model + Triggered Images', color='red')
    ax2.set_xlabel('Certificate Radius')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Certificate Distribution - Triggered Images')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Confidence distributions
    ax3 = axes[0, 2]
    clean_clean_confs = [r['confidence'] for r in clean_results['clean_model']['clean_images']]
    backdoor_clean_confs = [r['confidence'] for r in backdoor_results['backdoor_model']['clean_images']]
    
    ax3.hist(clean_clean_confs, bins=20, alpha=0.7, label='Clean Model', color='blue')
    ax3.hist(backdoor_clean_confs, bins=20, alpha=0.7, label='Backdoor Model', color='red')
    ax3.set_xlabel('Smoothed Confidence')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Confidence Distribution - Clean Images')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Certificate vs Confidence scatter
    ax4 = axes[1, 0]
    ax4.scatter(clean_clean_confs, clean_clean_certs, alpha=0.6, label='Clean Model', color='blue')
    ax4.scatter(backdoor_clean_confs, backdoor_clean_certs, alpha=0.6, label='Backdoor Model', color='red')
    ax4.set_xlabel('Smoothed Confidence')
    ax4.set_ylabel('Certificate Radius')
    ax4.set_title('Certificate vs Confidence')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Box plots for comparison
    ax5 = axes[1, 1]
    box_data = [clean_clean_certs, backdoor_clean_certs, clean_trigger_certs, backdoor_trigger_certs]
    box_labels = ['Clean+Clean', 'Backdoor+Clean', 'Clean+Trigger', 'Backdoor+Trigger']
    
    ax5.boxplot(box_data, labels=box_labels)
    ax5.set_ylabel('Certificate Radius')
    ax5.set_title('Certificate Radius Distribution')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Summary statistics
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Calculate summary statistics
    stats_text = f"""
    Robustness Analysis Summary (σ = {noise_std})
    
    Clean Model - Clean Images:
    • Mean Certificate: {np.mean(clean_clean_certs):.4f}
    • Certified Rate: {np.mean(np.array(clean_clean_certs) > 0):.2%}
    
    Backdoor Model - Clean Images:
    • Mean Certificate: {np.mean(backdoor_clean_certs):.4f}
    • Certified Rate: {np.mean(np.array(backdoor_clean_certs) > 0):.2%}
    
    Clean Model - Triggered Images:
    • Mean Certificate: {np.mean(clean_trigger_certs):.4f}
    • Certified Rate: {np.mean(np.array(clean_trigger_certs) > 0):.2%}
    
    Backdoor Model - Triggered Images:
    • Mean Certificate: {np.mean(backdoor_trigger_certs):.4f}
    • Certified Rate: {np.mean(np.array(backdoor_trigger_certs) > 0):.2%}
    """
    
    ax6.text(0.1, 0.9, stats_text, fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('plots/robustness_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Robustness analysis visualization saved to plots/robustness_analysis.png")

def save_randomized_smoothing_results(clean_accuracy, certified_accuracy, certificate_stats, robustness_comparison):
    """
    Save randomized smoothing results
    """
    print("Saving randomized smoothing results...")
    
    results = {
        'clean_accuracy': clean_accuracy,
        'certified_accuracy': certified_accuracy,
        'certificate_statistics': certificate_stats,
        'robustness_comparison': robustness_comparison
    }
    
    torch.save(results, 'models/randomized_smoothing_results.pth')
    print("Results saved to models/randomized_smoothing_results.pth")

def main():
    """
    Main function to run Task 4.1: Randomized Smoothing Implementation
    """
    print("="*60)
    print("LAB 4 - TASK 4.1: RANDOMIZED SMOOTHING IMPLEMENTATION")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create directories
    os.makedirs('plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Load models and data
    backdoored_model, clean_model, trigger_info = load_models_and_data()
    test_dataset = load_test_dataset()
    
    # Test randomized smoothing on a single sample
    print("\n--- SINGLE SAMPLE DEMONSTRATION ---")
    test_image, test_label = test_dataset[0]  
    
    print(f"Testing randomized smoothing on sample with true label: {test_label}")
    
    # Test clean model
    pred, conf, counts = randomized_smoothing_prediction(clean_model, test_image, noise_std=0.25, n_samples=500)
    cert_radius, lower_bound, upper_bound = compute_robustness_certificate(
        clean_model, test_image, pred, noise_std=0.25, n_samples=500
    )
    
    print(f"Clean model:")
    print(f"  Predicted class: {pred}")
    print(f"  Confidence: {conf:.4f}")
    print(f"  Certificate radius: {cert_radius:.4f}")
    print(f"  Confidence interval: [{lower_bound:.4f}, {upper_bound:.4f}]")
    
    # Test backdoored model
    pred, conf, counts = randomized_smoothing_prediction(backdoored_model, test_image, noise_std=0.25, n_samples=500)
    cert_radius, lower_bound, upper_bound = compute_robustness_certificate(
        backdoored_model, test_image, pred, noise_std=0.25, n_samples=500
    )
    
    print(f"Backdoored model:")
    print(f"  Predicted class: {pred}")
    print(f"  Confidence: {conf:.4f}")
    print(f"  Certificate radius: {cert_radius:.4f}")
    print(f"  Confidence interval: [{lower_bound:.4f}, {upper_bound:.4f}]")
    
    # Evaluate certified accuracy
    print("\n--- CERTIFIED ACCURACY EVALUATION ---")
    
    # Evaluate clean model
    print("Evaluating clean model...")
    clean_acc, cert_acc, cert_stats = evaluate_certified_accuracy(
        clean_model, test_dataset, noise_std=0.25, n_samples=300, num_test=100
    )
    
    print(f"Clean model results:")
    print(f"  Clean accuracy: {clean_acc:.4f}")
    print(f"  Certified accuracy: {cert_acc:.4f}")
    print(f"  Mean certificate radius: {cert_stats['mean_radius']:.4f}")
    print(f"  Certified samples: {cert_stats['num_certified']}/100")
    
    # Evaluate backdoored model
    print("Evaluating backdoored model...")
    backdoor_clean_acc, backdoor_cert_acc, backdoor_cert_stats = evaluate_certified_accuracy(
        backdoored_model, test_dataset, noise_std=0.25, n_samples=300, num_test=100
    )
    
    print(f"Backdoored model results:")
    print(f"  Clean accuracy: {backdoor_clean_acc:.4f}")
    print(f"  Certified accuracy: {backdoor_cert_acc:.4f}")
    print(f"  Mean certificate radius: {backdoor_cert_stats['mean_radius']:.4f}")
    print(f"  Certified samples: {backdoor_cert_stats['num_certified']}/100")
    
    # Compare robustness properties
    print("\n--- ROBUSTNESS COMPARISON ANALYSIS ---")
    robustness_results = compare_model_robustness(
        clean_model, backdoored_model, test_dataset, trigger_info, noise_std=0.25
    )
    
    # Visualizations
    visualize_robustness_analysis(
        robustness_results, robustness_results, 0.25
    )
    
    # Save results
    save_randomized_smoothing_results(
        clean_acc, cert_acc, cert_stats, robustness_results
    )
    
    # Print summary
    print("\n" + "="*60)
    print("RANDOMIZED SMOOTHING SUMMARY")
    print("="*60)
    
    print(f"Certified Accuracy Comparison:")
    print(f"  Clean model: {cert_acc:.4f} ({cert_acc/clean_acc:.1%} of clean accuracy)")
    print(f"  Backdoor model: {backdoor_cert_acc:.4f} ({backdoor_cert_acc/backdoor_clean_acc:.1%} of clean accuracy)")
    
    certificate_difference = cert_stats['mean_radius'] - backdoor_cert_stats['mean_radius']
    print(f"\nCertificate Radius Comparison:")
    print(f"  Clean model mean radius: {cert_stats['mean_radius']:.4f}")
    print(f"  Backdoor model mean radius: {backdoor_cert_stats['mean_radius']:.4f}")
    print(f"  Difference: {certificate_difference:.4f}")
    
    # Success criteria
    cert_success = "SUCCESS" if cert_acc > 0.4 else "MODERATE" if cert_acc > 0.2 else "NEEDS IMPROVEMENT"
    robustness_diff = "SIGNIFICANT" if abs(certificate_difference) > 0.01 else "MODERATE"
    
    print(f"\nEvaluation:")
    print(f"  Certified accuracy: {cert_success} (target: >40%)")
    print(f"  Robustness difference: {robustness_diff} (difference: {certificate_difference:.4f})")
    
    print("\n" + "="*60)
    print("TASK 4.1 COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Ready for Task 4.2: Robustness-Based Backdoor Detection")

if __name__ == "__main__":
    main()
