#!/usr/bin/env python3
"""
Lab 1 - Task 1.2: Statistical Detection Framework
Probabilistic Verification of Outsourced Models

This script implements statistical hypothesis testing methods for watermark detection.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2, kstest, anderson
import seaborn as sns
from datetime import datetime
import os
import json

# Import model architecture from task1
import sys
sys.path.append('.')

class SimpleCNN(nn.Module):
    """
    Simple CNN for CIFAR-10 classification (5 classes)
    Same architecture as Task 1.1
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

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_models_and_data():
    """
    Load all models and watermark data from Task 1.1
    """
    print("Loading models and watermark data from Task 1.1...")
    
    # Load baseline model
    baseline_model = SimpleCNN(num_classes=5).to(device)
    baseline_model.load_state_dict(torch.load('models/baseline_model.pth', map_location=device, weights_only=False))
    baseline_model.eval()
    
    # Load watermarked models
    watermarked_models = {}
    model_names = ['very_weak', 'weak', 'medium', 'strong']
    
    for name in model_names:
        model = SimpleCNN(num_classes=5).to(device)
        model.load_state_dict(torch.load(f'models/watermarked_model_{name}.pth', map_location=device, weights_only=False))
        model.eval()
        watermarked_models[name] = model
    
    # Load watermark data
    watermark_data = torch.load('models/watermark_data.pth', map_location=device, weights_only=False)
    
    print(f"Loaded baseline model and {len(watermarked_models)} watermarked models")
    return baseline_model, watermarked_models, watermark_data

def get_model_predictions(model, inputs, return_probabilities=True):
    """
    Get model predictions on given inputs
    
    Args:
        model: Neural network model
        inputs: Input tensor
        return_probabilities: If True, return softmax probabilities; else return logits
    
    Returns:
        predictions: Model outputs (probabilities or logits)
        predicted_classes: Predicted class indices
    """
    model.eval()
    with torch.no_grad():
        inputs = inputs.to(device)
        logits = model(inputs)
        
        if return_probabilities:
            predictions = torch.softmax(logits, dim=1)
        else:
            predictions = logits
            
        _, predicted_classes = torch.max(logits, 1)
    
    return predictions.cpu().numpy(), predicted_classes.cpu().numpy()

def chi_square_goodness_of_fit_test(observed_predictions, expected_distribution, alpha=0.05):
    """
    Perform chi-square goodness-of-fit test for watermark detection
    
    H0: Model outputs follow expected (random) distribution
    H1: Model outputs do not follow expected distribution (watermark present)
    
    Args:
        observed_predictions: Array of predicted class labels
        expected_distribution: Expected probability for each class (uniform for random)
        alpha: Significance level
    
    Returns:
        test_statistic: Chi-square test statistic
        p_value: P-value of the test
        is_watermarked: Boolean indicating if watermark is detected
        critical_value: Critical value for the test
    """
    n_samples = len(observed_predictions)
    n_classes = len(expected_distribution)
    
    # Count observed frequencies for each class
    observed_counts = np.bincount(observed_predictions, minlength=n_classes)
    
    # Calculate expected counts based on expected distribution
    expected_counts = n_samples * np.array(expected_distribution)
    
    # Ensure no zero expected counts (add small epsilon if needed)
    expected_counts = np.maximum(expected_counts, 1e-6)
    
    # Compute chi-square test statistic
    test_statistic = np.sum((observed_counts - expected_counts)**2 / expected_counts)
    
    # Degrees of freedom
    df = n_classes - 1
    
    # Calculate p-value
    p_value = 1 - chi2.cdf(test_statistic, df)
    
    # Critical value
    critical_value = chi2.ppf(1 - alpha, df)
    
    # Decision: reject H0 if test_statistic > critical_value (or p_value < alpha)
    is_watermarked = (test_statistic > critical_value)
    
    return test_statistic, p_value, is_watermarked, critical_value

def kolmogorov_smirnov_test(observed_probs, expected_distribution, alpha=0.05):
    """
    Perform Kolmogorov-Smirnov test for distribution matching
    
    Args:
        observed_probs: 2D array of output probabilities [n_samples, n_classes]
        expected_distribution: Expected uniform distribution
        alpha: Significance level
    
    Returns:
        ks_statistics: KS test statistics for each class
        p_values: P-values for each class
        is_watermarked: Boolean indicating if watermark is detected
    """
    n_classes = observed_probs.shape[1]
    ks_statistics = []
    p_values = []
    
    for class_idx in range(n_classes):
        # Get probabilities for this class
        class_probs = observed_probs[:, class_idx]
        
        # Create uniform distribution for comparison
        uniform_samples = np.random.uniform(0, 1, len(class_probs))
        
        # Perform KS test
        ks_stat, p_val = kstest(class_probs, 'uniform')
        
        ks_statistics.append(ks_stat)
        p_values.append(p_val)
    
    # Consider watermarked if any class shows significant deviation
    is_watermarked = any(p < alpha for p in p_values)
    
    return ks_statistics, p_values, is_watermarked

def likelihood_ratio_test(observed_predictions, baseline_predictions, alpha=0.05):
    """
    Perform likelihood ratio test comparing observed vs baseline predictions
    
    Args:
        observed_predictions: Predictions from test model
        baseline_predictions: Predictions from baseline (clean) model
        alpha: Significance level
    
    Returns:
        lr_statistic: Likelihood ratio test statistic
        p_value: P-value of the test
        is_watermarked: Boolean indicating if watermark is detected
    """
    n_classes = 5  # Number of classes in our setup
    
    # Count frequencies for both distributions
    obs_counts = np.bincount(observed_predictions, minlength=n_classes)
    base_counts = np.bincount(baseline_predictions, minlength=n_classes)
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-6
    obs_probs = (obs_counts + epsilon) / (np.sum(obs_counts) + n_classes * epsilon)
    base_probs = (base_counts + epsilon) / (np.sum(base_counts) + n_classes * epsilon)
    
    # Calculate log-likelihood ratio
    log_lr = 0
    for i in range(n_classes):
        if obs_counts[i] > 0:
            log_lr += obs_counts[i] * np.log(obs_probs[i] / base_probs[i])
    
    # Likelihood ratio test statistic (follows chi-square distribution)
    lr_statistic = 2 * log_lr
    
    # Degrees of freedom
    df = n_classes - 1
    
    # Calculate p-value
    p_value = 1 - chi2.cdf(lr_statistic, df)
    
    # Decision
    is_watermarked = (p_value < alpha)
    
    return lr_statistic, p_value, is_watermarked

def statistical_watermark_test(model, watermark_inputs, expected_outputs, alpha=0.05):
    """
    Comprehensive statistical test for watermark presence using multiple methods
    
    Args:
        model: Model to test
        watermark_inputs: Watermark input samples
        expected_outputs: Expected output labels for watermark samples
        alpha: Significance level
    
    Returns:
        results: Dictionary containing results from all statistical tests
    """
    print(f"Performing statistical watermark test (α = {alpha})...")
    
    # Get model predictions
    probs, predictions = get_model_predictions(model, watermark_inputs, return_probabilities=True)
    
    # Expected uniform distribution (random guessing)
    uniform_dist = np.ones(5) / 5  # 5 classes, equal probability
    
    # Test 1: Chi-square goodness-of-fit test
    chi2_stat, chi2_p, chi2_watermarked, chi2_critical = chi_square_goodness_of_fit_test(
        predictions, uniform_dist, alpha
    )
    
    # Test 2: Kolmogorov-Smirnov test
    ks_stats, ks_p_values, ks_watermarked = kolmogorov_smirnov_test(
        probs, uniform_dist, alpha
    )
    
    # Test 3: Compare with expected watermark outputs
    # Calculate accuracy on watermark samples
    watermark_accuracy = np.mean(predictions == expected_outputs.cpu().numpy())
    
    # Simple threshold-based test (if accuracy > threshold, likely watermarked)
    accuracy_threshold = 0.8  # 80% accuracy threshold
    threshold_watermarked = (watermark_accuracy > accuracy_threshold)
    
    results = {
        'chi_square': {
            'statistic': chi2_stat,
            'p_value': chi2_p,
            'is_watermarked': chi2_watermarked,
            'critical_value': chi2_critical
        },
        'kolmogorov_smirnov': {
            'statistics': ks_stats,
            'p_values': ks_p_values,
            'is_watermarked': ks_watermarked
        },
        'accuracy_threshold': {
            'watermark_accuracy': watermark_accuracy,
            'threshold': accuracy_threshold,
            'is_watermarked': threshold_watermarked
        },
        'predictions': predictions,
        'probabilities': probs
    }
    
    return results

def analyze_all_models():
    """
    Analyze all models (baseline + watermarked) using statistical tests
    """
    print("="*60)
    print("STATISTICAL WATERMARK DETECTION ANALYSIS")
    print("="*60)
    
    # Load models and data
    baseline_model, watermarked_models, watermark_data = load_models_and_data()
    
    # Use the new data structure from fixed Task 1.1
    if 'watermark_inputs_real' in watermark_data:
        watermark_inputs = watermark_data['watermark_inputs_real']
        watermark_labels = watermark_data['watermark_labels_real']
        print("Using real images with trigger patches")
    else:
        # Fallback to old structure
        watermark_inputs = watermark_data['watermark_inputs_noise']
        watermark_labels = watermark_data['watermark_labels_noise']
        print("Using artificial watermark data")
    
    all_results = {}
    
    # Test baseline model (should not be watermarked)
    print("\nTesting BASELINE model...")
    baseline_results = statistical_watermark_test(
        baseline_model, watermark_inputs, watermark_labels
    )
    all_results['baseline'] = baseline_results
    
    print_test_results('BASELINE', baseline_results)
    
    # Test all watermarked models
    for model_name, model in watermarked_models.items():
        print(f"\nTesting {model_name.upper()} watermarked model...")
        results = statistical_watermark_test(
            model, watermark_inputs, watermark_labels
        )
        all_results[model_name] = results
        
        print_test_results(model_name.upper(), results)
    
    return all_results

def print_test_results(model_name, results):
    """
    Print formatted results for a single model
    """
    print(f"\n--- {model_name} MODEL RESULTS ---")
    
    # Chi-square test
    chi2 = results['chi_square']
    print(f"Chi-square Test:")
    print(f"  Statistic: {chi2['statistic']:.4f}")
    print(f"  P-value: {chi2['p_value']:.6f}")
    print(f"  Critical value: {chi2['critical_value']:.4f}")
    print(f"  Watermarked: {'YES' if chi2['is_watermarked'] else 'NO'}")
    
    # KS test
    ks = results['kolmogorov_smirnov']
    print(f"Kolmogorov-Smirnov Test:")
    print(f"  Min P-value: {min(ks['p_values']):.6f}")
    print(f"  Watermarked: {'YES' if ks['is_watermarked'] else 'NO'}")
    
    # Accuracy test
    acc = results['accuracy_threshold']
    print(f"Accuracy Threshold Test:")
    print(f"  Watermark accuracy: {acc['watermark_accuracy']:.2%}")
    print(f"  Threshold: {acc['threshold']:.2%}")
    print(f"  Watermarked: {'YES' if acc['is_watermarked'] else 'NO'}")

def create_detection_visualizations(all_results):
    """
    Create visualizations for statistical detection results
    """
    print("\nCreating detection visualizations...")
    
    # Prepare data for plotting
    model_names = list(all_results.keys())
    chi2_stats = [all_results[name]['chi_square']['statistic'] for name in model_names]
    chi2_pvals = [all_results[name]['chi_square']['p_value'] for name in model_names]
    accuracies = [all_results[name]['accuracy_threshold']['watermark_accuracy'] for name in model_names]
    
    # Create subplot figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Chi-square statistics
    bars1 = ax1.bar(model_names, chi2_stats, color='skyblue', alpha=0.7)
    ax1.axhline(y=all_results['baseline']['chi_square']['critical_value'], 
                color='red', linestyle='--', label='Critical Value (α=0.05)')
    ax1.set_ylabel('Chi-square Statistic')
    ax1.set_title('Chi-square Test Statistics')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, stat in zip(bars1, chi2_stats):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{stat:.2f}', ha='center', va='bottom')
    
    # Plot 2: P-values (log scale)
    bars2 = ax2.bar(model_names, chi2_pvals, color='lightcoral', alpha=0.7)
    ax2.axhline(y=0.05, color='red', linestyle='--', label='Significance Level (α=0.05)')
    ax2.set_ylabel('P-value')
    ax2.set_title('Chi-square Test P-values')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Watermark accuracies
    bars3 = ax3.bar(model_names, accuracies, color='lightgreen', alpha=0.7)
    ax3.axhline(y=0.2, color='blue', linestyle='--', label='Random Accuracy (20%)')
    ax3.axhline(y=0.8, color='red', linestyle='--', label='Detection Threshold (80%)')
    ax3.set_ylabel('Watermark Accuracy')
    ax3.set_title('Watermark Recognition Accuracy')
    ax3.legend()
    ax3.tick_params(axis='x', rotation=45)
    
    # Add percentage labels
    for bar, acc in zip(bars3, accuracies):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.1%}', ha='center', va='bottom')
    
    # Plot 4: Detection summary heatmap
    detection_matrix = []
    test_names = ['Chi-square', 'KS Test', 'Accuracy']
    
    for model_name in model_names:
        results = all_results[model_name]
        detections = [
            1 if results['chi_square']['is_watermarked'] else 0,
            1 if results['kolmogorov_smirnov']['is_watermarked'] else 0,
            1 if results['accuracy_threshold']['is_watermarked'] else 0
        ]
        detection_matrix.append(detections)
    
    detection_matrix = np.array(detection_matrix).T
    
    im = ax4.imshow(detection_matrix, aspect='auto', cmap='RdYlBu_r')
    ax4.set_xticks(range(len(model_names)))
    ax4.set_xticklabels(model_names, rotation=45)
    ax4.set_yticks(range(len(test_names)))
    ax4.set_yticklabels(test_names)
    ax4.set_title('Detection Results Summary\n(Red = Detected, Blue = Not Detected)')
    
    # Add text annotations
    for i in range(len(test_names)):
        for j in range(len(model_names)):
            text = 'YES' if detection_matrix[i, j] == 1 else 'NO'
            ax4.text(j, i, text, ha="center", va="center", 
                    color="white" if detection_matrix[i, j] == 1 else "black", 
                    fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/statistical_detection_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved to plots/statistical_detection_results.png")

def save_results_to_file(all_results, filename='results/task2_statistical_detection_results.json'):
    """
    Save all results to JSON file for later analysis
    """
    os.makedirs('results', exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for model_name, results in all_results.items():
        serializable_results[model_name] = {
            'chi_square': {
                'statistic': float(results['chi_square']['statistic']),
                'p_value': float(results['chi_square']['p_value']),
                'is_watermarked': bool(results['chi_square']['is_watermarked']),
                'critical_value': float(results['chi_square']['critical_value'])
            },
            'kolmogorov_smirnov': {
                'statistics': [float(x) for x in results['kolmogorov_smirnov']['statistics']],
                'p_values': [float(x) for x in results['kolmogorov_smirnov']['p_values']],
                'is_watermarked': bool(results['kolmogorov_smirnov']['is_watermarked'])
            },
            'accuracy_threshold': {
                'watermark_accuracy': float(results['accuracy_threshold']['watermark_accuracy']),
                'threshold': float(results['accuracy_threshold']['threshold']),
                'is_watermarked': bool(results['accuracy_threshold']['is_watermarked'])
            }
        }
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {filename}")

def main():
    """
    Main function to run Task 1.2
    """
    print("="*60)
    print("LAB 1 - TASK 1.2: STATISTICAL DETECTION FRAMEWORK")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create directories if they don't exist
    os.makedirs('results', exist_ok=True)
    
    # Run statistical analysis on all models
    all_results = analyze_all_models()
    
    # Create visualizations
    create_detection_visualizations(all_results)
    
    # Save results
    save_results_to_file(all_results)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    detection_count = 0
    for model_name, results in all_results.items():
        if model_name != 'baseline':
            # Count how many tests detected watermark
            detections = [
                results['chi_square']['is_watermarked'],
                results['kolmogorov_smirnov']['is_watermarked'],
                results['accuracy_threshold']['is_watermarked']
            ]
            detection_rate = sum(detections) / len(detections)
            print(f"{model_name}: {sum(detections)}/3 tests detected watermark ({detection_rate:.1%})")
            if detection_rate > 0.5:
                detection_count += 1
    
    print(f"\nOverall: {detection_count}/4 watermarked models successfully detected")
    
    print("\n" + "="*60)
    print("TASK 1.2 COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Files saved:")
    print("- plots/statistical_detection_results.png")
    print("- results/task2_statistical_detection_results.json")

if __name__ == "__main__":
    main()