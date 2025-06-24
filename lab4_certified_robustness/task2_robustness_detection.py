#!/usr/bin/env python3
"""
Lab 4 - Task 4.2: Robustness-Based Backdoor Detection
Statistical Analysis of Robustness Certificates for Backdoor Detection

This script implements statistical methods to detect backdoors by analyzing
the distribution of robustness certificates across different models and inputs.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ks_2samp, mannwhitneyu, ttest_ind
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class SimpleNet(nn.Module):
    """Simple CNN for demonstration"""
    def __init__(self, num_classes=10):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def add_backdoor_trigger(image, trigger_size=4, trigger_value=1.0):
    """Add a simple square trigger to image"""
    backdoored = image.clone()
    # Add trigger in bottom-right corner
    backdoored[:, -trigger_size:, -trigger_size:] = trigger_value
    return backdoored

def randomized_smoothing_prediction(model, x, noise_std=0.25, n_samples=1000):
    """
    Make robust prediction using randomized smoothing
    Returns: (predicted_class, confidence, certificate_radius)
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Ensure input is on correct device
    if isinstance(x, torch.Tensor):
        x = x.to(device)
    
    votes = torch.zeros(10)  # Assuming 10 classes
    
    with torch.no_grad():
        for _ in range(n_samples):
            # Add Gaussian noise
            noise = torch.randn_like(x) * noise_std
            noisy_x = x + noise
            
            # Get prediction
            logits = model(noisy_x.unsqueeze(0))
            pred = torch.argmax(logits, dim=1).item()
            votes[pred] += 1
    
    # Get majority vote
    predicted_class = torch.argmax(votes).item()
    confidence = votes[predicted_class].item() / n_samples
    
    # Compute certificate radius using Neyman-Pearson lemma
    if confidence > 0.5:
        # Convert confidence to statistical bound
        alpha = 0.001  # Significance level
        z_score = stats.norm.ppf(confidence - alpha)
        certificate_radius = noise_std * z_score
    else:
        certificate_radius = 0.0
    
    return predicted_class, confidence, certificate_radius

def compute_robustness_certificate(model, x, predicted_class, noise_std, alpha=0.001):
    """
    Compute certified robustness radius with statistical guarantees
    """
    n_samples = 1000
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device)
    
    correct_votes = 0
    
    with torch.no_grad():
        for _ in range(n_samples):
            noise = torch.randn_like(x) * noise_std
            noisy_x = x + noise
            logits = model(noisy_x.unsqueeze(0))
            pred = torch.argmax(logits, dim=1).item()
            
            if pred == predicted_class:
                correct_votes += 1
    
    confidence = correct_votes / n_samples
    
    # Compute certificate with statistical confidence
    if confidence > 0.5:
        # Hoeffding's bound for confidence interval
        hoeffding_bound = np.sqrt(np.log(2/alpha) / (2 * n_samples))
        lower_bound = max(0.5, confidence - hoeffding_bound)
        
        # Certificate radius using inverse normal CDF
        z_score = stats.norm.ppf(lower_bound)
        certificate_radius = noise_std * z_score
    else:
        certificate_radius = 0.0
    
    return certificate_radius

def analyze_robustness_distribution(model, test_inputs, test_labels, noise_levels=[0.1, 0.25, 0.5]):
    """
    Analyze distribution of robustness certificates
    Returns: robustness statistics for backdoor detection
    """
    print("🔍 Analyzing robustness distribution...")
    
    robustness_stats = {
        'certificates': {noise: [] for noise in noise_levels},
        'confidences': {noise: [] for noise in noise_levels},
        'predictions': {noise: [] for noise in noise_levels},
        'correct_predictions': {noise: [] for noise in noise_levels}
    }
    
    model.eval()
    device = next(model.parameters()).device
    
    # Analyze subset of test data
    n_test = min(100, len(test_inputs))
    indices = np.random.choice(len(test_inputs), n_test, replace=False)
    
    for i, idx in enumerate(indices):
        if i % 20 == 0:
            print(f"  Processing sample {i+1}/{n_test}...")
            
        x = test_inputs[idx].to(device)
        true_label = test_labels[idx].item()
        
        for noise_std in noise_levels:
            # Get smoothed prediction
            pred_class, confidence, cert_radius = randomized_smoothing_prediction(
                model, x, noise_std=noise_std, n_samples=500  # Reduced for speed
            )
            
            # Store results
            robustness_stats['certificates'][noise_std].append(cert_radius)
            robustness_stats['confidences'][noise_std].append(confidence)
            robustness_stats['predictions'][noise_std].append(pred_class)
            robustness_stats['correct_predictions'][noise_std].append(pred_class == true_label)
    
    # Compute summary statistics
    summary_stats = {}
    for noise_std in noise_levels:
        certs = np.array(robustness_stats['certificates'][noise_std])
        confs = np.array(robustness_stats['confidences'][noise_std])
        correct = np.array(robustness_stats['correct_predictions'][noise_std])
        
        summary_stats[noise_std] = {
            'mean_certificate': np.mean(certs),
            'std_certificate': np.std(certs),
            'median_certificate': np.median(certs),
            'mean_confidence': np.mean(confs),
            'certified_accuracy': np.mean(correct),
            'certificate_distribution': certs,
            'confidence_distribution': confs
        }
    
    return robustness_stats, summary_stats

def statistical_robustness_test(clean_certificates, test_certificates, alpha=0.05):
    """
    Statistical test for detecting abnormal robustness patterns
    """
    print("📊 Performing statistical robustness tests...")
    
    results = {}
    
    # Convert to numpy arrays
    clean_certs = np.array(clean_certificates)
    test_certs = np.array(test_certificates)
    
    # Remove zeros for some tests
    clean_nonzero = clean_certs[clean_certs > 0]
    test_nonzero = test_certs[test_certs > 0]
    
    # 1. Kolmogorov-Smirnov test (distribution comparison)
    ks_stat, ks_pvalue = ks_2samp(clean_certs, test_certs)
    results['ks_test'] = {
        'statistic': ks_stat,
        'p_value': ks_pvalue,
        'significant': ks_pvalue < alpha,
        'interpretation': 'Distributions differ significantly' if ks_pvalue < alpha else 'Distributions similar'
    }
    
    # 2. Mann-Whitney U test (median comparison)
    mw_stat, mw_pvalue = mannwhitneyu(clean_certs, test_certs, alternative='two-sided')
    results['mannwhitney_test'] = {
        'statistic': mw_stat,
        'p_value': mw_pvalue,
        'significant': mw_pvalue < alpha,
        'interpretation': 'Medians differ significantly' if mw_pvalue < alpha else 'Medians similar'
    }
    
    # 3. T-test for means (if data is roughly normal)
    if len(clean_nonzero) > 10 and len(test_nonzero) > 10:
        t_stat, t_pvalue = ttest_ind(clean_nonzero, test_nonzero, equal_var=False)
        results['t_test'] = {
            'statistic': t_stat,
            'p_value': t_pvalue,
            'significant': t_pvalue < alpha,
            'interpretation': 'Means differ significantly' if t_pvalue < alpha else 'Means similar'
        }
    
    # 4. Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(clean_certs)-1)*np.var(clean_certs) + 
                         (len(test_certs)-1)*np.var(test_certs)) / 
                        (len(clean_certs) + len(test_certs) - 2))
    
    if pooled_std > 0:
        cohens_d = (np.mean(clean_certs) - np.mean(test_certs)) / pooled_std
        results['effect_size'] = {
            'cohens_d': cohens_d,
            'magnitude': 'Large' if abs(cohens_d) > 0.8 else 'Medium' if abs(cohens_d) > 0.5 else 'Small'
        }
    
    # 5. Backdoor Detection Decision
    significant_tests = sum([
        results['ks_test']['significant'],
        results['mannwhitney_test']['significant'],
        results.get('t_test', {}).get('significant', False)
    ])
    
    # Decision criteria
    mean_diff = np.mean(clean_certs) - np.mean(test_certs)
    large_effect = abs(results.get('effect_size', {}).get('cohens_d', 0)) > 0.5
    
    results['backdoor_detection'] = {
        'significant_tests_count': significant_tests,
        'mean_difference': mean_diff,
        'large_effect_size': large_effect,
        'detection': (significant_tests >= 2) and (mean_diff > 0.01) and large_effect,
        'confidence': 'High' if significant_tests >= 2 and large_effect else 'Medium' if significant_tests >= 1 else 'Low'
    }
    
    return results

def visualize_robustness_analysis(clean_stats, backdoor_stats, test_results):
    """Create comprehensive visualizations for robustness analysis"""
    print("📈 Creating robustness analysis visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Robustness-Based Backdoor Detection Analysis', fontsize=16, fontweight='bold')
    
    # 1. Certificate Distribution Comparison
    ax = axes[0, 0]
    noise_level = 0.25  # Focus on standard noise level
    
    clean_certs = clean_stats[noise_level]['certificate_distribution']
    backdoor_certs = backdoor_stats[noise_level]['certificate_distribution']
    
    ax.hist(clean_certs, bins=20, alpha=0.7, label='Clean Model', color='green', density=True)
    ax.hist(backdoor_certs, bins=20, alpha=0.7, label='Backdoored Model', color='red', density=True)
    ax.set_xlabel('Certificate Radius')
    ax.set_ylabel('Density')
    ax.set_title(f'Certificate Distribution (σ={noise_level})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Box Plot Comparison
    ax = axes[0, 1]
    data_to_plot = [clean_certs, backdoor_certs]
    box_plot = ax.boxplot(data_to_plot, labels=['Clean', 'Backdoored'], patch_artist=True)
    box_plot['boxes'][0].set_facecolor('lightgreen')
    box_plot['boxes'][1].set_facecolor('lightcoral')
    ax.set_ylabel('Certificate Radius')
    ax.set_title('Certificate Radius Comparison')
    ax.grid(True, alpha=0.3)
    
    # 3. Confidence vs Certificate Scatter
    ax = axes[0, 2]
    clean_confs = clean_stats[noise_level]['confidence_distribution']
    backdoor_confs = backdoor_stats[noise_level]['confidence_distribution']
    
    ax.scatter(clean_confs, clean_certs, alpha=0.6, label='Clean', color='green', s=30)
    ax.scatter(backdoor_confs, backdoor_certs, alpha=0.6, label='Backdoored', color='red', s=30)
    ax.set_xlabel('Prediction Confidence')
    ax.set_ylabel('Certificate Radius')
    ax.set_title('Confidence vs Certificate Radius')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Statistical Test Results
    ax = axes[1, 0]
    test_names = ['KS Test', 'Mann-Whitney', 'T-Test']
    p_values = [
        test_results['ks_test']['p_value'],
        test_results['mannwhitney_test']['p_value'],
        test_results.get('t_test', {}).get('p_value', 1.0)
    ]
    significance = [p < 0.05 for p in p_values]
    
    colors = ['red' if sig else 'blue' for sig in significance]
    bars = ax.bar(test_names, [-np.log10(p) for p in p_values], color=colors, alpha=0.7)
    ax.axhline(y=-np.log10(0.05), color='black', linestyle='--', label='α=0.05')
    ax.set_ylabel('-log₁₀(p-value)')
    ax.set_title('Statistical Test Significance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add p-values as text
    for i, (bar, p_val) in enumerate(zip(bars, p_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'p={p_val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 5. Multi-noise Level Analysis
    ax = axes[1, 1]
    noise_levels = [0.1, 0.25, 0.5]
    clean_means = [clean_stats[noise]['mean_certificate'] for noise in noise_levels]
    backdoor_means = [backdoor_stats[noise]['mean_certificate'] for noise in noise_levels]
    
    x = np.arange(len(noise_levels))
    width = 0.35
    
    ax.bar(x - width/2, clean_means, width, label='Clean', color='green', alpha=0.7)
    ax.bar(x + width/2, backdoor_means, width, label='Backdoored', color='red', alpha=0.7)
    ax.set_xlabel('Noise Level (σ)')
    ax.set_ylabel('Mean Certificate Radius')
    ax.set_title('Certificate Radius vs Noise Level')
    ax.set_xticks(x)
    ax.set_xticklabels(noise_levels)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Detection Summary
    ax = axes[1, 2]
    ax.axis('off')
    
    # Create detection summary text
    detection_result = test_results['backdoor_detection']
    summary_text = f"""
    🎯 BACKDOOR DETECTION SUMMARY
    
    Detection: {'✅ BACKDOOR DETECTED' if detection_result['detection'] else '❌ NO BACKDOOR DETECTED'}
    Confidence: {detection_result['confidence']}
    
    📊 Statistical Evidence:
    • Significant tests: {detection_result['significant_tests_count']}/3
    • Mean difference: {detection_result['mean_difference']:.4f}
    • Effect size: {test_results.get('effect_size', {}).get('magnitude', 'Unknown')}
    
    📈 Key Metrics (σ=0.25):
    • Clean mean cert: {clean_stats[0.25]['mean_certificate']:.4f}
    • Backdoor mean cert: {backdoor_stats[0.25]['mean_certificate']:.4f}
    • Clean cert accuracy: {clean_stats[0.25]['certified_accuracy']:.3f}
    • Backdoor cert accuracy: {backdoor_stats[0.25]['certified_accuracy']:.3f}
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('robustness_backdoor_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main execution function"""
    print("🚀 Lab 4 - Task 4.2: Robustness-Based Backdoor Detection")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create models
    print("\n🏗️  Creating models...")
    clean_model = SimpleNet(num_classes=10).to(device)
    backdoor_model = SimpleNet(num_classes=10).to(device)
    
    # Initialize with same weights then modify backdoor model
    clean_model.load_state_dict(clean_model.state_dict())
    backdoor_model.load_state_dict(clean_model.state_dict())
    
    # Simulate backdoor by modifying some weights (simple approach)
    with torch.no_grad():
        # Modify final layer to create backdoor behavior
        backdoor_model.fc2.weight.data += torch.randn_like(backdoor_model.fc2.weight.data) * 0.1
        backdoor_model.fc2.bias.data += torch.randn_like(backdoor_model.fc2.bias.data) * 0.1
    
    # Generate test data
    print("\n📊 Generating test data...")
    n_samples = 200
    test_inputs = torch.randn(n_samples, 3, 32, 32)  # CIFAR-10 like data
    test_labels = torch.randint(0, 10, (n_samples,))
    
    # Create backdoored test data
    backdoor_test_inputs = test_inputs.clone()
    for i in range(len(backdoor_test_inputs)):
        if i % 5 == 0:  # Add trigger to 20% of samples
            backdoor_test_inputs[i] = add_backdoor_trigger(backdoor_test_inputs[i])
    
    # Analyze robustness for clean model
    print("\n🔍 Analyzing clean model robustness...")
    clean_robustness, clean_summary = analyze_robustness_distribution(
        clean_model, test_inputs, test_labels
    )
    
    # Analyze robustness for backdoored model  
    print("\n🔍 Analyzing backdoored model robustness...")
    backdoor_robustness, backdoor_summary = analyze_robustness_distribution(
        backdoor_model, backdoor_test_inputs, test_labels
    )
    
    # Perform statistical tests
    print("\n📊 Performing statistical robustness tests...")
    # Use certificates from standard noise level (0.25)
    clean_certs = clean_summary[0.25]['certificate_distribution']
    backdoor_certs = backdoor_summary[0.25]['certificate_distribution']
    
    test_results = statistical_robustness_test(clean_certs, backdoor_certs)
    
    # Print results
    print("\n" + "="*60)
    print("📋 ROBUSTNESS ANALYSIS RESULTS")
    print("="*60)
    
    for noise in [0.1, 0.25, 0.5]:
        print(f"\n📊 Noise Level σ = {noise}:")
        print(f"  Clean Model:")
        print(f"    Mean Certificate: {clean_summary[noise]['mean_certificate']:.4f}")
        print(f"    Certified Accuracy: {clean_summary[noise]['certified_accuracy']:.3f}")
        print(f"  Backdoored Model:")
        print(f"    Mean Certificate: {backdoor_summary[noise]['mean_certificate']:.4f}")
        print(f"    Certified Accuracy: {backdoor_summary[noise]['certified_accuracy']:.3f}")
    
    print(f"\n🧪 Statistical Test Results:")
    for test_name, result in test_results.items():
        if test_name == 'backdoor_detection':
            continue
        print(f"  {test_name}:")
        if 'p_value' in result:
            print(f"    p-value: {result['p_value']:.6f}")
            print(f"    Significant: {result['significant']}")
            print(f"    {result['interpretation']}")
    
    print(f"\n🎯 Backdoor Detection Result:")
    detection = test_results['backdoor_detection']
    print(f"  Detection: {'✅ BACKDOOR DETECTED' if detection['detection'] else '❌ NO BACKDOOR DETECTED'}")
    print(f"  Confidence: {detection['confidence']}")
    print(f"  Significant tests: {detection['significant_tests_count']}/3")
    
    # Create visualizations
    visualize_robustness_analysis(clean_summary, backdoor_summary, test_results)
    
    print(f"\n✅ Task 4.2 completed successfully!")
    print(f"📊 Visualization saved as 'robustness_backdoor_analysis.png'")
    
    return {
        'clean_summary': clean_summary,
        'backdoor_summary': backdoor_summary,
        'test_results': test_results,
        'detection_success': detection['detection']
    }

if __name__ == "__main__":
    results = main()