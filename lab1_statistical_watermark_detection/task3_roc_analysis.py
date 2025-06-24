#!/usr/bin/env python3
"""
Lab 1 - Task 1.3: ROC Analysis
Probabilistic Verification of Outsourced Models

This script performs ROC analysis for watermark detection systems.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
import seaborn as sns
from datetime import datetime
import os
import json
import pandas as pd

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SimpleCNN(nn.Module):
    """
    Simple CNN for CIFAR-10 classification (5 classes)
    Same architecture as previous tasks
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

def load_experimental_results():
    """
    Load results from Tasks 1.1 and 1.2
    """
    print("Loading experimental results from previous tasks...")
    
    # Load Task 1.2 results (statistical detection)
    with open('results/task2_statistical_detection_results.json', 'r') as f:
        detection_results = json.load(f)
    
    # Load Task 1.1 results (watermark data)
    watermark_data = torch.load('models/watermark_data.pth', weights_only=False)
    
    print("Loaded results from Tasks 1.1 and 1.2")
    return detection_results, watermark_data

def generate_additional_models(baseline_model, watermark_inputs, watermark_labels, num_models=20):
    """
    Generate additional models with varying watermark strengths for comprehensive ROC analysis
    
    Args:
        baseline_model: Clean baseline model
        watermark_inputs: Watermark trigger samples
        watermark_labels: Watermark labels
        num_models: Number of additional models to generate
    
    Returns:
        models_data: List of model information and statistics
    """
    print(f"\nGenerating {num_models} additional models for ROC analysis...")
    
    # Define range of watermark strengths
    alphas = np.logspace(-6, -3, num_models)  # From 1e-6 to 1e-3
    
    models_data = []
    
    for i, alpha in enumerate(alphas):
        print(f"Generating model {i+1}/{num_models} (α={alpha:.2e})")
        
        # Create watermarked model
        watermarked_model = SimpleCNN(num_classes=5).to(device)
        watermarked_model.load_state_dict(baseline_model.state_dict())
        
        # Simple watermark embedding (reduced from main task for speed)
        optimizer = torch.optim.Adam(watermarked_model.parameters(), lr=alpha, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        watermarked_model.train()
        for epoch in range(2):  # Quick embedding
            optimizer.zero_grad()
            outputs = watermarked_model(watermark_inputs.to(device))
            loss = criterion(outputs, watermark_labels.to(device))
            
            # Add regularization
            l2_reg = sum(torch.norm(p)**2 for p in watermarked_model.parameters())
            loss += 0.01 * l2_reg
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(watermarked_model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Test watermark effectiveness
        watermarked_model.eval()
        with torch.no_grad():
            outputs = watermarked_model(watermark_inputs.to(device))
            _, predictions = torch.max(outputs, 1)
            watermark_accuracy = (predictions == watermark_labels.to(device)).float().mean().item()
        
        # Calculate detection statistics
        detection_stats = calculate_detection_statistics(watermarked_model, watermark_inputs, watermark_labels)
        
        models_data.append({
            'alpha': alpha,
            'watermark_accuracy': watermark_accuracy,
            'is_watermarked': True,
            'chi_square_stat': detection_stats['chi_square_stat'],
            'chi_square_pvalue': detection_stats['chi_square_pvalue'],
            'accuracy_score': watermark_accuracy
        })
    
    return models_data

def calculate_detection_statistics(model, watermark_inputs, watermark_labels):
    """
    Calculate detection statistics for a single model
    """
    model.eval()
    with torch.no_grad():
        outputs = model(watermark_inputs.to(device))
        _, predictions = torch.max(outputs, 1)
        predictions = predictions.cpu().numpy()
    
    # Chi-square test
    n_classes = 5
    observed_counts = np.bincount(predictions, minlength=n_classes)
    expected_counts = len(predictions) / n_classes
    
    chi_square_stat = np.sum((observed_counts - expected_counts)**2 / expected_counts)
    
    # Calculate p-value (simplified)
    from scipy.stats import chi2
    df = n_classes - 1
    chi_square_pvalue = 1 - chi2.cdf(chi_square_stat, df)
    
    return {
        'chi_square_stat': chi_square_stat,
        'chi_square_pvalue': chi_square_pvalue,
        'predictions': predictions
    }

def generate_clean_models(baseline_model, num_clean_models=10):
    """
    Generate clean (non-watermarked) models with slight variations for ROC analysis
    """
    print(f"\nGenerating {num_clean_models} clean model variations...")
    
    clean_models_data = []
    
    for i in range(num_clean_models):
        # Create slight variations by adding small random noise to weights
        clean_model = SimpleCNN(num_classes=5).to(device)
        clean_model.load_state_dict(baseline_model.state_dict())
        
        # Add small random variations to simulate different training runs
        with torch.no_grad():
            for param in clean_model.parameters():
                noise = torch.randn_like(param) * 0.001  # Small noise
                param.add_(noise)
        
        # Load watermark test data
        watermark_data = torch.load('models/watermark_data.pth', weights_only=False)
        watermark_inputs = watermark_data['watermark_inputs_real']
        watermark_labels = watermark_data['watermark_labels_real']
        
        # Calculate detection statistics
        detection_stats = calculate_detection_statistics(clean_model, watermark_inputs, watermark_labels)
        watermark_accuracy = (detection_stats['predictions'] == watermark_labels.numpy()).mean()
        
        clean_models_data.append({
            'alpha': 0,  # No watermark
            'watermark_accuracy': watermark_accuracy,
            'is_watermarked': False,
            'chi_square_stat': detection_stats['chi_square_stat'],
            'chi_square_pvalue': detection_stats['chi_square_pvalue'],
            'accuracy_score': watermark_accuracy
        })
    
    return clean_models_data

def create_roc_analysis(detection_results, additional_models_data, clean_models_data):
    """
    Create comprehensive ROC analysis from all experimental data
    """
    print("\nCreating ROC analysis...")
    
    # Combine original results with additional models
    all_models_data = []
    
    # Add original experimental results
    for model_name, results in detection_results.items():
        is_watermarked = (model_name != 'baseline')
        all_models_data.append({
            'name': model_name,
            'is_watermarked': is_watermarked,
            'chi_square_stat': results['chi_square']['statistic'],
            'chi_square_pvalue': results['chi_square']['p_value'],
            'accuracy_score': results['accuracy_threshold']['watermark_accuracy'],
            'source': 'original'
        })
    
    # Add additional models
    for data in additional_models_data:
        all_models_data.append({
            'name': f"alpha_{data['alpha']:.2e}",
            'is_watermarked': data['is_watermarked'],
            'chi_square_stat': data['chi_square_stat'],
            'chi_square_pvalue': data['chi_square_pvalue'],
            'accuracy_score': data['accuracy_score'],
            'source': 'generated'
        })
    
    # Add clean models
    for i, data in enumerate(clean_models_data):
        all_models_data.append({
            'name': f"clean_{i}",
            'is_watermarked': data['is_watermarked'],
            'chi_square_stat': data['chi_square_stat'],
            'chi_square_pvalue': data['chi_square_pvalue'],
            'accuracy_score': data['accuracy_score'],
            'source': 'clean'
        })
    
    return all_models_data

def plot_roc_curves(all_models_data):
    """
    Plot ROC curves for different detection methods
    """
    print("\nPlotting ROC curves...")
    
    # Extract data
    y_true = [int(model['is_watermarked']) for model in all_models_data]
    
    # Different detection scores
    chi_square_scores = [model['chi_square_stat'] for model in all_models_data]
    pvalue_scores = [-np.log10(max(model['chi_square_pvalue'], 1e-10)) for model in all_models_data]  # -log10(p-value)
    accuracy_scores = [model['accuracy_score'] for model in all_models_data]
    
    # Create ROC curves
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # ROC Curve 1: Chi-square statistic
    fpr1, tpr1, _ = roc_curve(y_true, chi_square_scores)
    auc1 = auc(fpr1, tpr1)
    ax1.plot(fpr1, tpr1, linewidth=2, label=f'Chi-square (AUC = {auc1:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve: Chi-square Test')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ROC Curve 2: P-value based
    fpr2, tpr2, _ = roc_curve(y_true, pvalue_scores)
    auc2 = auc(fpr2, tpr2)
    ax2.plot(fpr2, tpr2, linewidth=2, label=f'P-value (-log10) (AUC = {auc2:.3f})', color='orange')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve: P-value Test')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # ROC Curve 3: Accuracy based
    fpr3, tpr3, _ = roc_curve(y_true, accuracy_scores)
    auc3 = auc(fpr3, tpr3)
    ax3.plot(fpr3, tpr3, linewidth=2, label=f'Accuracy (AUC = {auc3:.3f})', color='green')
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('ROC Curve: Accuracy Test')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Combined ROC curves
    ax4.plot(fpr1, tpr1, linewidth=2, label=f'Chi-square (AUC = {auc1:.3f})')
    ax4.plot(fpr2, tpr2, linewidth=2, label=f'P-value (AUC = {auc2:.3f})')
    ax4.plot(fpr3, tpr3, linewidth=2, label=f'Accuracy (AUC = {auc3:.3f})')
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')
    ax4.set_title('Combined ROC Curves')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'chi_square_auc': auc1,
        'pvalue_auc': auc2,
        'accuracy_auc': auc3
    }

def plot_precision_recall_curves(all_models_data):
    """
    Plot Precision-Recall curves
    """
    print("\nPlotting Precision-Recall curves...")
    
    y_true = [int(model['is_watermarked']) for model in all_models_data]
    chi_square_scores = [model['chi_square_stat'] for model in all_models_data]
    accuracy_scores = [model['accuracy_score'] for model in all_models_data]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # PR Curve 1: Chi-square
    precision1, recall1, _ = precision_recall_curve(y_true, chi_square_scores)
    ap1 = average_precision_score(y_true, chi_square_scores)
    ax1.plot(recall1, precision1, linewidth=2, label=f'Chi-square (AP = {ap1:.3f})')
    ax1.axhline(y=np.mean(y_true), color='k', linestyle='--', alpha=0.5, label='Random')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall: Chi-square Test')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # PR Curve 2: Accuracy
    precision2, recall2, _ = precision_recall_curve(y_true, accuracy_scores)
    ap2 = average_precision_score(y_true, accuracy_scores)
    ax2.plot(recall2, precision2, linewidth=2, label=f'Accuracy (AP = {ap2:.3f})', color='green')
    ax2.axhline(y=np.mean(y_true), color='k', linestyle='--', alpha=0.5, label='Random')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall: Accuracy Test')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/precision_recall_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'chi_square_ap': ap1,
        'accuracy_ap': ap2
    }

def plot_detection_threshold_analysis(all_models_data):
    """
    Analyze optimal detection thresholds
    """
    print("\nAnalyzing detection thresholds...")
    
    y_true = np.array([int(model['is_watermarked']) for model in all_models_data])
    chi_square_scores = np.array([model['chi_square_stat'] for model in all_models_data])
    accuracy_scores = np.array([model['accuracy_score'] for model in all_models_data])
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Chi-square threshold analysis
    thresholds = np.linspace(0, max(chi_square_scores), 100)
    tpr_list, fpr_list, f1_list = [], [], []
    
    for thresh in thresholds:
        predictions = (chi_square_scores >= thresh).astype(int)
        tp = np.sum((predictions == 1) & (y_true == 1))
        fp = np.sum((predictions == 1) & (y_true == 0))
        fn = np.sum((predictions == 0) & (y_true == 1))
        tn = np.sum((predictions == 0) & (y_true == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tpr
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        f1_list.append(f1)
    
    ax1.plot(thresholds, tpr_list, label='True Positive Rate', linewidth=2)
    ax1.plot(thresholds, fpr_list, label='False Positive Rate', linewidth=2)
    ax1.set_xlabel('Chi-square Threshold')
    ax1.set_ylabel('Rate')
    ax1.set_title('TPR vs FPR by Chi-square Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # F1 score by threshold
    ax2.plot(thresholds, f1_list, linewidth=2, color='purple')
    optimal_idx = np.argmax(f1_list)
    optimal_threshold = thresholds[optimal_idx]
    ax2.axvline(x=optimal_threshold, color='red', linestyle='--', 
                label=f'Optimal = {optimal_threshold:.2f}')
    ax2.set_xlabel('Chi-square Threshold')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score by Chi-square Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Distribution of scores
    watermarked_scores = chi_square_scores[y_true == 1]
    clean_scores = chi_square_scores[y_true == 0]
    
    ax3.hist(clean_scores, bins=20, alpha=0.7, label='Clean Models', color='blue')
    ax3.hist(watermarked_scores, bins=20, alpha=0.7, label='Watermarked Models', color='red')
    ax3.axvline(x=optimal_threshold, color='green', linestyle='--', linewidth=2, 
                label=f'Optimal Threshold = {optimal_threshold:.2f}')
    ax3.set_xlabel('Chi-square Statistic')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Chi-square Statistics')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Accuracy score distribution
    watermarked_acc = accuracy_scores[y_true == 1]
    clean_acc = accuracy_scores[y_true == 0]
    
    ax4.hist(clean_acc, bins=20, alpha=0.7, label='Clean Models', color='blue')
    ax4.hist(watermarked_acc, bins=20, alpha=0.7, label='Watermarked Models', color='red')
    ax4.set_xlabel('Watermark Accuracy')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Watermark Accuracies')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return optimal_threshold

def create_summary_report(roc_results, pr_results, optimal_threshold, all_models_data):
    """
    Create comprehensive summary report
    """
    print("\nCreating summary report...")
    
    # Calculate statistics
    watermarked_models = [m for m in all_models_data if m['is_watermarked']]
    clean_models = [m for m in all_models_data if not m['is_watermarked']]
    
    report = {
        'experiment_summary': {
            'total_models': len(all_models_data),
            'watermarked_models': len(watermarked_models),
            'clean_models': len(clean_models),
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'roc_analysis': {
            'chi_square_auc': roc_results['chi_square_auc'],
            'pvalue_auc': roc_results['pvalue_auc'],
            'accuracy_auc': roc_results['accuracy_auc']
        },
        'precision_recall_analysis': {
            'chi_square_ap': pr_results['chi_square_ap'],
            'accuracy_ap': pr_results['accuracy_ap']
        },
        'threshold_analysis': {
            'optimal_chi_square_threshold': optimal_threshold
        },
        'performance_summary': {
            'best_detection_method': max(roc_results.items(), key=lambda x: x[1]),
            'detection_difficulty': 'Easy' if max(roc_results.values()) > 0.9 else 'Moderate' if max(roc_results.values()) > 0.7 else 'Hard'
        }
    }
    
    # Save report
    with open('results/task3_roc_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("ROC ANALYSIS SUMMARY REPORT")
    print("="*60)
    print(f"Total Models Analyzed: {report['experiment_summary']['total_models']}")
    print(f"Watermarked Models: {report['experiment_summary']['watermarked_models']}")
    print(f"Clean Models: {report['experiment_summary']['clean_models']}")
    print("\nDetection Performance (AUC):")
    print(f"  Chi-square Test: {report['roc_analysis']['chi_square_auc']:.3f}")
    print(f"  P-value Test: {report['roc_analysis']['pvalue_auc']:.3f}")
    print(f"  Accuracy Test: {report['roc_analysis']['accuracy_auc']:.3f}")
    print(f"\nBest Detection Method: {report['performance_summary']['best_detection_method'][0]} (AUC = {report['performance_summary']['best_detection_method'][1]:.3f})")
    print(f"Detection Difficulty: {report['performance_summary']['detection_difficulty']}")
    print(f"Optimal Chi-square Threshold: {report['threshold_analysis']['optimal_chi_square_threshold']:.2f}")
    
    return report

def main():
    """
    Main function to run Task 1.3: ROC Analysis
    """
    print("="*60)
    print("LAB 1 - TASK 1.3: ROC ANALYSIS")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create directories if they don't exist
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Load experimental results from previous tasks
    detection_results, watermark_data = load_experimental_results()
    
    # Load baseline model
    baseline_model = SimpleCNN(num_classes=5).to(device)
    baseline_model.load_state_dict(torch.load('models/baseline_model.pth', weights_only=False))
    baseline_model.eval()
    
    # Get watermark test data
    watermark_inputs = watermark_data['watermark_inputs_real']
    watermark_labels = watermark_data['watermark_labels_real']
    
    # Generate additional models for comprehensive ROC analysis
    additional_models = generate_additional_models(baseline_model, watermark_inputs, watermark_labels, num_models=15)
    
    # Generate clean model variations
    clean_models = generate_clean_models(baseline_model, num_clean_models=8)
    
    # Combine all data for ROC analysis
    all_models_data = create_roc_analysis(detection_results, additional_models, clean_models)
    
    # Perform ROC analysis
    roc_results = plot_roc_curves(all_models_data)
    
    # Perform Precision-Recall analysis
    pr_results = plot_precision_recall_curves(all_models_data)
    
    # Analyze detection thresholds
    optimal_threshold = plot_detection_threshold_analysis(all_models_data)
    
    # Create summary report
    report = create_summary_report(roc_results, pr_results, optimal_threshold, all_models_data)
    
    print("\n" + "="*60)
    print("TASK 1.3 COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Files saved:")
    print("- plots/roc_curves.png")
    print("- plots/precision_recall_curves.png") 
    print("- plots/threshold_analysis.png")
    print("- results/task3_roc_analysis_report.json")
    print("\nROC analysis demonstrates the trade-offs between detection")
    print("accuracy and false positive rates for watermark detection!")

if __name__ == "__main__":
    main()