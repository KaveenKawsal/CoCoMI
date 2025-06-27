# visualize.py

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_results(metrics, save_dir):
    """Plots and saves training and validation metrics."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    epochs = range(1, len(metrics['train_losses']) + 1)

    # Loss
    axes[0, 0].plot(epochs, metrics['train_losses'], 'bo-', label='Train Loss')
    axes[0, 0].plot(epochs, metrics['val_losses'], 'ro-', label='Validation Loss')
    axes[0, 0].set_title('Loss vs. Epochs')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy
    axes[0, 1].plot(epochs, metrics['train_accuracies'], 'bo-', label='Train Accuracy')
    axes[0, 1].plot(epochs, metrics['val_accuracies'], 'ro-', label='Validation Accuracy')
    axes[0, 1].set_title('Accuracy vs. Epochs')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # F1 Score
    axes[1, 0].plot(epochs, metrics['train_f1s'], 'bo-', label='Train F1 Score (Weighted)')
    axes[1, 0].plot(epochs, metrics['val_f1s'], 'ro-', label='Validation F1 Score (Weighted)')
    axes[1, 0].set_title('F1 Score vs. Epochs')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Jaccard Index
    axes[1, 1].plot(epochs, metrics['train_jaccards'], 'bo-', label='Train Jaccard (Weighted)')
    axes[1, 1].plot(epochs, metrics['val_jaccards'], 'ro-', label='Validation Jaccard (Weighted)')
    axes[1, 1].set_title('Jaccard Index vs. Epochs')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Jaccard Index')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.close()

def plot_experiment_comparison(results, metric, title, save_path):
    """Plots a comparison of a specific metric across different experiments."""
    plt.figure(figsize=(12, 8))
    for name, data in results.items():
        plt.plot(data[metric], label=name.replace('_', ' '))
    
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(metric.replace("_", " ").title())
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_final_metrics_bar_chart(results, save_path):
    """Plots a bar chart comparing the final F1 score and accuracy."""
    fig, ax = plt.subplots(figsize=(14, 9))
    
    names = list(results.keys())
    final_f1s = [results[name]['final_val_f1'] for name in names]
    final_accuracies = [results[name]['final_val_accuracy'] for name in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, final_f1s, width, label='F1 Score', color='skyblue')
    bars2 = ax.bar(x + width/2, final_accuracies, width, label='Accuracy', color='salmon')
    
    ax.set_xlabel('Experiment Variants', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Final Validation Metrics Comparison', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace('_', ' ') for name in names], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()