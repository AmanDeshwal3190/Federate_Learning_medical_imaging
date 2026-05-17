import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Set global style
sns.set_theme(style="darkgrid")
COLORS = ['#2196F3', '#FF5722', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4']
PLOT_DIR = "results/plots"


def _save_plot(fig: plt.Figure, name: str, plot_dir: str = PLOT_DIR) -> str:
    """Save figure to plots directory with timestamp."""
    os.makedirs(plot_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(plot_dir, f"{name}_{timestamp}.png")
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return filepath


def plot_training_curves(history: dict, title: str = "Training Curves") -> str:
    """Plot training and validation loss/accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], color=COLORS[0], label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], color=COLORS[1], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{title} - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy (if available)
    if 'train_accuracy' in history and 'val_accuracy' in history:
        axes[1].plot(epochs, history['train_accuracy'], color=COLORS[2], label='Train Accuracy', linewidth=2)
        axes[1].plot(epochs, history['val_accuracy'], color=COLORS[3], label='Val Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title(f'{title} - Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'Accuracy data not available', ha='center', va='center', transform=axes[1].transAxes)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    return _save_plot(fig, "training_curves")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                           class_names: List[str], title: str = "Confusion Matrix") -> str:
    """Plot confusion matrix as a heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title, fontweight='bold')
    fig.tight_layout()
    return _save_plot(fig, "confusion_matrix")


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, title: str = "ROC Curve") -> str:
    """Plot ROC curve with AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color=COLORS[0], linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_plot(fig, "roc_curve")


def plot_segmentation_overlay(image_slice: np.ndarray, mask_true: np.ndarray,
                                mask_pred: np.ndarray, title: str = "Segmentation") -> str:
    """Plot MRI slice with ground truth and predicted segmentation overlays."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image_slice, cmap='gray')
    axes[0].set_title('MRI Slice')
    axes[0].axis('off')
    
    axes[1].imshow(image_slice, cmap='gray')
    axes[1].imshow(mask_true, alpha=0.4, cmap='jet')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(image_slice, cmap='gray')
    axes[2].imshow(mask_pred, alpha=0.4, cmap='jet')
    axes[2].set_title('Predicted')
    axes[2].axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    return _save_plot(fig, "segmentation_overlay")


def plot_federated_rounds(round_metrics: List[Dict[str, float]], 
                           metric_name: str = "accuracy",
                           title: str = "Federated Learning Progress") -> str:
    """Plot metric progression across FL rounds."""
    rounds = range(1, len(round_metrics) + 1)
    values = [m.get(metric_name, 0.0) for m in round_metrics]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rounds, values, color=COLORS[0], marker='o', linewidth=2, markersize=6)
    ax.set_xlabel('Federation Round')
    ax.set_ylabel(metric_name.replace('_', ' ').title())
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_plot(fig, "federated_rounds")


def plot_comparison_bar(federated_metrics: Dict[str, float], 
                         centralized_metrics: Dict[str, float],
                         title: str = "Federated vs Centralized") -> str:
    """Plot side-by-side comparison of federated vs centralized metrics."""
    metrics = list(federated_metrics.keys())
    fed_vals = [federated_metrics[m] for m in metrics]
    cent_vals = [centralized_metrics.get(m, 0.0) for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, fed_vals, width, label='Federated', color=COLORS[0], alpha=0.85)
    bars2 = ax.bar(x + width/2, cent_vals, width, label='Centralized', color=COLORS[1], alpha=0.85)
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Value')
    ax.set_title(title, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=30, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    
    fig.tight_layout()
    return _save_plot(fig, "comparison_bar")
