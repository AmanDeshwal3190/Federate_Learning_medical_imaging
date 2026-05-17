import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, Tuple, Optional

def dice_coefficient(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-7) -> float:
    """
    Compute Dice Similarity Coefficient.
    DSC = 2 * |X ∩ Y| / (|X| + |Y|)
    Used for brain tumor segmentation evaluation.
    """
    y_true_flat = y_true.flatten().astype(np.float32)
    y_pred_flat = y_pred.flatten().astype(np.float32)
    intersection = np.sum(y_true_flat * y_pred_flat)
    return (2.0 * intersection + smooth) / (np.sum(y_true_flat) + np.sum(y_pred_flat) + smooth)


def dice_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Dice loss = 1 - DSC. Used as training loss for segmentation."""
    return 1.0 - dice_coefficient(y_true, y_pred)


def multiclass_dice(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[int, float]:
    """Compute per-class Dice coefficients for multi-class segmentation."""
    dice_per_class = {}
    for cls in range(num_classes):
        true_cls = (y_true == cls).astype(np.float32)
        pred_cls = (y_pred == cls).astype(np.float32)
        dice_per_class[cls] = dice_coefficient(true_cls, pred_cls)
    return dice_per_class


def sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Sensitivity = TP / (TP + FN). Also known as Recall / True Positive Rate."""
    if len(np.unique(y_true)) > 2 or len(np.unique(y_pred)) > 2:
        return recall_score(y_true, y_pred, average='macro', zero_division=0)
    # Default binary assumption if <= 2 unique values
    # Try getting confusion matrix elements if it's strictly 0,1
    # Fallback to recall_score if values aren't 0 and 1
    if set(np.unique(y_true)).union(set(np.unique(y_pred))).issubset({0, 1}):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return recall_score(y_true, y_pred, average='macro', zero_division=0)


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Specificity = TN / (TN + FP). Also known as True Negative Rate."""
    if set(np.unique(y_true)).union(set(np.unique(y_pred))).issubset({0, 1}):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    # For multi-class or unexpected format, compute macro specificity
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[0] < 2:
        return 0.0
    spec_per_class = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - tp - fp - fn
        spec_per_class.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return np.mean(spec_per_class)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                         y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute all classification metrics used in the paper.
    
    Args:
        y_true: Ground truth labels (binary: 0 or 1)
        y_pred: Predicted labels (binary: 0 or 1) 
        y_prob: Predicted probabilities for ROC AUC (optional)
    
    Returns:
        Dictionary with all metric values
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": sensitivity(y_true, y_pred),
        "specificity": specificity(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average='macro', zero_division=0),
    }
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) > 2:
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            else:
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = 0.0
    return metrics


def compute_segmentation_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                   num_classes: int = 4) -> Dict[str, float]:
    """
    Compute segmentation-specific metrics for brain tumor segmentation.
    BraTS regions: Background(0), NCR/NET(1), ED(2), ET(4->3)
    """
    dice_scores = multiclass_dice(y_true, y_pred, num_classes)
    
    # Whole tumor (all non-zero classes)
    wt_true = (y_true > 0).astype(np.float32)
    wt_pred = (y_pred > 0).astype(np.float32)
    
    # Tumor core (classes 1 and 3, excluding edema)
    tc_true = np.isin(y_true, [1, 3]).astype(np.float32)
    tc_pred = np.isin(y_pred, [1, 3]).astype(np.float32)
    
    # Enhancing tumor (class 3 only)
    et_true = (y_true == 3).astype(np.float32)
    et_pred = (y_pred == 3).astype(np.float32)
    
    return {
        "dice_per_class": dice_scores,
        "dice_whole_tumor": dice_coefficient(wt_true, wt_pred),
        "dice_tumor_core": dice_coefficient(tc_true, tc_pred),
        "dice_enhancing_tumor": dice_coefficient(et_true, et_pred),
        "mean_dice": np.mean(list(dice_scores.values())),
    }


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray, 
                                  target_names: list = None) -> str:
    """Generate and return a formatted classification report."""
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
