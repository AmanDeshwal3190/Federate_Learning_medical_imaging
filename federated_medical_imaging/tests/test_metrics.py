import pytest
import numpy as np
from utils.metrics import (
    dice_coefficient, sensitivity, specificity, 
    compute_all_metrics, compute_segmentation_metrics
)

def test_dice_coefficient():
    y_true = np.ones(10)
    y_pred = np.ones(10)
    assert np.isclose(dice_coefficient(y_true, y_pred), 1.0)
    
    y_pred_zero = np.zeros(10)
    assert np.isclose(dice_coefficient(y_true, y_pred_zero), 0.0)

def test_sensitivity_specificity():
    # y_true = [1, 1, 0, 0]
    # y_pred = [1, 0, 0, 0] -> TP=1, FN=1, TN=2, FP=0
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0])
    
    sens = sensitivity(y_true, y_pred)
    spec = specificity(y_true, y_pred)
    
    assert np.isclose(sens, 0.5)  # 1 / (1 + 1)
    assert np.isclose(spec, 1.0)  # 2 / (2 + 0)

def test_compute_all_metrics():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])
    y_prob = np.array([0.9, 0.1, 0.4, 0.2])
    
    metrics = compute_all_metrics(y_true, y_pred, y_prob)
    
    expected_keys = ["accuracy", "sensitivity", "specificity", "precision", "recall", "f1_score", "roc_auc"]
    for key in expected_keys:
        assert key in metrics

def test_compute_segmentation_metrics():
    # Synthetic mask with 4 classes
    y_true = np.array([[0, 1], [2, 3]])
    y_pred = np.array([[0, 1], [2, 2]])
    
    metrics = compute_segmentation_metrics(y_true, y_pred, num_classes=4)
    
    assert "dice_per_class" in metrics
    assert "dice_whole_tumor" in metrics
    assert "dice_tumor_core" in metrics
    assert "dice_enhancing_tumor" in metrics
    assert "mean_dice" in metrics
