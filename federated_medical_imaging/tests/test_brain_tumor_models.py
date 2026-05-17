"""Tests for brain tumor models."""
import pytest
import numpy as np
import tensorflow as tf

from models.brain_tumor.unet3d import build_unet3d, dice_loss_tf, combined_loss, SqueezeExcitation3D
from models.brain_tumor.densenet_extractor import build_densenet_extractor, InceptionMultiscaleBlock, SpatialAttention
from models.brain_tumor.ensemble_classifier import BrainTumorEnsembleClassifier


def test_unet3d_build():
    model = build_unet3d(input_shape=(32, 32, 32, 4), num_classes=4, num_filters=[16, 32])
    assert model is not None
    assert isinstance(model, tf.keras.Model)

def test_unet3d_output_shape():
    model = build_unet3d(input_shape=(32, 32, 32, 4), num_classes=4, num_filters=[8, 16])
    dummy_input = tf.random.normal((1, 32, 32, 32, 4))
    output = model(dummy_input)
    assert output.shape == (1, 32, 32, 32, 4)

def test_unet3d_parameter_count():
    model = build_unet3d(input_shape=(64, 64, 64, 4), num_classes=4, num_filters=[16, 32, 64])
    assert model.count_params() > 100000

def test_dice_loss_perfect():
    # Perfect overlap
    y_true = np.zeros((1, 32, 32, 32, 4))
    y_true[..., 0] = 1 # all background
    y_pred = y_true.copy()
    
    loss = dice_loss_tf(tf.constant(y_true, dtype=tf.float32), tf.constant(y_pred, dtype=tf.float32))
    assert np.isclose(loss.numpy(), 0.0, atol=1e-5)

def test_dice_loss_no_overlap():
    y_true = np.zeros((1, 32, 32, 32, 2))
    y_true[..., 0] = 1 # class 0
    y_pred = np.zeros((1, 32, 32, 32, 2))
    y_pred[..., 1] = 1 # class 1
    
    loss = dice_loss_tf(tf.constant(y_true, dtype=tf.float32), tf.constant(y_pred, dtype=tf.float32))
    # No overlap -> Intersection is 0 -> Dice is 0 -> Loss is 1
    assert np.isclose(loss.numpy(), 1.0, atol=1e-5)

def test_se_block():
    block = SqueezeExcitation3D(filters=16)
    dummy_input = tf.random.normal((2, 16, 16, 16, 16))
    output = block(dummy_input)
    assert output.shape == (2, 16, 16, 16, 16)

def test_densenet_extractor_shape():
    model = build_densenet_extractor(input_shape=(64, 64, 3), num_features=256)
    dummy_input = tf.random.normal((2, 64, 64, 3))
    output = model(dummy_input)
    assert output.shape == (2, 256)

def test_ensemble_fit_predict():
    clf = BrainTumorEnsembleClassifier()
    X = np.random.randn(50, 100)
    y = np.random.randint(0, 3, 50)
    
    clf.fit(X, y)
    X_test = np.random.randn(10, 100)
    preds = clf.predict(X_test)
    assert preds.shape == (10,)

def test_ensemble_cross_validate():
    clf = BrainTumorEnsembleClassifier()
    X = np.random.randn(50, 100)
    y = np.random.randint(0, 3, 50)
    
    metrics = clf.cross_validate(X, y, n_folds=3)
    assert 'accuracy' in metrics
    assert len(metrics['accuracy']) == 3

def test_inception_block_shape():
    block = InceptionMultiscaleBlock(filters_1x1=16, filters_3x3=32, filters_5x5=8, filters_pool=8)
    dummy_input = tf.random.normal((2, 32, 32, 16))
    output = block(dummy_input)
    # output channels = 16 + 32 + 8 + 8 = 64
    assert output.shape == (2, 32, 32, 64)

def test_spatial_attention_shape():
    block = SpatialAttention()
    dummy_input = tf.random.normal((2, 32, 32, 16))
    output = block(dummy_input)
    assert output.shape == (2, 32, 32, 16)
