"""Tests for Alzheimer's classification models."""
import pytest
import numpy as np
import tensorflow as tf
from models.alzheimer.vgg3d import build_vgg3d, build_vgg3d_small, build_conv_block
from models.alzheimer.trainer import AlzheimerTrainer
from models.alzheimer.transfer_learning import AlzheimerEnsemble

def test_vgg3d_build_adni():
    model = build_vgg3d(input_shape=(182, 218, 182, 1), num_classes=2)
    assert model is not None
    assert model.name == "VGG3D"

def test_vgg3d_build_oasis():
    model = build_vgg3d(input_shape=(176, 208, 176, 1), num_classes=2)
    assert model is not None

def test_vgg3d_output_shape():
    model = build_vgg3d(input_shape=(182, 218, 182, 1), num_classes=2)
    dummy_input = tf.random.normal((1, 182, 218, 182, 1))
    output = model(dummy_input)
    assert output.shape == (1, 2)

def test_vgg3d_output_probabilities():
    model = build_vgg3d(input_shape=(182, 218, 182, 1), num_classes=2)
    dummy_input = tf.random.normal((1, 182, 218, 182, 1))
    output = model(dummy_input)
    # output sums to 1 (softmax)
    assert np.isclose(tf.reduce_sum(output[0]).numpy(), 1.0)

def test_conv_block():
    inputs = tf.keras.layers.Input(shape=(32, 32, 32, 1))
    x = build_conv_block(inputs, num_conv_layers=2, filters=16, block_name="test_block")
    model = tf.keras.Model(inputs, x)
    dummy_input = tf.random.normal((1, 32, 32, 32, 1))
    output = model(dummy_input)
    # MaxPool(2,2,2) should halve dimensions -> 16, 16, 16
    assert output.shape == (1, 16, 16, 16, 16) 

def test_dropout_effect():
    model = build_vgg3d_small(input_shape=(32, 32, 32, 1)) # Use small shape for fast test
    dummy_input = tf.random.normal((1, 32, 32, 32, 1))
    out_train = model(dummy_input, training=True)
    out_infer = model(dummy_input, training=False)
    # Predictions should be different due to dropout in training
    assert not np.allclose(out_train.numpy(), out_infer.numpy())

def test_parameter_count():
    model = build_vgg3d(input_shape=(182, 218, 182, 1), num_classes=2)
    params = model.count_params()
    # Expect ~5-10M parameters (could be slightly higher or lower depending on exact dense input)
    assert params > 1000000

def test_compile():
    model = build_vgg3d_small(input_shape=(32, 32, 32, 1))
    trainer = AlzheimerTrainer()
    compiled_model = trainer.compile_model(model)
    assert compiled_model.optimizer is not None
    assert compiled_model.loss == "binary_crossentropy"

def test_forward_pass_gradient():
    model = build_vgg3d_small(input_shape=(32, 32, 32, 1))
    trainer = AlzheimerTrainer()
    model = trainer.compile_model(model)
    
    dummy_input = tf.random.normal((1, 32, 32, 32, 1))
    dummy_target = tf.constant([[1.0, 0.0]])
    
    with tf.GradientTape() as tape:
        predictions = model(dummy_input, training=True)
        loss = tf.keras.losses.binary_crossentropy(dummy_target, predictions)
        
    gradients = tape.gradient(loss, model.trainable_variables)
    assert any(g is not None for g in gradients)

def test_small_model():
    model = build_vgg3d_small(input_shape=(96, 112, 96, 1), num_classes=2)
    dummy_input = tf.random.normal((1, 96, 112, 96, 1))
    output = model(dummy_input)
    assert output.shape == (1, 2)

def test_ensemble_init():
    ensemble = AlzheimerEnsemble(num_models=3, dataset_type="adni")
    assert len(ensemble.models) == 3

def test_ensemble_predict_shape():
    ensemble = AlzheimerEnsemble(num_models=2, dataset_type="adni")
    # monkey patch models with smaller ones for test speed
    for i in range(2):
        ensemble.models[i] = build_vgg3d_small(input_shape=(32, 32, 32, 1))
        
    dummy_input = np.random.normal(size=(2, 32, 32, 32, 1))
    
    out_avg = ensemble.predict_averaged(dummy_input)
    assert out_avg.shape == (2, 2)
    
    out_weight = ensemble.predict_weighted(dummy_input)
    assert out_weight.shape == (2, 2)
    
    out_maj = ensemble.predict_majority_vote(dummy_input)
    assert out_maj.shape == (2, 2)

def test_trainer_callbacks():
    trainer = AlzheimerTrainer()
    callbacks = trainer.get_callbacks(fold_num=0)
    # Check that EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger are there
    callback_types = [type(c) for c in callbacks]
    assert tf.keras.callbacks.EarlyStopping in callback_types
    assert tf.keras.callbacks.ReduceLROnPlateau in callback_types
    assert tf.keras.callbacks.ModelCheckpoint in callback_types
    assert tf.keras.callbacks.CSVLogger in callback_types
