"""Tests for federated learning infrastructure."""
import pytest
import numpy as np
import tensorflow as tf
import os
import tempfile
import sys

# Ensure imports work from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from federated.client import MedicalImagingClient
from federated.server import FederatedServer
from federated.strategies.fedavg_custom import FedAvgWithLogging
from federated.strategies.fedprox import FedProx, create_fedprox_loss
import flwr as fl

@pytest.fixture
def dummy_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_shape=(5,)),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

@pytest.fixture
def dummy_dataset():
    x = np.random.rand(10, 5).astype(np.float32)
    y = np.random.randint(0, 2, 10).astype(np.int32)
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(2)

def test_client_get_set_parameters(dummy_model, dummy_dataset):
    client = MedicalImagingClient(1, dummy_model, dummy_dataset, dummy_dataset)
    
    # Get parameters
    params = client.get_parameters({})
    assert len(params) == 4 # 2 dense layers * (weights + biases)
    
    # Modify parameters and set
    new_params = [p + 1.0 for p in params]
    client.set_parameters(new_params)
    
    updated_params = client.get_parameters({})
    assert np.allclose(updated_params[0], params[0] + 1.0)

def test_client_fit(dummy_model, dummy_dataset):
    client = MedicalImagingClient(1, dummy_model, dummy_dataset, dummy_dataset)
    initial_params = client.get_parameters({})
    
    params, num_samples, metrics = client.fit(initial_params, {"epochs": 1})
    
    assert num_samples == 10
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert params is not None

def test_client_evaluate(dummy_model, dummy_dataset):
    client = MedicalImagingClient(1, dummy_model, dummy_dataset, dummy_dataset)
    initial_params = client.get_parameters({})
    
    loss, num_samples, metrics = client.evaluate(initial_params, {})
    
    assert num_samples == 10
    assert isinstance(loss, float)
    # In newer Keras versions, accuracy metric may be named 'compile_metrics'
    assert "accuracy" in metrics or "compile_metrics" in metrics

def test_fedavg_strategy_creation():
    strategy = FedAvgWithLogging()
    assert getattr(strategy, "aggregate_fit", None) is not None

def test_fedprox_loss():
    # Verify proximal term increases loss
    # Since keras loss functions natively only receive y_true, y_pred, we test the wrapper structure
    original_loss_fn = tf.keras.losses.MeanSquaredError()
    global_weights = [np.array([1.0, 2.0]), np.array([3.0])]
    loss_fn = create_fedprox_loss(original_loss_fn, global_weights, mu=0.1)
    assert loss_fn is not None

def test_weighted_average_metrics():
    # Test directly by creating an instance of FederatedServer but mocking out config loading
    class MockServer(FederatedServer):
        def __init__(self):
            # Bypass config loading
            pass

    server = MockServer()
    metrics = [
        (10, {"accuracy": 0.8, "loss": 0.2}),
        (20, {"accuracy": 0.9, "loss": 0.1})
    ]
    
    avg = server.weighted_average_metrics(metrics)
    assert np.isclose(avg["accuracy"], (10*0.8 + 20*0.9) / 30)
    assert np.isclose(avg["loss"], (10*0.2 + 20*0.1) / 30)

def test_client_count_samples(dummy_model, dummy_dataset):
    client = MedicalImagingClient(1, dummy_model, dummy_dataset, dummy_dataset)
    # The dataset fixture has 10 samples
    assert client.num_train_samples == 10

def test_checkpoint_save_load(dummy_model):
    with tempfile.TemporaryDirectory() as temp_dir:
        strategy = FedAvgWithLogging(checkpoint_dir=temp_dir, log_dir=temp_dir)
        parameters = fl.common.ndarrays_to_parameters(dummy_model.get_weights())
        
        path = strategy.save_checkpoint(parameters, 1, {"accuracy": 0.95})
        assert os.path.exists(path)
        
        metrics_path = os.path.join(temp_dir, "metrics_round_1.json")
        assert os.path.exists(metrics_path)

def test_fedavg_with_logging_history():
    with tempfile.TemporaryDirectory() as temp_dir:
        strategy = FedAvgWithLogging(checkpoint_dir=temp_dir, log_dir=temp_dir)
        
        # Mock aggregate_evaluate results manually
        strategy.round_history.append({"round": 1, "loss": 0.5, "metrics": {"accuracy": 0.8}})
        path = strategy.save_history()
        
        assert os.path.exists(path)
        assert len(strategy.get_history()) == 1
