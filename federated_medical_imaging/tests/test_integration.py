"""
End-to-end integration tests for the federated medical imaging pipeline.
Uses synthetic data for testing.
"""
import pytest
import numpy as np
import tensorflow as tf
import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestImports:
    """Test that all modules can be imported."""
    
    def test_import_config(self):
        from utils.config_loader import ConfigLoader
    
    def test_import_logger(self):
        from utils.logger import get_logger, TrainingLogger
    
    def test_import_metrics(self):
        from utils.metrics import dice_coefficient, compute_all_metrics
    
    def test_import_brain_tumor_preprocessing(self):
        from data.brain_tumor.preprocessing import BrainTumorPreprocessor
    
    def test_import_alzheimer_preprocessing(self):
        from data.alzheimer.preprocessing import AlzheimerPreprocessor
    
    def test_import_unet3d(self):
        from models.brain_tumor.unet3d import build_unet3d
    
    def test_import_vgg3d(self):
        from models.alzheimer.vgg3d import build_vgg3d_small
    
    def test_import_fl_server(self):
        from federated.server import FederatedServer
    
    def test_import_fl_client(self):
        from federated.client import MedicalImagingClient
    
    def test_import_dashboard(self):
        from dashboard.app import create_app

class TestModelBuilds:
    """Test that models build correctly."""
    
    def test_unet3d_builds(self):
        from models.brain_tumor.unet3d import build_unet3d
        model = build_unet3d(input_shape=(32, 32, 32, 4), num_classes=4,
                              num_filters=[8, 16, 32, 64, 128])
        assert model is not None
        assert model.count_params() > 0
    
    def test_unet3d_forward_pass(self):
        from models.brain_tumor.unet3d import build_unet3d
        model = build_unet3d(input_shape=(32, 32, 32, 4), num_classes=4,
                              num_filters=[8, 16, 32, 64, 128])
        dummy = tf.random.normal((1, 32, 32, 32, 4))
        output = model(dummy)
        assert output.shape == (1, 32, 32, 32, 4)
    
    def test_vgg3d_builds(self):
        from models.alzheimer.vgg3d import build_vgg3d_small
        model = build_vgg3d_small(input_shape=(32, 40, 32, 1), num_classes=2)
        assert model is not None
    
    def test_vgg3d_forward_pass(self):
        from models.alzheimer.vgg3d import build_vgg3d_small
        model = build_vgg3d_small(input_shape=(32, 40, 32, 1), num_classes=2)
        dummy = tf.random.normal((1, 32, 40, 32, 1))
        output = model(dummy)
        assert output.shape == (1, 2)
        # Check softmax output
        assert abs(tf.reduce_sum(output).numpy() - 1.0) < 0.01

class TestMetrics:
    """Test metric computations."""
    
    def test_dice_perfect(self):
        from utils.metrics import dice_coefficient
        y = np.ones((10, 10)).astype(np.float32)
        assert float(dice_coefficient(y, y)) > 0.99
    
    def test_dice_no_overlap(self):
        from utils.metrics import dice_coefficient
        y1 = np.zeros((10, 10)).astype(np.float32)
        y2 = np.ones((10, 10)).astype(np.float32)
        assert float(dice_coefficient(y1, y2)) < 0.01
    
    def test_all_metrics(self):
        from utils.metrics import compute_all_metrics
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        metrics = compute_all_metrics(y_true, y_pred)
        assert "accuracy" in metrics
        assert "sensitivity" in metrics
        assert "specificity" in metrics
        assert "f1_score" in metrics
        assert 0 <= metrics["accuracy"] <= 1

class TestEnsembleClassifier:
    """Test ensemble classifier."""
    
    def test_fit_predict(self):
        from models.brain_tumor.ensemble_classifier import BrainTumorEnsembleClassifier
        clf = BrainTumorEnsembleClassifier()
        X = np.random.randn(100, 64).astype(np.float32)
        y = np.random.randint(0, 3, 100)
        clf.fit(X[:80], y[:80])
        preds = clf.predict(X[80:])
        assert len(preds) == 20
        assert set(preds).issubset({0, 1, 2})

class TestFlowerIntegration:
    """Test Flower FL components."""
    
    def test_weight_serialization(self):
        import flwr as fl
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(4, input_shape=(3,)),
            tf.keras.layers.Dense(2)
        ])
        weights = model.get_weights()
        params = fl.common.ndarrays_to_parameters(weights)
        recovered = fl.common.parameters_to_ndarrays(params)
        for w, r in zip(weights, recovered):
            assert np.allclose(w, r)

class TestDashboard:
    """Test Flask dashboard."""
    
    def test_app_creates(self):
        from dashboard.app import create_app
        app, socketio = create_app()
        assert app is not None
    
    def test_index_route(self):
        from dashboard.app import create_app
        app, socketio = create_app()
        with app.test_client() as client:
            response = client.get('/')
            assert response.status_code == 200
    
    def test_status_api(self):
        from dashboard.app import create_app
        app, socketio = create_app()
        with app.test_client() as client:
            response = client.get('/api/status')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert "training_active" in data

class TestPipelineOrchestrator:
    """Test the master orchestrator."""
    
    def test_init(self):
        from scripts.run_full_pipeline import PipelineOrchestrator
        orch = PipelineOrchestrator(disease="brain_tumor")
        assert orch.disease == "brain_tumor"
    
    def test_progress_save_load(self):
        from scripts.run_full_pipeline import PipelineOrchestrator
        orch = PipelineOrchestrator()
        orch.progress["stages_completed"].append("preprocessing")
        orch.save_progress()
        loaded = orch.load_progress()
        assert "preprocessing" in loaded.get("stages_completed", [])
