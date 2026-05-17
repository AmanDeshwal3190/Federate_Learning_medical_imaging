"""
Training orchestrator for the brain tumor segmentation and classification pipeline.
"""
import tensorflow as tf
import numpy as np
import os
import time
from typing import Dict, Optional, Tuple
from utils.config_loader import ConfigLoader
from utils.logger import get_logger, TrainingLogger
from utils.metrics import compute_segmentation_metrics, compute_all_metrics
from utils.common import set_seed, get_device, save_json, ensure_dir

# Assuming visualization functions exist or can be skipped safely safely
try:
    from utils.visualization import plot_training_curves, plot_segmentation_overlay
except ImportError:
    plot_training_curves = lambda *args, **kwargs: None
    plot_segmentation_overlay = lambda *args, **kwargs: None

from models.brain_tumor.unet3d import build_unet3d, combined_loss
from models.brain_tumor.densenet_extractor import build_densenet_extractor, extract_features
from models.brain_tumor.ensemble_classifier import BrainTumorEnsembleClassifier

logger = get_logger("brain_tumor_trainer")


class BrainTumorTrainer:
    """
    Complete training pipeline for brain tumor detection.
    
    Pipeline:
    1. Train 3D U-Net for segmentation
    2. Extract features from segmented regions using DenseNet
    3. Train ensemble classifier on extracted features
    """
    
    def __init__(self, config_path: str = "config/brain_tumor_config.yaml"):
        try:
            self.config = ConfigLoader.load(config_path)
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
            class MockConfig:
                model = type('obj', (object,), {'segmentation': type('obj', (object,), {'input_shape': (128,128,128,4), 'num_filters': [32, 64, 128, 256, 512], 'dropout_rate': 0.3})})
                dataset = type('obj', (object,), {'num_classes': 4})
                training = type('obj', (object,), {'learning_rate': 0.001, 'batch_size': 4, 'epochs': 60})
                paths = type('obj', (object,), {'checkpoint_dir': 'results/checkpoints/brain_tumor', 'results_dir': 'results/logs/brain_tumor'})
            self.config = MockConfig()
            
        self.training_logger = TrainingLogger("brain_tumor_training")
        set_seed(42)
        device = get_device()
        logger.info(f"Training device: {device}")
    
    def train_segmentation(self, train_dataset: tf.data.Dataset,
                             val_dataset: tf.data.Dataset) -> Tuple[tf.keras.Model, dict]:
        """
        Train the 3D U-Net segmentation model.
        
        Training procedure:
        1. Build model with config parameters
        2. Compile with Adam(lr=0.001), combined loss (Dice + CE)
        3. Callbacks: EarlyStopping(patience=10), ReduceLROnPlateau(patience=5, factor=0.5),
                      ModelCheckpoint(save_best_only=True)
        4. Train for 60 epochs with batch_size=4
        5. Return model and training history
        
        Args:
            train_dataset: tf.data.Dataset for training
            val_dataset: tf.data.Dataset for validation
        Returns:
            Tuple of (trained_model, history_dict)
        """
        logger.info("Building 3D U-Net for segmentation...")
        model = build_unet3d(
            input_shape=tuple(self.config.model.segmentation.input_shape),
            num_classes=self.config.dataset.num_classes,
            num_filters=self.config.model.segmentation.num_filters,
            dropout_rate=self.config.model.segmentation.dropout_rate
        )
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.training.learning_rate)
        model.compile(optimizer=optimizer, loss=combined_loss, metrics=['accuracy'])
        
        ckpt_path = os.path.join(self.config.paths.checkpoint_dir, "best_unet3d.keras")
        ensure_dir(os.path.dirname(ckpt_path))
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, monitor='val_loss'),
            tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, save_best_only=True, monitor='val_loss')
        ]
        
        logger.info(f"Starting segmentation training for {self.config.training.epochs} epochs.")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.config.training.epochs,
            callbacks=callbacks
        )
        
        plot_training_curves(history.history, os.path.join(self.config.paths.results_dir, "unet_training.png"))
        return model, history.history
    
    def train_classification(self, train_features: np.ndarray, train_labels: np.ndarray,
                               val_features: np.ndarray, val_labels: np.ndarray
                               ) -> Tuple[BrainTumorEnsembleClassifier, Dict[str, float]]:
        """
        Train the ensemble classifier on DenseNet features.
        
        1. Initialize ensemble with config parameters
        2. Train on training features
        3. Evaluate on validation features
        4. Save model checkpoint
        
        Returns:
            Tuple of (trained_classifier, validation_metrics)
        """
        logger.info("Initializing BrainTumorEnsembleClassifier...")
        clf = BrainTumorEnsembleClassifier()
        
        logger.info("Training ensemble classifier...")
        clf.fit(train_features, train_labels)
        
        logger.info("Evaluating ensemble classifier...")
        val_metrics = clf.evaluate(val_features, val_labels)
        logger.info(f"Validation Metrics: {val_metrics}")
        
        ckpt_path = os.path.join(self.config.paths.checkpoint_dir, "best_ensemble.pkl")
        clf.save(ckpt_path)
        
        return clf, val_metrics
    
    def run_full_pipeline(self, train_seg_data: tf.data.Dataset,
                           val_seg_data: tf.data.Dataset,
                           train_cls_data: tf.data.Dataset,
                           val_cls_data: tf.data.Dataset) -> Dict[str, any]:
        """
        Run the complete brain tumor detection pipeline:
        
        1. Train segmentation model (U-Net 3D)
        2. Build DenseNet feature extractor
        3. Extract features from classified regions
        4. Train ensemble classifier
        5. Evaluate everything
        6. Save all results, plots, and checkpoints
        
        Returns:
            Dict with all metrics and file paths to saved artifacts
        """
        results = {}
        
        # 1. Train Segmentation
        logger.info("=== STAGE 1: SEGMENTATION ===")
        seg_model, seg_history = self.train_segmentation(train_seg_data, val_seg_data)
        results['segmentation_history'] = seg_history
        
        # 2. Build Extractor
        logger.info("=== STAGE 2: FEATURE EXTRACTION ===")
        extractor = build_densenet_extractor()
        
        # 3. Extract Features
        train_features, train_labels = extract_features(extractor, train_cls_data)
        val_features, val_labels = extract_features(extractor, val_cls_data)
        logger.info(f"Extracted features. Train shape: {train_features.shape}, Val shape: {val_features.shape}")
        
        # 4. Train Classifier
        logger.info("=== STAGE 3: CLASSIFICATION ===")
        clf, cls_metrics = self.train_classification(
            train_features, train_labels, val_features, val_labels
        )
        results['classification_metrics'] = cls_metrics
        
        # 5. Get individual scores
        indiv_scores = clf.get_individual_scores(val_features, val_labels)
        results['individual_classifier_metrics'] = indiv_scores
        
        ensure_dir(self.config.paths.results_dir)
        save_json(results, os.path.join(self.config.paths.results_dir, "pipeline_results.json"))
        
        return results
    
    def get_model_for_federated(self) -> tf.keras.Model:
        """
        Get the segmentation model ready for federated learning.
        Returns compiled model that Flower client can use.
        """
        model = build_unet3d(
            input_shape=tuple(self.config.model.segmentation.input_shape),
            num_classes=self.config.dataset.num_classes,
            num_filters=self.config.model.segmentation.num_filters,
            dropout_rate=self.config.model.segmentation.dropout_rate
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.training.learning_rate),
            loss=combined_loss,
            metrics=['accuracy']
        )
        return model
