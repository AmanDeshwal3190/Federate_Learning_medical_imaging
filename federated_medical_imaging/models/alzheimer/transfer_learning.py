"""
Transfer learning approach for Alzheimer's classification.
Uses pretrained 3D models and ensemble methods for improved accuracy.

From the paper:
- Transfer learning from VGGNet-16, GoogLeNet, ResNet-152 outperforms training from scratch
- Deep ensemble learning improves accuracy by 4% over individual models
- Best result: 96.88% accuracy, 100% sensitivity, 94.12% specificity
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from typing import List, Dict, Tuple, Optional
from models.alzheimer.vgg3d import build_vgg3d
from utils.logger import get_logger

logger = get_logger("alzheimer_transfer_learning")


class AlzheimerEnsemble:
    """
    Deep ensemble for Alzheimer's classification.
    Trains multiple models with different initializations and averages predictions.
    
    Ensemble methods compared in paper:
    1. Simple averaging
    2. Weighted averaging (based on validation accuracy)
    3. Majority voting
    4. Stacking (meta-learner)
    
    Result: Deep ensemble improves accuracy by ~4% over single model.
    """
    
    def __init__(self, num_models: int = 5, dataset_type: str = "adni"):
        """
        Initialize ensemble with num_models VGG3D instances.
        Each model uses a different random seed for initialization.
        """
        self.num_models = num_models
        self.dataset_type = dataset_type
        
        # We need to load config to get input shapes
        from utils.config_loader import ConfigLoader
        config = ConfigLoader.load("config/alzheimer_config.yaml")
        
        if dataset_type == "adni":
            self.input_shape = tuple(config.model.input_shape_adni)
        else:
            self.input_shape = tuple(config.model.input_shape_oasis)
            
        self.models = []
        for i in range(num_models):
            # Try to force different initialization implicitly by state or by explicitly using a seed inside layer initializers.
            tf.random.set_seed(42 + i)
            model = build_vgg3d(input_shape=self.input_shape, num_classes=2)
            
            # Compile each model
            optimizer = tf.keras.optimizers.Adam(learning_rate=config.training.learning_rate)
            model.compile(
                optimizer=optimizer,
                loss=config.training.loss,
                metrics=['accuracy']
            )
            self.models.append(model)
            
        self.model_weights = np.ones(num_models) / num_models # For weighted average

    
    def train_ensemble(self, train_dataset: tf.data.Dataset,
                        val_dataset: tf.data.Dataset,
                        epochs: int = 200) -> List[Dict]:
        """
        Train all models in the ensemble independently.
        Each model trains on the same data but with different initialization.
        """
        logger.info(f"Training ensemble of {self.num_models} models")
        all_histories = []
        
        for i, model in enumerate(self.models):
            logger.info(f"Training model {i+1}/{self.num_models}")
            
            # Use simple early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=20, restore_best_weights=True, verbose=1
            )
            
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1
            )
            
            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            all_histories.append(history.history)
            
            # Update weights based on validation accuracy
            val_acc = max(history.history['val_accuracy'])
            self.model_weights[i] = val_acc
            
        # Normalize weights
        sum_weights = np.sum(self.model_weights)
        if sum_weights > 0:
            self.model_weights = self.model_weights / sum_weights
            
        return all_histories

    def _get_all_predictions(self, x: np.ndarray) -> np.ndarray:
        """Helper to get predictions from all models. Returns shape (num_models, batch_size, num_classes)"""
        preds = []
        for model in self.models:
            # Handle tf.data.Dataset or np.ndarray
            if isinstance(x, tf.data.Dataset):
                p = model.predict(x, verbose=0)
            else:
                p = model.predict(x, verbose=0)
            preds.append(p)
        return np.array(preds)

    def predict_averaged(self, x: np.ndarray) -> np.ndarray:
        """Average predictions from all models."""
        all_preds = self._get_all_predictions(x)
        return np.mean(all_preds, axis=0)
    
    def predict_weighted(self, x: np.ndarray) -> np.ndarray:
        """Weighted average predictions based on validation accuracy."""
        all_preds = self._get_all_predictions(x)
        weighted_preds = np.zeros_like(all_preds[0])
        for i, pred in enumerate(all_preds):
            weighted_preds += pred * self.model_weights[i]
        return weighted_preds
    
    def predict_majority_vote(self, x: np.ndarray) -> np.ndarray:
        """Majority voting across all models."""
        all_preds = self._get_all_predictions(x)
        # Convert to class predictions
        all_classes = np.argmax(all_preds, axis=-1)
        # all_classes shape: (num_models, batch)
        # Find mode along the first axis
        num_samples = all_classes.shape[1]
        final_preds = np.zeros(num_samples, dtype=int)
        
        for i in range(num_samples):
            counts = np.bincount(all_classes[:, i])
            final_preds[i] = np.argmax(counts)
            
        # Convert back to one-hot structure for consistency
        num_classes = all_preds.shape[-1]
        one_hot = np.zeros((num_samples, num_classes))
        one_hot[np.arange(num_samples), final_preds] = 1.0
        return one_hot
    
    def evaluate_all_methods(self, test_dataset: tf.data.Dataset) -> Dict[str, Dict[str, float]]:
        """
        Compare all ensemble methods on test data.
        Returns metrics for each method.
        """
        from utils.metrics import compute_all_metrics
        logger.info("Evaluating all ensemble methods")
        
        y_true = []
        # Pre-fetch all inputs to a list to ensure consistent evaluation for non-deterministic iterators
        x_list = []
        for images, labels in test_dataset:
            y_true.extend(labels.numpy())
            x_list.append(images.numpy())
            
        # if x_list is empty return empty
        if not x_list:
            return {}
            
        x_all = np.concatenate(x_list, axis=0)
        y_true = np.array(y_true)
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true_class = np.argmax(y_true, axis=1)
        else:
            y_true_class = y_true

        methods = {
            "averaged": self.predict_averaged(x_all),
            "weighted": self.predict_weighted(x_all),
            "majority": self.predict_majority_vote(x_all)
        }
        
        results = {}
        for method_name, preds_prob in methods.items():
            preds_class = np.argmax(preds_prob, axis=1)
            metrics = compute_all_metrics(y_true_class, preds_class)
            results[method_name] = metrics
            
        return results


def create_pretrained_3d_model(base_architecture: str = "vgg3d",
                                 input_shape: Tuple = (182, 218, 182, 1),
                                 pretrained_weights_path: str = None) -> Model:
    """
    Create a 3D model with optional pretrained weight loading.
    
    If pretrained_weights_path is provided, load weights (partial match allowed).
    Freeze early layers and fine-tune later layers.
    
    Freeze strategy:
    - Freeze blocks 1-2 (low-level features)
    - Fine-tune blocks 3-4 and FC layers (high-level features)
    
    Args:
        base_architecture: Model type ("vgg3d")
        input_shape: Volume shape
        pretrained_weights_path: Path to .h5 weights file
    Returns:
        Model with selective freezing applied
    """
    if base_architecture != "vgg3d":
        raise ValueError(f"Unsupported base architecture: {base_architecture}")
        
    model = build_vgg3d(input_shape=input_shape, num_classes=2)
    
    if pretrained_weights_path and tf.io.gfile.exists(pretrained_weights_path):
        logger.info(f"Loading pretrained weights from {pretrained_weights_path}")
        model.load_weights(pretrained_weights_path, by_name=True, skip_mismatch=True)
        
        # Apply freezing strategy
        logger.info("Freezing blocks 1 and 2 for transfer learning")
        for layer in model.layers:
            if layer.name.startswith("block1") or layer.name.startswith("block2"):
                layer.trainable = False
                
    return model
