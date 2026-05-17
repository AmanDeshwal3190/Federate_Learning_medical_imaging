"""
Training pipeline for Alzheimer's Disease classification with 5-fold cross-validation.
"""
import tensorflow as tf
import numpy as np
import os
import time
import json
from typing import Dict, List, Tuple, Optional
from utils.config_loader import ConfigLoader
from utils.logger import get_logger, TrainingLogger
from utils.metrics import compute_all_metrics, sensitivity, specificity
from utils.common import set_seed, get_device, count_parameters, save_json, ensure_dir
from utils.visualization import (plot_training_curves, plot_confusion_matrix, 
                                  plot_roc_curve, plot_comparison_bar)
from models.alzheimer.vgg3d import build_vgg3d, build_vgg3d_small

logger = get_logger("alzheimer_trainer")


class AlzheimerTrainer:
    """
    Training orchestrator for Alzheimer's classification.
    Implements 5-fold cross-validation with proper subject-level splitting.
    """
    
    def __init__(self, config_path: str = "config/alzheimer_config.yaml"):
        self.config = ConfigLoader.load(config_path)
        self.training_logger = TrainingLogger("alzheimer_training")
        set_seed(42)
        self.device = get_device()
        logger.info(f"Training device: {self.device}")
        
        # Training parameters from paper
        self.batch_size = self.config.training.batch_size      # 2
        self.epochs = self.config.training.epochs               # 200
        self.learning_rate = self.config.training.learning_rate  # 0.0001
        self.num_folds = self.config.training.cv_folds          # 5
    
    def compile_model(self, model: tf.keras.Model) -> tf.keras.Model:
        """
        Compile model with paper-specified parameters.
        
        - Optimizer: Adam with learning_rate=0.0001
        - Loss: binary_crossentropy (L = -(y*log(ŷ) + (1-y)*log(1-ŷ)))
        - Metrics: accuracy
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=self.config.training.loss, # "binary_crossentropy"
            metrics=['accuracy']
        )
        return model
    
    def get_callbacks(self, fold_num: int) -> List[tf.keras.callbacks.Callback]:
        """
        Create training callbacks for a given fold.
        
        Callbacks:
        1. EarlyStopping: monitor='val_loss', patience=20, restore_best_weights=True
        2. ReduceLROnPlateau: monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7
        3. ModelCheckpoint: save best model for this fold
        4. CSVLogger: save per-epoch metrics to CSV
        
        Args:
            fold_num: Current fold number (0-4) for naming files
        Returns:
            List of Keras callbacks
        """
        checkpoint_dir = self.config.training.checkpoint_dir
        ensure_dir(checkpoint_dir)
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.training.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
        
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, f"best_model_fold_{fold_num}.h5"),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False, # Save complete model
            verbose=1
        )
        
        csv_logger = tf.keras.callbacks.CSVLogger(
            filename=os.path.join(checkpoint_dir, f"training_log_fold_{fold_num}.csv"),
            append=False
        )
        
        return [early_stopping, reduce_lr, model_checkpoint, csv_logger]
    
    def train_single_fold(self, fold_num: int, 
                            train_dataset: tf.data.Dataset,
                            val_dataset: tf.data.Dataset,
                            dataset_type: str = "adni") -> Dict[str, any]:
        """
        Train model for a single fold.
        """
        logger.info(f"Starting training for fold {fold_num} on {dataset_type} dataset.")
        
        if dataset_type == "adni":
            input_shape = tuple(self.config.model.input_shape_adni)
        else:
            input_shape = tuple(self.config.model.input_shape_oasis)
            
        model = build_vgg3d(input_shape=input_shape, num_classes=2, dropout_rate=self.config.model.dropout_rate)
        model = self.compile_model(model)
        
        if fold_num == 0:
            count_parameters(model)
            
        callbacks = self.get_callbacks(fold_num)
        
        start_time = time.time()
        
        history = model.fit(
            train_dataset,
            epochs=self.epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        best_epoch = np.argmin(history.history['val_loss']) + 1
        
        # Plot curves
        ensure_dir("results/logs/alzheimer_plots")
        plot_path = f"results/logs/alzheimer_plots/training_curves_fold_{fold_num}_{dataset_type}.png"
        plot_training_curves(history.history, plot_path)
        
        model_path = os.path.join(self.config.training.checkpoint_dir, f"best_model_fold_{fold_num}.h5")
        
        # Evaluate on val data to pass back val_metrics
        loss, acc = model.evaluate(val_dataset, verbose=0)
        
        val_metrics = {
            'val_loss': float(loss),
            'val_accuracy': float(acc)
        }
        
        result = {
            'fold': fold_num,
            'history': history.history,
            'val_metrics': val_metrics,
            'model_path': model_path,
            'best_epoch': int(best_epoch),
            'training_time': training_time
        }
        
        self.training_logger.log_metrics(val_metrics, step=fold_num)
        return result
    
    def run_cross_validation(self, fold_datasets: List[Tuple[tf.data.Dataset, tf.data.Dataset]],
                               dataset_type: str = "adni") -> Dict[str, any]:
        """
        Run complete 5-fold cross-validation.
        """
        logger.info(f"Starting 5-fold cross-validation for {dataset_type} dataset.")
        
        all_results = []
        val_accuracies = []
        
        for fold, (train_ds, val_ds) in enumerate(fold_datasets):
            fold_result = self.train_single_fold(fold, train_ds, val_ds, dataset_type)
            all_results.append(fold_result)
            val_accuracies.append(fold_result['val_metrics']['val_accuracy'])
            
        mean_acc = np.mean(val_accuracies)
        std_acc = np.std(val_accuracies)
        
        logger.info(f"Cross-validation completed. Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        
        # Determine target accuracy based on dataset
        target_acc = self.config.evaluation.target_accuracy_adni if dataset_type == "adni" else self.config.evaluation.target_accuracy_oasis
        logger.info(f"Target Accuracy for {dataset_type}: {target_acc:.4f}")
        
        summary = {
            'dataset_type': dataset_type,
            'num_folds': len(fold_datasets),
            'mean_accuracy': float(mean_acc),
            'std_accuracy': float(std_acc),
            'target_accuracy': float(target_acc),
            'folds': all_results
        }
        
        ensure_dir("results/logs")
        save_json(summary, f"results/logs/cv_summary_{dataset_type}.json")
        
        return summary
    
    def evaluate_model(self, model: tf.keras.Model, 
                        test_dataset: tf.data.Dataset) -> Dict[str, float]:
        """
        Comprehensive evaluation of a trained model.
        """
        from sklearn.metrics import confusion_matrix, roc_auc_score
        
        y_true = []
        y_pred_probs = []
        
        for images, labels in test_dataset:
            preds = model.predict(images, verbose=0)
            y_pred_probs.extend(preds)
            y_true.extend(labels.numpy())
            
        y_true = np.array(y_true)
        y_pred_probs = np.array(y_pred_probs)
        
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true_class = np.argmax(y_true, axis=1)
        else:
            y_true_class = y_true
            
        y_pred_class = np.argmax(y_pred_probs, axis=1)
        
        metrics = compute_all_metrics(y_true_class, y_pred_class)
        
        # Calculate ROC AUC
        pos_probs = y_pred_probs[:, 1]
        try:
            auc = roc_auc_score(y_true_class, pos_probs)
        except ValueError:
            auc = 0.5
            
        metrics['roc_auc'] = float(auc)
        
        cm = confusion_matrix(y_true_class, y_pred_class)
        ensure_dir("results/logs/alzheimer_plots")
        plot_confusion_matrix(cm, ['HC', 'AD'], "results/logs/alzheimer_plots/eval_cm.png")
        plot_roc_curve(y_true_class, pos_probs, "results/logs/alzheimer_plots/eval_roc.png")
        
        return metrics
    
    def get_model_for_federated(self, dataset_type: str = "adni") -> tf.keras.Model:
        """
        Get compiled model ready for federated learning.
        """
        if dataset_type == "adni":
            input_shape = tuple(self.config.model.input_shape_adni)
        else:
            input_shape = tuple(self.config.model.input_shape_oasis)
        
        model = build_vgg3d(input_shape=input_shape, num_classes=2, dropout_rate=self.config.model.dropout_rate)
        return self.compile_model(model)
