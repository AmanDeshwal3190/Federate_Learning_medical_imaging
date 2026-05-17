"""
Ensemble voting classifier for brain tumor classification.
Combines Random Forest, SVM, and KNN with soft voting.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import os
from typing import Dict, Tuple, List, Optional
from utils.logger import get_logger
from utils.metrics import compute_all_metrics
from utils.common import ensure_dir

logger = get_logger("ensemble_classifier")


class BrainTumorEnsembleClassifier:
    """
    Ensemble soft-voting classifier combining RF + SVM + KNN.
    
    From the paper:
    - Random Forest: n_estimators=200, max_depth=20, random_state=42
    - SVM: kernel='rbf', C=10.0, gamma='scale', probability=True
    - KNN: n_neighbors=7, weights='distance'
    - Voting: soft (uses predicted probabilities)
    - Evaluation: 5-fold cross-validation
    
    The ensemble forecast probability for each test sample:
    P(class) = (1/N) * sum(P_i(class)) for i in classifiers
    The class with the highest average probability wins.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize ensemble classifier with components from config.
        
        If config is None, use default paper values.
        """
        if config is None:
            rf_config = {'n_estimators': 200, 'max_depth': 20, 'random_state': 42}
            svm_config = {'kernel': 'rbf', 'C': 10.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
            knn_config = {'n_neighbors': 7, 'weights': 'distance'}
        else:
            rf_config = config.get('rf', {'n_estimators': 200, 'max_depth': 20, 'random_state': 42})
            svm_config = config.get('svm', {'kernel': 'rbf', 'C': 10.0, 'gamma': 'scale', 'probability': True, 'random_state': 42})
            knn_config = config.get('knn', {'n_neighbors': 7, 'weights': 'distance'})
            
        self.rf = RandomForestClassifier(**rf_config)
        self.svm = SVC(**svm_config)
        self.knn = KNeighborsClassifier(**knn_config)
        
        self.voting_clf = VotingClassifier(
            estimators=[('rf', self.rf), ('svm', self.svm), ('knn', self.knn)],
            voting='soft'
        )
        
        # Wrap everything in a standard scaler pipeline
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ensemble', self.voting_clf)
        ])
        
        self.is_fitted = False
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the ensemble on feature vectors.
        
        Args:
            X_train: Training features, shape (N, num_features)
            y_train: Training labels, shape (N,)
        """
        logger.info(f"Training ensemble classifier on {X_train.shape[0]} samples...")
        self.pipeline.fit(X_train, y_train)
        self.is_fitted = True
        logger.info("Ensemble training complete.")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before predict.")
        return self.pipeline.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before predict_proba.")
        return self.pipeline.predict_proba(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate on test set. Returns all metrics.
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before evaluate.")
        
        preds = self.predict(X_test)
        preds_proba = self.predict_proba(X_test)
        
        metrics = compute_all_metrics(y_test, preds, preds_proba)
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                        n_folds: int = 5) -> Dict[str, List[float]]:
        """
        Perform 5-fold cross-validation and return per-fold metrics.
        
        For each fold:
        1. Split data into train/val
        2. Train ensemble on train
        3. Predict on val
        4. Compute all metrics
        
        Returns dict with metric_name -> list of per-fold values
        """
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        metrics_dict = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': []
        }
        
        fold = 1
        for train_idx, val_idx in skf.split(X, y):
            logger.info(f"Running CV Fold {fold}/{n_folds}")
            X_train_f, X_val_f = X[train_idx], X[val_idx]
            y_train_f, y_val_f = y[train_idx], y[val_idx]
            
            # Create a fresh pipeline to avoid data leakage
            clf = BrainTumorEnsembleClassifier()
            clf.fit(X_train_f, y_train_f)
            
            fold_metrics = clf.evaluate(X_val_f, y_val_f)
            
            for k, v in fold_metrics.items():
                if k in metrics_dict:
                    metrics_dict[k].append(v)
                    
            fold += 1
            
        logger.info(f"CV complete. Average Acc: {np.mean(metrics_dict['accuracy']):.4f}")
        return metrics_dict
    
    def save(self, filepath: str) -> None:
        """Save trained ensemble using joblib."""
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before saving.")
        ensure_dir(os.path.dirname(filepath))
        joblib.dump(self.pipeline, filepath)
        logger.info(f"Ensemble saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load trained ensemble from file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found at {filepath}")
        self.pipeline = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Ensemble loaded from {filepath}")
    
    def get_individual_scores(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Get accuracy of each individual classifier for comparison.
        Returns dict with classifier_name → metrics_dict
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before scoring.")
            
        scaler = self.pipeline.named_steps['scaler']
        X_test_scaled = scaler.transform(X_test)
        
        results = {}
        for name, clf in [('Random Forest', self.voting_clf.named_estimators_['rf']),
                          ('SVM', self.voting_clf.named_estimators_['svm']),
                          ('KNN', self.voting_clf.named_estimators_['knn'])]:
            preds = clf.predict(X_test_scaled)
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X_test_scaled)
            else:
                probs = None
                
            metrics = compute_all_metrics(y_test, preds, probs)
            results[name] = metrics
            
        return results
