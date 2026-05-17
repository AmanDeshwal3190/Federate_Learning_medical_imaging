"""
Federated Learning Client using Flower.
Each client represents a hospital with local medical imaging data.
"""
import flwr as fl
from flwr.common import NDArrays, Scalar
import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from utils.config_loader import ConfigLoader
from utils.logger import get_logger
from utils.metrics import compute_all_metrics, compute_segmentation_metrics

logger = get_logger("fl_client")

class MedicalImagingClient(fl.client.NumPyClient):
    """
    Flower NumPy client for medical imaging federated learning.
    
    Each client:
    1. Receives global model weights from server
    2. Trains locally on its own hospital's data
    3. Sends updated weights back to server
    4. Never shares actual patient data
    """
    
    def __init__(self, client_id: int, model: tf.keras.Model,
                 train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset,
                 model_type: str = "brain_tumor",
                 local_epochs: int = 1, batch_size: int = 4):
        """
        Initialize a federated client.
        
        Args:
            client_id: Unique integer ID for this client/hospital
            model: Compiled Keras model (same architecture as server's global model)
            train_dataset: Local training data (tf.data.Dataset)
            val_dataset: Local validation data (tf.data.Dataset)
            model_type: "brain_tumor" or "alzheimer"
            local_epochs: Number of local training epochs per FL round
            batch_size: Training batch size
        """
        self.client_id = client_id
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model_type = model_type
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.num_train_samples = self._count_samples(train_dataset)
        self.num_val_samples = self._count_samples(val_dataset)
        logger.info(f"Client {client_id} initialized | "
                    f"Train: {self.num_train_samples} | Val: {self.num_val_samples}")
    
    def _count_samples(self, dataset: tf.data.Dataset) -> int:
        """Count total number of samples in a tf.data.Dataset."""
        count = 0
        for batch in dataset:
            count += int(batch[0].shape[0])
        return count
    
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """
        Return model weights as a list of numpy arrays.
        Called by Flower to get the current local model weights.
        """
        return self.model.get_weights()
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """
        Set model weights from a list of numpy arrays.
        Called by Flower to update local model with global weights.
        """
        self.model.set_weights(parameters)
    
    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]
            ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Train model on local data. Called by Flower each FL round.
        """
        self.set_parameters(parameters)
        
        epochs = int(config.get("epochs", self.local_epochs))
        
        history = self.model.fit(
            self.train_dataset,
            epochs=epochs,
            verbose=0
        )
        
        metrics_dict = {k: float(v[-1]) for k, v in history.history.items()}
        
        return self.get_parameters(config={}), self.num_train_samples, metrics_dict
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]
                 ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate model on local validation data. Called by Flower each FL round.
        """
        self.set_parameters(parameters)
        
        evaluation_results = self.model.evaluate(self.val_dataset, verbose=0)
        
        if not isinstance(evaluation_results, list):
            evaluation_results = [evaluation_results]
        
        loss = float(evaluation_results[0])
        
        metrics_dict = {}
        for metric_name, value in zip(self.model.metrics_names[1:], evaluation_results[1:]):
            metrics_dict[metric_name] = float(value)
            
        return loss, self.num_val_samples, metrics_dict

def create_client_fn(client_configs: List[Dict], model_type: str = "brain_tumor"
                     ) -> Callable:
    """
    Create a client_fn for Flower simulation.
    """
    # Import specific model
    if model_type == "brain_tumor":
        from models.brain_tumor.trainer import BrainTumorTrainer
        trainer_class = BrainTumorTrainer
        config_path = "config/brain_tumor_config.yaml"
    elif model_type == "alzheimer":
        from models.alzheimer.trainer import AlzheimerTrainer
        trainer_class = AlzheimerTrainer
        config_path = "config/alzheimer_config.yaml"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    def client_fn(cid: str) -> fl.client.Client:
        client_id_int = int(cid)
        client_config = next((c for c in client_configs if c["client_id"] == client_id_int), None)
        
        if client_config is None:
            raise ValueError(f"No config found for client ID: {cid}")
            
        trainer = trainer_class(config_path=config_path)
        model = trainer.get_model_for_federated()
        
        return MedicalImagingClient(
            client_id=client_id_int,
            model=model,
            train_dataset=client_config["train_dataset"],
            val_dataset=client_config["val_dataset"],
            model_type=model_type
        ).to_client()
        
    return client_fn
