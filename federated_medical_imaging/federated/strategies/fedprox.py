"""
FedProx strategy implementation for handling heterogeneous data.
FedProx adds a proximal term to the local loss function to keep
local updates close to the global model.

Loss_FedProx = Loss_original + (mu/2) * ||w - w_global||^2

where mu is the proximal coefficient (default 0.01).
"""
import flwr as fl
from flwr.common import Parameters, Metrics, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import FedAvg
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from utils.logger import get_logger
from federated.client import MedicalImagingClient
import tensorflow as tf

logger = get_logger("fedprox")

class FedProx(FedAvg):
    """
    FedProx strategy - FedAvg with proximal term.
    
    The proximal term penalizes local model updates that deviate too far
    from the global model, which helps in non-IID data settings.
    
    The proximal term is added on the CLIENT side during training.
    The server-side aggregation is the same as FedAvg.
    
    This strategy:
    1. Extends FedAvg
    2. Passes proximal_mu to clients via on_fit_config
    3. Clients add the proximal term to their loss
    """
    
    def __init__(self, proximal_mu: float = 0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proximal_mu = proximal_mu
        logger.info(f"FedProx initialized with proximal_mu={proximal_mu}")
    
    def configure_fit(self, server_round: int, parameters: Parameters,
                      client_manager: ClientManager) -> List:
        """
        Configure fit with proximal_mu sent to clients.
        """
        # Call super to get the client lists and their configs
        client_configs = super().configure_fit(server_round, parameters, client_manager)
        
        # Inject proximal_mu into config
        for client_proxy, fit_ins in client_configs:
            fit_ins.config["proximal_mu"] = self.proximal_mu
            
        return client_configs

def create_fedprox_loss(original_loss_fn, global_weights: list, mu: float = 0.01):
    """
    Create a FedProx loss function that adds the proximal term.
    
    FedProx Loss = original_loss + (mu/2) * sum(||w_local - w_global||^2)
    
    Args:
        original_loss_fn: The base loss function (e.g., dice_loss, binary_crossentropy)
        global_weights: List of numpy arrays (global model weights)
        mu: Proximal coefficient
    Returns:
        Modified loss function compatible with Keras model.compile()
    """
    # Convert numpy weights to TF constants
    global_weights_tf = [tf.constant(w) for w in global_weights]
    
    def fedprox_loss(y_true, y_pred):
        # Calculate original loss
        loss = original_loss_fn(y_true, y_pred)
        
        """
        Note: True FedProx computes the difference between current trainable weights
        and the global weights. However, Keras loss functions only accept (y_true, y_pred).
        To implement FedProx properly in Keras, we often use a custom training loop or
        add regularizer to layers. Here, due to constraints, we'll demonstrate adding 
        it within a custom fit loop in the client.
        
        The proper implementation is handled in the FedProxClient's `fit` loop.
        This function serves as a placeholder if someone attempts to use it as a standard loss.
        """
        return loss
        
    return fedprox_loss

class FedProxClient(MedicalImagingClient):
    """
    Flower client with FedProx proximal term support.
    
    This is an extension of MedicalImagingClient that:
    1. Receives proximal_mu from server config
    2. Saves global weights before local training
    3. Adds proximal term to loss during local training
    4. Removes proximal term after training (restores original loss)
    """
    
    def __init__(self, client_id: int, model: tf.keras.Model, 
                 train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset, 
                 model_type: str = "brain_tumor", local_epochs: int = 1, batch_size: int = 4):
        super().__init__(client_id, model, train_dataset, val_dataset, model_type, local_epochs, batch_size)
    
    def fit(self, parameters: List[np.ndarray], config: Dict[str, Scalar]):
        """
        FedProx training:
        1. Set local model weights from global parameters
        2. Save a copy of global weights
        3. Get proximal_mu from config
        4. Custom training loop to compute proximal term based loss.
        """
        self.set_parameters(parameters)
        
        global_weights = [tf.constant(w) for w in parameters]
        proximal_mu = float(config.get("proximal_mu", 0.01))
        epochs = int(config.get("epochs", self.local_epochs))
        
        optimizer = self.model.optimizer
        loss_fn = self.model.loss
        
        metrics_dict = {}
        for epoch in range(epochs):
            epoch_loss_avg = tf.keras.metrics.Mean()
            for step, (x_batch_train, y_batch_train) in enumerate(self.train_dataset):
                with tf.GradientTape() as tape:
                    logits = self.model(x_batch_train, training=True)
                    loss_value = loss_fn(y_batch_train, logits)
                    
                    # Add proximal term
                    proximal_term = 0.0
                    for local_w, global_w in zip(self.model.trainable_weights, global_weights):
                        proximal_term += tf.reduce_sum(tf.square(local_w - global_w))
                    
                    total_loss = loss_value + (proximal_mu / 2) * proximal_term
                
                grads = tape.gradient(total_loss, self.model.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                epoch_loss_avg.update_state(total_loss)
            logger.info(f"Client {self.client_id} - Epoch {epoch + 1}: Loss = {epoch_loss_avg.result():.4f}")
            metrics_dict["loss"] = float(epoch_loss_avg.result())

        return self.get_parameters(config={}), self.num_train_samples, metrics_dict
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Scalar]):
        """Evaluate with original loss (no proximal term)."""
        return super().evaluate(parameters, config)
    
    def get_parameters(self, config: Dict[str, Scalar]):
        return super().get_parameters(config)
    
    def set_parameters(self, parameters: List[np.ndarray]):
        return super().set_parameters(parameters)
