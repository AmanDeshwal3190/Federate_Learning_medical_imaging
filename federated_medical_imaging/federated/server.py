"""
Federated Learning Server using Flower (flwr).
Orchestrates multi-round training across simulated hospital clients.
"""
import flwr as fl
from flwr.common import Metrics, ndarrays_to_parameters, parameters_to_ndarrays
import numpy as np
import tensorflow as tf
import os
import json
import time
from typing import Dict, List, Optional, Tuple, Callable
from utils.config_loader import ConfigLoader
from utils.logger import get_logger, TrainingLogger
from utils.common import save_json, ensure_dir

# Import models
from models.brain_tumor.trainer import BrainTumorTrainer
from models.alzheimer.trainer import AlzheimerTrainer

logger = get_logger("fl_server")

class FederatedServer:
    """
    Federated Learning server that coordinates training across hospital clients.
    
    Responsibilities:
    1. Initialize global model and distribute weights to clients
    2. Receive updated weights from each client after local training
    3. Aggregate weights using FedAvg strategy
    4. Evaluate the global model after each round
    5. Log per-round metrics
    6. Save best global model checkpoint
    """
    
    def __init__(self, config_path: str = "config/federated_config.yaml",
                 model_type: str = "brain_tumor"):
        """
        Initialize FL server.
        
        Args:
            config_path: Path to federated config
            model_type: "brain_tumor" or "alzheimer" — determines which model architecture to use
        """
        self.config_path = config_path
        self.config = ConfigLoader.load(config_path)
        self.model_type = model_type
        self.training_logger = TrainingLogger("fl_server")
        self.round_metrics = []  # Store metrics per round
        
        # Load the appropriate model config
        if self.model_type == "brain_tumor":
            self.model_config = ConfigLoader.load("config/brain_tumor_config.yaml")
            self.trainer = BrainTumorTrainer(config_path="config/brain_tumor_config.yaml")
        elif self.model_type == "alzheimer":
            self.model_config = ConfigLoader.load("config/alzheimer_config.yaml")
            self.trainer = AlzheimerTrainer(config_path="config/alzheimer_config.yaml")
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
            
        # Build the initial global model
        self.model = self.trainer.get_model_for_federated()
        
    def get_initial_parameters(self) -> fl.common.Parameters:
        """
        Get initial model parameters for distribution to clients.
        
        Converts model weights (list of numpy arrays) to Flower Parameters object.
        """
        weights = self.model.get_weights()
        return ndarrays_to_parameters(weights)
    
    def weighted_average_metrics(self, metrics: List[Tuple[int, Metrics]]) -> Metrics:
        """
        Aggregate evaluation metrics from multiple clients using weighted average.
        
        Args:
            metrics: List of (num_examples, metrics_dict) tuples from clients
        Returns:
            Aggregated metrics dictionary
        """
        if not metrics:
            return {}
            
        # Extract unique metric keys
        metric_keys = metrics[0][1].keys()
        aggregated_metrics = {}
        
        total_examples = sum([num_examples for num_examples, _ in metrics])
        
        if total_examples == 0:
            return {key: 0.0 for key in metric_keys}
            
        for key in metric_keys:
            weighted_sum = sum([num_examples * m[key] for num_examples, m in metrics if key in m])
            aggregated_metrics[key] = weighted_sum / total_examples
            
        return aggregated_metrics

    def create_strategy(self, strategy_class=None, **kwargs) -> fl.server.strategy.Strategy:
        """
        Create the FL aggregation strategy.
        """
        server_config = self.config.get("server", {})
        strategy_config = self.config.get("strategy", {})
        
        min_fit_clients = server_config.get("min_fit_clients", 2)
        min_evaluate_clients = server_config.get("min_evaluate_clients", 2)
        min_available_clients = server_config.get("min_available_clients", 2)
        
        fraction_fit = strategy_config.get("fraction_fit", 1.0)
        fraction_evaluate = strategy_config.get("fraction_evaluate", 1.0)
        
        if strategy_class is None:
            from federated.strategies.fedavg_custom import FedAvgWithLogging
            strategy_class = FedAvgWithLogging
            
        strategy = strategy_class(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            initial_parameters=self.get_initial_parameters(),
            evaluate_metrics_aggregation_fn=self.weighted_average_metrics,
            fit_metrics_aggregation_fn=self.weighted_average_metrics,
            on_fit_config_fn=self.on_fit_config,
            on_evaluate_config_fn=self.on_evaluate_config,
            **kwargs
        )
        return strategy
    
    def on_fit_config(self, server_round: int) -> Dict[str, fl.common.Scalar]:
        """
        Configuration sent to clients before each training round.
        """
        training_config = self.model_config.get("training", {})
        batch_size = training_config.get("batch_size", 4)
        
        # Allow federated config to override epochs (e.g., local_epochs)
        strategy_config = self.config.get("strategy", {})
        epochs = strategy_config.get("local_epochs", 1)  # Default 1 epoch per client
        
        return {
            "epochs": epochs,
            "batch_size": batch_size,
            "server_round": server_round,
            "model_type": self.model_type
        }
    
    def on_evaluate_config(self, server_round: int) -> Dict[str, fl.common.Scalar]:
        """Configuration sent to clients for evaluation."""
        training_config = self.model_config.get("training", {})
        batch_size = training_config.get("batch_size", 4)
        
        return {
            "batch_size": batch_size,
            "server_round": server_round,
            "model_type": self.model_type
        }
    
    def start(self, client_fn: Callable, num_clients: int, 
              strategy: Optional[fl.server.strategy.Strategy] = None) -> fl.server.History:
        """
        Start the federated learning server in simulation mode.
        """
        server_config = self.config.get("server", {})
        num_rounds = server_config.get("num_rounds", 3)
        
        if strategy is None:
            strategy = self.create_strategy()
            
        logger.info(f"Starting FL Simulation for {num_rounds} rounds with {num_clients} clients...")
        
        # Avoid TF GPU memory growth issues before simulation
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                 for gpu in gpus:
                     tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                 logger.error(f"Cannot set GPU memory growth: {e}")
                 
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 2, "num_gpus": 0.5 if gpus else 0.0}
        )
        
        return history
    
    def save_results(self, history: fl.server.History, output_dir: str) -> None:
        """
        Save FL training results.
        """
        output_dir = ensure_dir(output_dir)
        
        # Extract metrics
        metrics = {"losses_distributed": history.losses_distributed,
                   "metrics_distributed": history.metrics_distributed,
                   "metrics_distributed_fit": history.metrics_distributed_fit}
                   
        save_json(metrics, os.path.join(output_dir, "fl_history.json"))
        
        # Plot evaluation history
        import matplotlib.pyplot as plt
        if history.metrics_distributed and "accuracy" in history.metrics_distributed:
            acc_list = history.metrics_distributed["accuracy"]
            rounds = [r for r, _ in acc_list]
            accs = [a for _, a in acc_list]
            
            plt.figure(figsize=(10, 6))
            plt.plot(rounds, accs, marker='o', linestyle='-')
            plt.title("Federated Evaluation Accuracy")
            plt.xlabel("Round")
            plt.ylabel("Accuracy")
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "eval_accuracy.png"))
            plt.close()
            
        logger.info(f"Results saved to {output_dir}")
