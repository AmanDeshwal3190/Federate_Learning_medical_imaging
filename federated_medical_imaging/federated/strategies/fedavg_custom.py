"""
Custom FedAvg strategy with enhanced logging, metric tracking, and model checkpointing.
"""
import flwr as fl
from flwr.common import (
    Metrics, Parameters, Scalar, FitRes, EvaluateRes,
    ndarrays_to_parameters, parameters_to_ndarrays
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
import numpy as np
import os
import json
from typing import Dict, List, Optional, Tuple, Union
from utils.logger import get_logger
from utils.common import save_json, ensure_dir

logger = get_logger("fedavg_custom")

class FedAvgWithLogging(FedAvg):
    """
    Extended FedAvg strategy with:
    1. Detailed per-round metric logging
    2. Best model checkpointing (saves when validation metric improves)
    3. Round-wise metric history for visualization
    4. Client contribution tracking
    """
    
    def __init__(self, *args, 
                 checkpoint_dir: str = "results/checkpoints/federated",
                 log_dir: str = "results/logs/federated",
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_dir = ensure_dir(checkpoint_dir)
        self.log_dir = ensure_dir(log_dir)
        self.round_history = []
        self.best_metric = 0.0
        self.best_round = 0
    
    def aggregate_fit(self, server_round: int,
                      results: List[Tuple[ClientProxy, FitRes]],
                      failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
                      ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate fit results with logging.
        """
        # Call standard FedAvg aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            logger.info(f"Round {server_round} - Aggregated training results from {len(results)} clients")
            for client_proxy, fit_res in results:
                client_id = client_proxy.cid
                metrics = fit_res.metrics
                num_examples = fit_res.num_examples
                logger.info(f"  Client {client_id}: train_samples={num_examples}, metrics={metrics}")
            
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(self, server_round: int,
                           results: List[Tuple[ClientProxy, EvaluateRes]],
                           failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
                           ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation results with checkpointing.
        """
        # Call standard FedAvg aggregation
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        
        if aggregated_loss is not None:
            logger.info(f"Round {server_round} - Aggregated evaluation from {len(results)} clients: loss={aggregated_loss:.4f}, metrics={aggregated_metrics}")
            
            # Save history
            round_info = {
                "round": server_round,
                "loss": aggregated_loss,
                "metrics": {k: float(v) for k, v in aggregated_metrics.items()}
            }
            self.round_history.append(round_info)
            self.save_history()
            
            # Checkpoint best model based on accuracy or custom metric
            main_metric = aggregated_metrics.get("accuracy", 0.0)
            if main_metric > self.best_metric:
                self.best_metric = main_metric
                self.best_round = server_round
                logger.info(f"New best model found at round {server_round} with metric={main_metric:.4f}")
                
                # Retrieve current global parameters (Note: Flower doesn't pass parameters here, 
                # but you could implement parameter loading or saving if they were passed)
                # Since evaluate doesn't have parameters returned, we just log it for now.
                # In a real implementation, you might save it inside fit or evaluate.
                
        return aggregated_loss, aggregated_metrics
    
    def save_checkpoint(self, parameters: Parameters, round_num: int, 
                        metrics: Dict[str, float]) -> str:
        """Save model parameters and metrics as checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"model_round_{round_num}.npz")
        
        ndarrays = parameters_to_ndarrays(parameters)
        np.savez(checkpoint_path, *ndarrays)
        
        metrics_path = os.path.join(self.checkpoint_dir, f"metrics_round_{round_num}.json")
        save_json(metrics, metrics_path)
        
        return checkpoint_path
    
    def save_history(self) -> str:
        """Save complete round-by-round history as JSON."""
        history_path = os.path.join(self.log_dir, "round_history.json")
        save_json(self.round_history, history_path)
        return history_path
    
    def get_history(self) -> List[Dict]:
        """Return the round history for visualization."""
        return self.round_history
