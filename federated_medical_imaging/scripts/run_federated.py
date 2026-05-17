"""
Main script to run federated learning experiments.

Usage:
    python scripts/run_federated.py --model brain_tumor --rounds 20
    python scripts/run_federated.py --model alzheimer --rounds 20 --fedprox
    python scripts/run_federated.py --model brain_tumor --compare
    
Options:
    --model: Model type (brain_tumor/alzheimer)
    --rounds: Number of FL rounds (default: 20)
    --clients: Number of clients (default: 3)
    --fedprox: Use FedProx instead of FedAvg
    --compare: Also train centralized baseline and compare
    --config: Path to federated config
"""

import argparse
import os
import sys

# Add root directory to path to allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from federated.simulation import FederatedSimulation
from utils.logger import get_logger

logger = get_logger("run_federated")

def main():
    parser = argparse.ArgumentParser(description="Run federated learning experiments.")
    parser.add_argument("--model", type=str, choices=["brain_tumor", "alzheimer"], required=True,
                        help="Model type (brain_tumor/alzheimer)")
    parser.add_argument("--rounds", type=int, default=20, help="Number of FL rounds")
    parser.add_argument("--clients", type=int, default=3, help="Number of clients")
    parser.add_argument("--fedprox", action="store_true", help="Use FedProx strategy")
    parser.add_argument("--compare", action="store_true", help="Train centralized baseline and compare")
    parser.add_argument("--config", type=str, default="config/federated_config.yaml", help="Path to config")
    
    args = parser.parse_args()
    
    logger.info(f"Configuration: Model={args.model}, Rounds={args.rounds}, Clients={args.clients}, FedProx={args.fedprox}")
    
    simulation = FederatedSimulation(config_path=args.config)
    
    # Overwrite rounds and clients in config if specifically provided via arguments
    # (If we wanted to dynamically overwrite config, we'd do it here)
    
    logger.info("--------- RUNNING FEDERATED SIMULATION ---------")
    fed_results = simulation.run_simulation(model_type=args.model, use_fedprox=args.fedprox)
    logger.info(f"Simulation completed. Output: {fed_results['output_dir']}")
    
    if args.compare:
        logger.info("--------- RUNNING CENTRALIZED BASELINE ---------")
        central_results = simulation.run_centralized_baseline(model_type=args.model)
        logger.info(f"Centralized baseline completed. Output: {central_results['output_dir']}")
        
        logger.info("--------- GENERATING COMPARISON ---------")
        comparison = simulation.compare_federated_vs_centralized(fed_results, central_results)
        
        logger.info("Comparison Summary:")
        logger.info(f"Federated Final Accuracy: {comparison['federated_final_acc']:.4f}")
        logger.info(f"Centralized Final Accuracy: {comparison['centralized_final_acc']:.4f}")
        logger.info(f"Federated Time: {comparison['federated_time']:.2f}s")
        logger.info(f"Centralized Time: {comparison['centralized_time']:.2f}s")
        
    logger.info("Experiment run successfully.")

if __name__ == "__main__":
    main()
