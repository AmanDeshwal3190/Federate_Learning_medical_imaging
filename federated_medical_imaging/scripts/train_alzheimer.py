"""
Main training script for Alzheimer's Disease classification.

Usage:
    python scripts/train_alzheimer.py --config config/alzheimer_config.yaml --dataset adni
    python scripts/train_alzheimer.py --config config/alzheimer_config.yaml --dataset oasis
    python scripts/train_alzheimer.py --config config/alzheimer_config.yaml --dataset both
    python scripts/train_alzheimer.py --config config/alzheimer_config.yaml --dataset adni --ensemble
    
Options:
    --config: Path to config YAML
    --dataset: Which dataset to use (adni/oasis/both)
    --ensemble: Enable ensemble learning (trains 5 models)
    --small: Use smaller model variant for faster testing
    --epochs: Override epoch count from config
"""
import argparse
import sys
import os
import json

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config_loader import ConfigLoader
from utils.logger import get_logger
from data.alzheimer.dataset import AlzheimerDataset
from models.alzheimer.trainer import AlzheimerTrainer
from models.alzheimer.transfer_learning import AlzheimerEnsemble

logger = get_logger("train_alzheimer")

def parse_args():
    parser = argparse.ArgumentParser(description="Train Alzheimer's 3D CNN model.")
    parser.add_argument('--config', type=str, default='config/alzheimer_config.yaml', help='Path to config file')
    parser.add_argument('--dataset', type=str, choices=['adni', 'oasis', 'both'], required=True, help='Dataset to train on')
    parser.add_argument('--ensemble', action='store_true', help='Enable ensemble training')
    parser.add_argument('--small', action='store_true', help='Use small model variant')
    parser.add_argument('--epochs', type=int, default=None, help='Override number of epochs')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load config
    config = ConfigLoader.load(args.config)
    
    # Override
    if args.epochs is not None:
        config.training.epochs = args.epochs
        
    datasets_to_run = ['adni', 'oasis'] if args.dataset == 'both' else [args.dataset]
    
    all_results = {}
    
    for ds_type in datasets_to_run:
        logger.info(f"==== Starting run for dataset: {ds_type.upper()} ====")
        
        # Load datasets (using AlzheimerDataset)
        dataset_loader = AlzheimerDataset(args.config)
        
        # Get 5-fold CV splits
        fold_datasets = dataset_loader.get_cv_splits(dataset_type=ds_type, num_folds=config.training.cv_folds)
        
        if args.ensemble:
            logger.info("Running ensemble training...")
            ensemble = AlzheimerEnsemble(num_models=5, dataset_type=ds_type)
            train_ds, val_ds = fold_datasets[0]
            
            histories = ensemble.train_ensemble(train_ds, val_ds, epochs=config.training.epochs)
            
            eval_results = ensemble.evaluate_all_methods(val_ds)
            logger.info("Ensemble Evaluation Results on Fold 0 Val Split:")
            for method, metrics in eval_results.items():
                logger.info(f"Method: {method} - Accuracy: {metrics.get('accuracy', 0):.4f}")
                
            all_results[f"{ds_type}_ensemble"] = eval_results
            
        else:
            logger.info("Running 5-fold cross-validation...")
            trainer = AlzheimerTrainer(args.config)
            
            cv_summary = trainer.run_cross_validation(fold_datasets, dataset_type=ds_type)
            all_results[f"{ds_type}_cv"] = cv_summary
            
            # Print summary table
            print(f"\n===========================================")
            print(f"Summary for {ds_type.upper()}")
            print(f"Mean Accuracy: {cv_summary['mean_accuracy']:.4f} ± {cv_summary['std_accuracy']:.4f}")
            print(f"Target Accuracy: {cv_summary['target_accuracy']:.4f}")
            print(f"===========================================\n")

    # Save all results
    os.makedirs("results/logs", exist_ok=True)
    with open("results/logs/alzheimer_results.json", "w") as f:
        json.dump(all_results, f, indent=4)
        
    logger.info("All training completed.")

if __name__ == "__main__":
    main()
