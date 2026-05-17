import logging
import os
from datetime import datetime
from typing import Optional

def get_logger(
    name: str,
    log_dir: str = "results/logs",
    level: int = logging.INFO,
    log_to_file: bool = True
) -> logging.Logger:
    """Get a configured logger with console and optional file handlers."""
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class TrainingLogger:
    """Specialized logger for tracking training metrics per epoch."""
    
    def __init__(self, name: str, log_dir: str = "results/logs"):
        self.logger = get_logger(name, log_dir)
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
        }
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                  train_metrics: Optional[dict] = None, val_metrics: Optional[dict] = None) -> None:
        """Log metrics for a single training epoch."""
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        if train_metrics:
            self.history["train_metrics"].append(train_metrics)
        if val_metrics:
            self.history["val_metrics"].append(val_metrics)
        
        msg = f"Epoch {epoch:04d} | train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f}"
        if val_metrics:
            for k, v in val_metrics.items():
                msg += f" | {k}: {v:.4f}"
        self.logger.info(msg)
    
    def log_round(self, round_num: int, metrics: dict) -> None:
        """Log metrics for a federated learning round."""
        msg = f"FL Round {round_num:03d}"
        for k, v in metrics.items():
            msg += f" | {k}: {v:.4f}"
        self.logger.info(msg)
    
    def get_history(self) -> dict:
        """Return the full training history."""
        return self.history
