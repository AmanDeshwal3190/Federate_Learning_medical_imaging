"""
Script to launch the monitoring dashboard.

Usage:
    python scripts/run_dashboard.py --port 5000
    python scripts/run_dashboard.py --port 5000 --demo
    
Options:
    --port: Port number (default 5000)
    --host: Host address (default 0.0.0.0)
    --demo: Run with simulated demo data
    --debug: Enable Flask debug mode
"""
import argparse
import sys
import os
import time
import threading
import requests
import random
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.app import run_dashboard

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("dashboard_runner")


def simulate_fl_training(port: int, total_rounds: int = 20):
    """
    Background thread to generate synthetic federated learning metrics
    and push them to the dashboard API.
    """
    url = f"http://127.0.0.1:{port}/api/push-metrics"
    logger.info("Demo mode: Waiting for server to start before pushing metrics...")
    time.sleep(3)  # Give Flask time to boot
    
    # Base starting metrics
    global_acc = 0.45
    global_loss = 1.2
    
    clients = ["Hospital_A", "Hospital_B", "Hospital_C", "Clinic_X"]
    client_params = {
        "Hospital_A": {"samples": 500, "skill": 1.05},
        "Hospital_B": {"samples": 350, "skill": 0.95},
        "Hospital_C": {"samples": 400, "skill": 1.0},
        "Clinic_X": {"samples": 200, "skill": 0.9},
    }
    
    for r in range(1, total_rounds + 1):
        logger.info(f"Demo mode: Generating data for round {r}/{total_rounds}...")
        
        # Improve global metrics steadily with some noise
        improvement = random.uniform(0.01, 0.05) if r < 15 else random.uniform(0.001, 0.01)
        global_acc = min(0.95, global_acc + improvement)
        global_loss = max(0.1, global_loss * 0.85)
        
        # Generate client metrics
        client_metrics = {}
        for c in clients:
            # Active probability 90%
            if random.random() < 0.9:
                skill = client_params[c]["skill"]
                c_acc = min(0.99, global_acc * skill * random.uniform(0.95, 1.05))
                c_loss = max(0.05, global_loss * (1/skill) * random.uniform(0.9, 1.1))
                
                client_metrics[c] = {
                    "accuracy": c_acc,
                    "loss": c_loss,
                    "samples": client_params[c]["samples"]
                }
        
        payload = {
            "round": r,
            "global_metrics": {
                "accuracy": global_acc,
                "loss": global_loss
            },
            "client_metrics": client_metrics
        }
        
        try:
            res = requests.post(url, json=payload)
            if res.status_code == 200:
                logger.info(f"Demo mode: Pushed round {r} successfully.")
            else:
                logger.error(f"Demo mode: Failed to push round {r}. Status: {res.status_code}")
        except Exception as e:
            logger.error(f"Demo mode: Connection error: {e}")
            
        # Wait before next round
        time.sleep(3)
        
    logger.info("Demo mode: Completed all simulated rounds.")


def main():
    parser = argparse.ArgumentParser(description="Start the FL monitoring dashboard.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=5000, help="Port number")
    parser.add_argument("--demo", action="store_true", help="Run with simulated demo data")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    
    args = parser.parse_args()
    
    if args.demo:
        logger.info("Starting demo thread...")
        demo_thread = threading.Thread(
            target=simulate_fl_training,
            args=(args.port, 20),
            daemon=True
        )
        demo_thread.start()
        
    logger.info(f"Access the dashboard at http://127.0.0.1:{args.port}/")
    
    # Note: socketio.run internally starts the eventlet server if async_mode='eventlet'
    run_dashboard(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
