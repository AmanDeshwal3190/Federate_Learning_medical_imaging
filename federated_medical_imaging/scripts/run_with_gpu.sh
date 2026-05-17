#!/bin/bash
# =============================================================================
# Run Federated Medical Imaging Pipeline with GPU support in WSL2
# Usage: bash scripts/run_with_gpu.sh [--mode test|full|preprocess|etc]
# =============================================================================

PROJECT_DIR="/mnt/c/Users/Shrikant/OneDrive/Desktop/thefinalprojectmajor/copilot/federated_medical_imaging"
cd "$PROJECT_DIR"

# Activate virtual environment
if [ -d "venv_wsl" ]; then
    source venv_wsl/bin/activate
else
    echo "ERROR: Virtual environment not found. Run setup_wsl_tensorflow.sh first."
    exit 1
fi

# Verify GPU
echo "Checking GPU..."
python3 -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'GPUs: {len(gpus)} detected')"

# Run pipeline with all arguments passed through
echo "Running pipeline..."
python scripts/run_full_pipeline.py "$@"
