Federated Learning for Disease Detection in Medical Imaging
Overview
This project implements a privacy-preserving federated learning framework for:

Brain Tumor Segmentation & Classification using a 3D U-Net combined with a DenseNet feature extractor and an Ensemble classifier (Random Forest, SVM, KNN) on the BraTS2020 and Figshare datasets.
Alzheimer's Disease Classification using a custom 3D VGG CNN leveraging 5-fold cross-validation on the ADNI and OASIS datasets.
Multiple hospitals (simulated clients) can collaboratively train AI models without sharing raw patient data, maintaining strict privacy while achieving near-centralized model accuracy.

Authors
Shrikant (shrikantyadav9024@gmail.com) — Roll No: 2021977
Aman Kumar (amank756195@gmail.com) — Roll No: 2021937
Yash Vardhan S. Parmar (yvardhan887@gmail.com) — Roll No: 2021992
Supervisor: Ms. Ruchira Rawat, Graphic Era (Deemed to be University), Dehradun

Key Features
Federated Training: Implemented using Flower (flwr) with custom FedAvg and FedProx strategies.
Advanced 3D Deep Learning: Full implementations of 3D U-Net and 3D VGG architectures using TensorFlow/Keras.
Privacy Preservation: Data remains on local clients; only model weights are transmitted and aggregated securely on the central server.
Comprehensive Monitoring: A real-time Flask + Socket.IO web dashboard built with dark-mode styling for tracking training progress and global model metrics.
End-to-End Orchestration: Easy-to-use pipeline to handle preprocessing, local training, federated simulation, and extensive statistical evaluation.
Architecture
The system consists of three main components:

Central Server: Coordinates FL rounds, handles global weight aggregation, and runs the evaluation process.
Hospital Clients: 3 simulated clients which run local training on isolated data splits.
Web Dashboard: Listens to server events and visualizes metrics in real-time.
Requirements
Python 3.9+
NVIDIA GPU with CUDA (highly recommended for 3D CNNs)
~10GB RAM (minimal) / ~32GB RAM (ideal for large sets)
Required Packages (see requirements.txt)
Installation
Clone the Repository

git clone https://github.com/shrikantyadav870/Federate_Learning.git
cd Federate_Learning
Create a Virtual Environment

python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
Install Dependencies

pip install -r requirements.txt
Quick Start
You can run a quick simulation using synthetic data without needing the massive raw MRI datasets:

# Generate tiny synthetic 3D MRI data for both tasks
python scripts/generate_synthetic_data.py --type both --num_samples 10

# Run the master pipeline in 'test' mode
python scripts/run_full_pipeline.py --mode test
Dataset Setup
For a full run, download the raw MRI datasets and place them into the data/raw/ directory structure as defined below:

Brain Tumor Data:
BraTS2020: data/raw/brain_tumor/brats/...
Figshare: data/raw/brain_tumor/figshare/...
Alzheimer's Data:
ADNI: data/raw/alzheimer/adni/...
OASIS: data/raw/alzheimer/oasis/...
Running the Pipeline
You can run the full process or individual stages:

Full Process (Everything automagically)

python scripts/run_full_pipeline.py --mode full
Run Preprocessing Only

python scripts/run_full_pipeline.py --mode preprocess
Train Centralized Baselines (No FL)

python scripts/run_full_pipeline.py --mode train_local
Run Federated Simulation

python scripts/run_full_pipeline.py --mode federated
Generate Final Evaluation Report & Plots

python scripts/run_full_pipeline.py --mode evaluate
Monitoring Dashboard
Open a new terminal and activate the virtual environment.
Launch the real-time FL Dashboard:
python scripts/run_dashboard.py
Open http://localhost:5000 in your web browser. Monitor the federated training seamlessly.
Results
A summary of expected performance metrics compared to centralized models according to our research benchmarks:

Method	Dataset	Accuracy	Sensitivity	Specificity	Dice Score
Centralized U-Net	BraTS2020	99.94%	92.3%	92.0%	~0.88
Federated U-Net	BraTS2020	98.7%	90.1%	91.5%	~0.86
Centralized VGG3D	ADNI	73.4%	-	-	-
Federated VGG3D	ADNI	71.5%	-	-	-
Project Structure
federated_medical_imaging/
├── config/              # YAML config files for FL and diseases
├── dashboard/           # Flask dashboard scripts, CSS, and HTML
├── data/                # Data pipelines (preprocessing, datasets)
├── federated/           # FL components (Server, Client, strategies)
├── models/              # Model architectures (3D U-Net, VGG3D)
├── results/             # Checkpoints, logs, and evaluation reports
├── scripts/             # Main execution scripts
├── tests/               # Pytest integration files
└── utils/               # Loggers, metric calculators, and visualizations
Citation
If you use our work, please refer to the corresponding research paper structure associated with graphic era university guidelines.

License
MIT License
