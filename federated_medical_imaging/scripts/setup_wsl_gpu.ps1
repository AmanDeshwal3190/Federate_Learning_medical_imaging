# =============================================================================
# WSL2 + TensorFlow GPU Setup Script for Federated Medical Imaging
# =============================================================================
# Run this script in an ELEVATED (Administrator) PowerShell terminal
# AFTER restarting your PC (to activate WSL/VM features).
#
# Usage:
#   Right-click PowerShell -> Run as Administrator
#   cd "c:\Users\Shrikant\OneDrive\Desktop\thefinalprojectmajor\copilot\federated_medical_imaging"
#   .\scripts\setup_wsl_gpu.ps1
# =============================================================================

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " WSL2 + TensorFlow GPU Setup" -ForegroundColor Cyan
Write-Host " For: Federated Medical Imaging Project" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check if WSL features are enabled
Write-Host "[Step 1] Checking WSL features..." -ForegroundColor Yellow
$wslFeature = Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux
$vmFeature = Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform

if ($wslFeature.State -ne "Enabled" -or $vmFeature.State -ne "Enabled") {
    Write-Host "  WSL or VirtualMachinePlatform features are NOT enabled." -ForegroundColor Red
    Write-Host "  Enabling them now..." -ForegroundColor Yellow
    dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
    dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
    Write-Host ""
    Write-Host "  >>> RESTART YOUR PC and re-run this script! <<<" -ForegroundColor Red
    pause
    exit
}
Write-Host "  WSL features are enabled." -ForegroundColor Green

# Step 2: Set WSL2 as default
Write-Host "[Step 2] Setting WSL2 as default version..." -ForegroundColor Yellow
wsl --set-default-version 2
Write-Host "  Done." -ForegroundColor Green

# Step 3: Install Ubuntu if not already installed
Write-Host "[Step 3] Installing Ubuntu 24.04..." -ForegroundColor Yellow
$distros = wsl --list --quiet 2>$null
if ($distros -match "Ubuntu") {
    Write-Host "  Ubuntu is already installed." -ForegroundColor Green
} else {
    wsl --install -d Ubuntu-24.04 --no-launch
    Write-Host "  Ubuntu 24.04 installed. You'll need to set a username/password on first launch." -ForegroundColor Yellow
}

# Step 4: Create the setup script that will run INSIDE WSL
Write-Host "[Step 4] Creating WSL setup script..." -ForegroundColor Yellow

$wslSetupScript = @'
#!/bin/bash
set -e

echo "============================================"
echo " Setting up TensorFlow GPU inside WSL2"
echo "============================================"

# Update system
echo "[1/6] Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python and tools
echo "[2/6] Installing Python 3.12 and tools..."
sudo apt install -y python3 python3-pip python3-venv python3-dev build-essential

# Verify nvidia-smi works in WSL
echo "[3/6] Verifying GPU access..."
if nvidia-smi > /dev/null 2>&1; then
    echo "  GPU detected in WSL2!"
    nvidia-smi
else
    echo "  WARNING: nvidia-smi not found. Make sure NVIDIA drivers are installed on Windows host."
    echo "  GPU features may not work."
fi

# Create virtual environment
echo "[4/6] Creating Python virtual environment..."
PROJECT_DIR="/mnt/c/Users/Shrikant/OneDrive/Desktop/thefinalprojectmajor/copilot/federated_medical_imaging"
cd "$PROJECT_DIR"

python3 -m venv venv_wsl
source venv_wsl/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install TensorFlow with CUDA support
echo "[5/6] Installing TensorFlow with GPU support..."
pip install 'tensorflow[and-cuda]'

# Install project dependencies
echo "[6/6] Installing project dependencies..."
pip install -r requirements.txt

# Verify GPU detection
echo ""
echo "============================================"
echo " Verification"
echo "============================================"
python3 -c "
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs detected: {len(gpus)}')
for gpu in gpus:
    print(f'  - {gpu}')
if gpus:
    print('SUCCESS: TensorFlow GPU is working!')
else:
    print('WARNING: No GPU detected. Check CUDA installation.')
"

echo ""
echo "============================================"
echo " Setup Complete!"
echo "============================================"
echo ""
echo "To run the project with GPU acceleration:"
echo "  1. Open WSL terminal (wsl)"
echo "  2. cd $PROJECT_DIR"
echo "  3. source venv_wsl/bin/activate"
echo "  4. python scripts/run_full_pipeline.py --mode test"
echo ""
'@

$scriptPath = "scripts/setup_wsl_tensorflow.sh"
$wslSetupScript | Out-File -FilePath $scriptPath -Encoding utf8 -NoNewline
# Fix line endings to Unix format
(Get-Content $scriptPath -Raw) -replace "`r`n", "`n" | Set-Content $scriptPath -NoNewline

Write-Host "  Created: $scriptPath" -ForegroundColor Green

# Step 5: Instructions
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host " NEXT STEPS" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Launch Ubuntu from Start Menu (first time: set username/password)" -ForegroundColor White
Write-Host "2. In the Ubuntu terminal, run:" -ForegroundColor White
Write-Host ""
Write-Host '   cd "/mnt/c/Users/Shrikant/OneDrive/Desktop/thefinalprojectmajor/copilot/federated_medical_imaging"' -ForegroundColor Green
Write-Host "   chmod +x scripts/setup_wsl_tensorflow.sh" -ForegroundColor Green
Write-Host "   bash scripts/setup_wsl_tensorflow.sh" -ForegroundColor Green
Write-Host ""
Write-Host "3. After setup completes, run your project:" -ForegroundColor White
Write-Host "   source venv_wsl/bin/activate" -ForegroundColor Green
Write-Host "   python scripts/run_full_pipeline.py --mode test" -ForegroundColor Green
Write-Host ""
Write-Host "Your RTX 3050 will then be used for model training!" -ForegroundColor Yellow
pause
