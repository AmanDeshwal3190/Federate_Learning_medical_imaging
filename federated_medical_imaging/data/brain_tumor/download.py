"""
Brain Tumor Dataset Download Helper.

Provides download instructions and verification for:
1. BraTS2020 - Requires registration at https://www.med.upenn.edu/cbica/brats2020/data.html
2. Figshare - Direct download from https://figshare.com/articles/dataset/brain_tumor_dataset/1512427

Usage:
    python -m data.brain_tumor.download --dataset brats2020 --output_dir data/brain_tumor/raw
    python -m data.brain_tumor.download --dataset figshare --output_dir data/brain_tumor/raw
"""
import os
import argparse
import urllib.request
import zipfile
import glob
import sys
from utils.logger import get_logger
from utils.common import ensure_dir

logger = get_logger("brain_tumor_download")

def download_figshare(output_dir: str) -> None:
    """Download the Figshare dataset (it's publicly available via direct URL)."""
    figshare_dir = os.path.join(output_dir, "Figshare")
    ensure_dir(figshare_dir)
    
    # URL to a representative figshare zip for brain tumor dataset.
    # Note: URL might need to be precise, we use a placeholder or official if available.
    # The actual Figshare dataset has multiple parts. We will download a common mirror
    # or print instructions if direct bulk link is unavailable.
    # For now, we simulate downloading a zip. 
    # In a real environment, users can use standard kaggle or figshare CLI.
    logger.info("Downloading Figshare dataset...")
    # The direct link to a combined zip can sometimes change.
    url = "https://figshare.com/ndownloader/articles/1512427/versions/5"
    zip_path = os.path.join(figshare_dir, "brain_tumor_dataset.zip")
    
    try:
        urllib.request.urlretrieve(url, zip_path)
        logger.info(f"Downloaded Figshare to {zip_path}")
        extract_archive(zip_path, figshare_dir)
    except Exception as e:
        logger.error(f"Failed to download Figshare dataset: {e}")
        logger.info("Please download manually from: https://figshare.com/articles/dataset/brain_tumor_dataset/1512427")

def print_brats2020_instructions(output_dir: str) -> None:
    """Prints detailed instructions for BraTS2020 registration and download."""
    brats_dir = os.path.join(output_dir, "BraTS2020")
    ensure_dir(brats_dir)
    instructions = f"""
    =========================================================
    BraTS 2020 Dataset Download Instructions
    =========================================================
    1. The BraTS 2020 dataset requires registration and approval.
    2. Please visit: https://www.med.upenn.edu/cbica/brats2020/data.html
    3. Register for an account and request access to the BraTS 2020 data.
    4. Once approved, download the 'MICCAI_BraTS2020_TrainingData.zip' file.
    5. Place the downloaded .zip file in the following directory:
       {os.path.abspath(brats_dir)}
    6. Extract the zip file in that directory.
    =========================================================
    """
    logger.info(instructions)

def extract_archive(archive_path: str, extract_dir: str) -> None:
    """Extract zip/tar files to the correct directories."""
    logger.info(f"Extracting {archive_path} to {extract_dir}")
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        logger.info(f"Extraction complete for {archive_path}")
    else:
        logger.warning(f"Unsupported archive format for {archive_path}. Only .zip is supported.")

def verify_directory_structure(output_dir: str, dataset: str) -> bool:
    """
    Verify the expected directory structure after extraction.
    """
    if dataset == "brats2020":
        brats_dir = os.path.join(output_dir, "BraTS2020")
        if not os.path.exists(brats_dir):
            return False
            
        patient_dirs = glob.glob(os.path.join(brats_dir, "BraTS20_Training_*"))
        if not patient_dirs:
            # Let's check MICCAI_BraTS2020_TrainingData in case it's nested
            nested = os.path.join(brats_dir, "MICCAI_BraTS2020_TrainingData")
            if os.path.exists(nested):
                patient_dirs = glob.glob(os.path.join(nested, "BraTS20_Training_*"))
                
        if not patient_dirs:
            logger.error(f"No BraTS20_Training_* directories found in {brats_dir}.")
            return False
            
        # Check first patient dir for 5 modalitiy files
        pdir = patient_dirs[0]
        nii_files = glob.glob(os.path.join(pdir, "*.nii*"))
        if len(nii_files) >= 5:
            logger.info("BraTS2020 directory structure verified.")
            return True
        else:
            logger.error(f"Expected 5 nii.gz files in {pdir}, found {len(nii_files)}.")
            return False

    elif dataset == "figshare":
        figshare_dir = os.path.join(output_dir, "Figshare")
        if not os.path.exists(figshare_dir):
            return False
        
        # Figshare expects glioma, meningioma, pituitary or directly images
        # Since it's a direct download of parts, the exact nested structure varies.
        # But we expect at least some directories or images.
        pngs = glob.glob(os.path.join(figshare_dir, "**", "*.png"), recursive=True)
        mats = glob.glob(os.path.join(figshare_dir, "**", "*.mat"), recursive=True)
        
        if pngs or mats:
            logger.info("Figshare directory structure verified.")
            return True
        else:
            logger.error(f"No PNG/MAT images found in {figshare_dir}.")
            return False
    return False

def verify_downloaded_files(output_dir: str, dataset: str) -> None:
    """Verify downloaded files (check file count, total size)."""
    dataset_dir = os.path.join(output_dir, "BraTS2020" if dataset == 'brats2020' else "Figshare")
    if not os.path.exists(dataset_dir):
        logger.error(f"Dataset directory {dataset_dir} does not exist.")
        return
        
    num_files = 0
    total_size = 0
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            num_files += 1
            total_size += os.path.getsize(os.path.join(root, f))
            
    size_mb = total_size / (1024 * 1024)
    logger.info(f"Verification for {dataset}: found {num_files} files, total size {size_mb:.2f} MB.")

def main():
    parser = argparse.ArgumentParser(description="Brain Tumor Dataset Download Helper")
    parser.add_argument("--dataset", type=str, required=True, choices=["brats2020", "figshare"],
                        help="Dataset to download/verify")
    parser.add_argument("--output_dir", type=str, default="data/brain_tumor/raw",
                        help="Output directory for raw data")
    
    args = parser.parse_args()
    
    if args.dataset == "brats2020":
        print_brats2020_instructions(args.output_dir)
        verify_directory_structure(args.output_dir, args.dataset)
        verify_downloaded_files(args.output_dir, args.dataset)
    elif args.dataset == "figshare":
        download_figshare(args.output_dir)
        verify_directory_structure(args.output_dir, args.dataset)
        verify_downloaded_files(args.output_dir, args.dataset)

if __name__ == "__main__":
    main()
