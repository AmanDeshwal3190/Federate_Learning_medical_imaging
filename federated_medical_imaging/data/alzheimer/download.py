"""
Alzheimer's Dataset Download Helper.
"""
import os
import argparse
import glob
import nibabel as nib
from utils.logger import get_logger
from utils.common import ensure_dir

logger = get_logger("alzheimer_download")

def print_adni_instructions():
    print("=" * 60)
    print("ADNI DATASET DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print("1. Register at: https://adni.loni.usc.edu/")
    print("2. Navigate to: Download -> Image Collections -> Advanced Search")
    print("3. Filter: Modality=MRI, Phase=ADNI2, Format=NIfTI")
    print("4. Select 100 AD + 100 HC subjects")
    print("5. Download and extract to: data/alzheimer/raw/ADNI/")
    print("   Structure should be:")
    print("   data/alzheimer/raw/ADNI/AD/*.nii.gz")
    print("   data/alzheimer/raw/ADNI/HC/*.nii.gz\n")

def print_oasis_instructions():
    print("=" * 60)
    print("OASIS DATASET DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print("1. Visit URL: https://www.oasis-brains.org/")
    print("2. Download OASIS-1 dataset")
    print("3. Extract to: data/alzheimer/raw/OASIS/")
    print("   Structure should be:")
    print("   data/alzheimer/raw/OASIS/AD/*.nii.gz")
    print("   data/alzheimer/raw/OASIS/HC/*.nii.gz\n")

def verify_dataset_structure(data_dir: str, dataset_name: str) -> bool:
    """Verify standard directory structure and NIfTI properties."""
    ad_dir = os.path.join(data_dir, "AD")
    hc_dir = os.path.join(data_dir, "HC")
    
    if not os.path.exists(ad_dir) or not os.path.exists(hc_dir):
        logger.error(f"Missing AD or HC subdirectories in {data_dir}")
        return False
        
    ad_files = glob.glob(os.path.join(ad_dir, "**", "*.nii*"), recursive=True)
    hc_files = glob.glob(os.path.join(hc_dir, "**", "*.nii*"), recursive=True)
    
    logger.info(f"{dataset_name} AD Scans found: {len(ad_files)}")
    logger.info(f"{dataset_name} HC Scans found: {len(hc_files)}")
    
    failed = 0
    all_files = ad_files + hc_files
    for f in all_files:
        try:
            nib.load(f)
        except Exception:
            logger.error(f"Corrupted or invalid NIfTI file: {f}")
            failed += 1
            
    if failed > 0:
        logger.warning(f"Found {failed} corrupted files.")
        return False
        
    return True

def verify_adni_data(data_dir: str):
    logger.info("Verifying ADNI data structure...")
    verify_dataset_structure(data_dir, "ADNI")

def verify_oasis_data(data_dir: str):
    logger.info("Verifying OASIS data structure...")
    verify_dataset_structure(data_dir, "OASIS")

def create_mni_template(output_path: str):
    print("=" * 60)
    print("MNI152 TEMPLATE INSTRUCTIONS")
    print("=" * 60)
    print(f"Target path: {output_path}")
    print("To obtain MNI152 1mm template:")
    print("1. Try to download from: https://nist.mni.mcgill.ca/files/atlases/MNI152/")
    print("2. Or use FSL: copy ${FSLDIR}/data/standard/MNI152_T1_1mm.nii.gz")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alzheimer Dataset Download Helper")
    parser.add_argument("--dataset", type=str, choices=["adni", "oasis", "both"], default="both")
    parser.add_argument("--verify", action="store_true", help="Verify downloaded data")
    parser.add_argument("--output_dir", type=str, default="data/alzheimer/raw", help="Expected raw data dir")
    args = parser.parse_args()

    if args.dataset in ["adni", "both"]:
        print_adni_instructions()
        if args.verify:
            verify_adni_data(os.path.join(args.output_dir, "ADNI"))
            
    if args.dataset in ["oasis", "both"]:
        print_oasis_instructions()
        if args.verify:
            verify_oasis_data(os.path.join(args.output_dir, "OASIS"))
            
    create_mni_template(os.path.join(args.output_dir, "MNI152_1mm.nii.gz"))
