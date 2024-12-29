import os
import sys
import logging
from pathlib import Path
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
ROOT_DIR = Path(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = ROOT_DIR / 'data' / 'mslr'
MODEL_DIR = ROOT_DIR / 'models'
MSLR_FILES = ['train.pkl', 'vali.pkl', 'test.pkl']
MODEL_FILE = MODEL_DIR / 'lightgbm_ranker.txt'

def check_mslr_data() -> bool:
    """Check if MSLR data files exist."""
    return all((DATA_DIR / file).exists() for file in MSLR_FILES)

def check_mslr_model() -> bool:
    """Check if trained MSLR model exists."""
    return MODEL_FILE.exists()

def setup_mslr():
    """Setup MSLR data and model if not already present."""
    # Create directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check and download/process data if needed
    if not check_mslr_data():
        logger.info("MSLR data not found. Downloading and processing...")
        subprocess.run([sys.executable, 'scripts/download_mslr.py'], check=True)
    else:
        logger.info("MSLR data already exists.")
    
    # Check and train model if needed
    if not check_mslr_model():
        logger.info("MSLR model not found. Training model...")
        subprocess.run([sys.executable, 'scripts/train_mslr.py'], check=True)
    else:
        logger.info("MSLR model already exists.")
        
    logger.info("MSLR setup complete!")

def main():
    """Main function to setup MSLR."""
    try:
        setup_mslr()
    except Exception as e:
        logger.error(f"Error setting up MSLR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
