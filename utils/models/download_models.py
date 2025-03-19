"""
Script to download YOLO models for traffic violation detection.
This can be run separately to pre-download models before starting the application.
"""

import os
import urllib.request
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directory to save models
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = SCRIPT_DIR

# Model URLs
YOLO_WEIGHTS_URL = 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights'
YOLO_CONFIG_URL = 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg'
COCO_NAMES_URL = 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names'

# Model file paths
YOLO_WEIGHTS = os.path.join(MODEL_DIR, 'yolov4-tiny.weights')
YOLO_CONFIG = os.path.join(MODEL_DIR, 'yolov4-tiny.cfg')
COCO_NAMES = os.path.join(MODEL_DIR, 'coco.names')

def download_file(url, save_path):
    """Download a file from URL to the specified path."""
    try:
        logger.info(f"Downloading {url}...")
        urllib.request.urlretrieve(url, save_path)
        logger.info(f"Downloaded successfully to {save_path}")
        return True
    except Exception as e:
        logger.error(f"Error downloading {url}: {str(e)}")
        return False

def download_models():
    """Download all required model files."""
    # Create directory if it doesn't exist
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        logger.info(f"Created directory: {MODEL_DIR}")
    
    # Download YOLO weights
    if not os.path.exists(YOLO_WEIGHTS):
        download_file(YOLO_WEIGHTS_URL, YOLO_WEIGHTS)
    else:
        logger.info(f"YOLOv4-tiny weights already exist at {YOLO_WEIGHTS}")
    
    # Download YOLO config
    if not os.path.exists(YOLO_CONFIG):
        download_file(YOLO_CONFIG_URL, YOLO_CONFIG)
    else:
        logger.info(f"YOLOv4-tiny config already exists at {YOLO_CONFIG}")
    
    # Download COCO names
    if not os.path.exists(COCO_NAMES):
        download_file(COCO_NAMES_URL, COCO_NAMES)
    else:
        logger.info(f"COCO names already exist at {COCO_NAMES}")
    
    logger.info("Model download complete.")

if __name__ == "__main__":
    download_models()
