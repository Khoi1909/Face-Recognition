"""
Download script for required model files
"""

import os
import urllib.request
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Model URLs
MODEL_URLS = {
    'opencv_face_detector_uint8.pb': 'https://github.com/spmallick/learnopencv/raw/master/FaceDetectionComparison/models/opencv_face_detector_uint8.pb',
    'opencv_face_detector.pbtxt': 'https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/opencv_face_detector.pbtxt'
}

def download_file(url: str, filepath: str) -> bool:
    """Download a file from URL"""
    try:
        print(f"Downloading {os.path.basename(filepath)}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"✅ Downloaded {os.path.basename(filepath)}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {os.path.basename(filepath)}: {e}")
        return False

def download_models():
    """Download all required model files"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("🔽 Downloading required model files...")
    print("=" * 50)
    
    success_count = 0
    for filename, url in MODEL_URLS.items():
        filepath = models_dir / filename
        
        if filepath.exists():
            print(f"⏭️  {filename} already exists, skipping...")
            success_count += 1
            continue
            
        if download_file(url, str(filepath)):
            success_count += 1
    
    print("=" * 50)
    print(f"✅ Successfully downloaded {success_count}/{len(MODEL_URLS)} model files")
    
    if success_count == len(MODEL_URLS):
        print("🎉 All model files are ready!")
        return True
    else:
        print("⚠️  Some model files failed to download. DNN detector may not work properly.")
        return False

if __name__ == "__main__":
    download_models() 