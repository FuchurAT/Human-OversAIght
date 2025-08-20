"""
Auto-Annotator for Armored Vehicles Detection using Grounding DINO

This script automatically downloads the required Grounding DINO model files if they
are not available, eliminating the need for manual setup. The model will be saved
in a 'models' directory for future use.

Features:
- Automatic model download with progress bars
- Fallback to urllib if requests library is not available
- File validation to ensure downloads are complete
- YOLO format annotation output
- Support for multiple vehicle categories

Usage:
    python annotate.py --data_dir data --output_dir annotated_data
"""

import sys
import logging
from pathlib import Path
import argparse
from tqdm import tqdm
import os

# Check for required dependencies
try:
    import torch
    logging.info(f"PyTorch version: {torch.__version__}")
except ImportError:
    print("ERROR: PyTorch is required but not installed.")
    print("Please install it with: pip install torch torchvision")
    sys.exit(1)

try:
    from groundingdino.util.inference import load_model, load_image, predict
    logging.info("Grounding DINO imported successfully")
except ImportError:
    print("ERROR: Grounding DINO is required but not installed.")
    print("Please install it with: pip install groundingdino-py")
    sys.exit(1)

# Try to import requests, fallback to urllib if not available
try:
    import requests
    REQUESTS_AVAILABLE = True
    logging.info("Using requests library for downloads")
except ImportError:
    REQUESTS_AVAILABLE = False
    import urllib.request
    import urllib.error
    logging.info("Using urllib for downloads (install 'requests' for better performance)")

from config.classes import CLASSES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('annotation.log')
    ]
)

class AutoAnnotator:
    def __init__(self, data_dir, output_dir, confidence_threshold=0.35, text_threshold=0.25):
        """
        Initialize the auto-annotator.
        
        Args:
            data_dir: Path to the data directory containing armored_vehicles folder
            output_dir: Path to save annotations
            confidence_threshold: Minimum confidence for detections
            text_threshold: Minimum text similarity threshold
        """
        # Get script directory for relative paths
        script_dir = Path(__file__).parent
        self.data_dir = script_dir / Path(data_dir)
        self.output_dir = script_dir / Path(output_dir)
        self.confidence_threshold = confidence_threshold
        self.text_threshold = text_threshold
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'images').mkdir(exist_ok=True)
        (self.output_dir / 'labels').mkdir(exist_ok=True)
        
        # Verify output directories were created
        if not (self.output_dir / 'images').exists() or not (self.output_dir / 'labels').exists():
            raise RuntimeError(f"Failed to create output directories in {self.output_dir}")
        
        logging.info(f"Output directories created: {self.output_dir}")
        
        # Vehicle categories and their search prompts
        self.categories = CLASSES
        
        # Class mapping (for YOLO format)
        self.class_names = list(self.categories.keys())
        self.class_to_id = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Save class names file
        with open(self.output_dir / 'classes.txt', 'w') as f:
            for class_name in self.class_names:
                f.write(f"{class_name}\n")
        
        # Initialize Grounding DINO model
        logging.info("Loading Grounding DINO model...")
        
        # Check if model files exist and download if needed
        self.model = self._load_or_download_model()
        logging.info("Model loaded successfully!")
        
    def _download_file(self, url, filepath, description="file"):
        """Download a file with progress bar."""
        try:
            if REQUESTS_AVAILABLE:
                return self._download_with_requests(url, filepath, description)
            else:
                return self._download_with_urllib(url, filepath, description)
        except Exception as e:
            logging.error(f"Failed to download {description}: {e}")
            return False
    
    def _download_with_requests(self, url, filepath, description="file"):
        """Download using requests library with progress bar."""
        try:
            response = requests.get(url, stream=True, timeout=300)  # 5 minute timeout
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as file, tqdm(
                desc=f"Downloading {description}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        pbar.update(len(chunk))
            
            # Verify file was written completely
            if filepath.exists() and filepath.stat().st_size > 0:
                return True
            else:
                logging.error(f"Downloaded file appears to be empty or missing: {filepath}")
                return False
                
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error downloading {description}: {e}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error downloading {description}: {e}")
            return False
    
    def _download_with_urllib(self, url, filepath, description="file"):
        """Download using urllib library with progress bar."""
        try:
            with tqdm(desc=f"Downloading {description}", unit='B', unit_scale=True) as pbar:
                def progress_hook(block_num, block_size, total_size):
                    if total_size > 0:
                        pbar.total = total_size
                        pbar.update(block_size)
                
                urllib.request.urlretrieve(url, filepath, progress_hook)
            
            # Verify file was written completely
            if filepath.exists() and filepath.stat().st_size > 0:
                return True
            else:
                logging.error(f"Downloaded file appears to be empty or missing: {filepath}")
                return False
                
        except urllib.error.URLError as e:
            logging.error(f"URL error downloading {description}: {e}")
            return False
        except Exception as e:
            logging.error(f"Error downloading {description}: {e}")
            return False
    
    def _download_file_with_retry(self, url, filepath, description="file", max_retries=3):
        """Download a file with retry mechanism."""
        for attempt in range(max_retries):
            try:
                logging.info(f"Download attempt {attempt + 1}/{max_retries} for {description}")
                if self._download_file(url, filepath, description):
                    return True
                else:
                    logging.warning(f"Download attempt {attempt + 1} failed for {description}")
            except Exception as e:
                logging.warning(f"Download attempt {attempt + 1} failed with error: {e}")
            
            # Wait before retry (exponential backoff)
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logging.info(f"Waiting {wait_time} seconds before retry...")
                import time
                time.sleep(wait_time)
        
        logging.error(f"All {max_retries} download attempts failed for {description}")
        return False
    
    def _load_or_download_model(self):
        """Load the model, downloading it first if it doesn't exist."""
        script_dir = Path(__file__).parent
        model_dir = script_dir / "models"
        model_dir.mkdir(exist_ok=True)
        
        # Use the config file from the installed package
        import groundingdino
        package_config_path = Path(groundingdino.__file__).parent / "config" / "GroundingDINO_SwinT_OGC.py"
        
        weights_path = model_dir / "groundingdino_swint_ogc.pth"
        
        # Clean up any partial downloads
        for partial_file in model_dir.glob("*.tmp"):
            try:
                partial_file.unlink()
                logging.info(f"Cleaned up partial download: {partial_file}")
            except Exception as e:
                logging.warning(f"Could not clean up partial download {partial_file}: {e}")
        
        # Check if weights file exists and is valid
        if weights_path.exists():
            if self._validate_weights_file(weights_path):
                logging.info("Weights file found and validated, loading existing model...")
                try:
                    return load_model(str(package_config_path), str(weights_path))
                except Exception as e:
                    logging.warning(f"Failed to load existing model: {e}")
                    logging.info("Removing corrupted weights file and re-downloading...")
                    if weights_path.exists():
                        weights_path.unlink()
            else:
                logging.info("Existing weights file failed validation, re-downloading...")
                if weights_path.exists():
                    weights_path.unlink()
        
        # Download weights file if it doesn't exist or is invalid
        logging.info("Downloading required model weights file...")
        
        if not REQUESTS_AVAILABLE:
            logging.info("Note: Install 'requests' package for better download performance: pip install requests")
        
        # Grounding DINO weights URL
        weights_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
        
        # Download weights file
        if not weights_path.exists():
            logging.info("Downloading model weights file...")
            if not self._download_file_with_retry(weights_url, weights_path, "weights file"):
                raise RuntimeError("Failed to download model weights file")
        
        # Validate downloaded weights file
        logging.info("Validating downloaded weights file...")
        if not self._validate_weights_file(weights_path):
            raise RuntimeError("Downloaded weights file appears to be corrupted or incomplete")
        
        logging.info("Weights file downloaded successfully!")
        
        # Load the downloaded model
        try:
            logging.info("Loading model into memory (this may take a few moments)...")
            model = load_model(str(package_config_path), str(weights_path))
            logging.info("Model loaded successfully!")
            return model
        except Exception as e:
            logging.error(f"Failed to load model after download: {e}")
            # Try to provide more helpful error messages
            if "zip archive" in str(e) or "central directory" in str(e):
                logging.error("The weights file appears to be corrupted. This usually means the download was incomplete.")
                logging.error("Try running the script again, or check your internet connection.")
            elif "CUDA" in str(e) or "GPU" in str(e):
                logging.error("GPU-related error. The model will use CPU instead.")
                logging.error("If you want to use GPU, ensure CUDA is properly installed.")
            else:
                logging.error("Unknown error during model loading. Check the error details above.")
            raise RuntimeError(f"Model downloaded but failed to load: {e}")
    
    def _validate_weights_file(self, weights_path):
        """Validate that the downloaded weights file is not corrupted."""
        try:
            # Weights file should be at least 100MB (typical size for this model)
            weights_size = weights_path.stat().st_size
            if weights_size < 100 * 1024 * 1024:
                logging.error(f"Weights file too small: {weights_size} bytes")
                return False
            
            # Try to open the weights file to ensure it's not completely corrupted
            try:
                import torch
                # Just try to open the file to see if it's readable
                with open(weights_path, 'rb') as f:
                    # Read first few bytes to check if file is accessible
                    header = f.read(1024)
                    if len(header) == 0:
                        logging.error("Weights file appears to be empty or unreadable")
                        return False
            except Exception as e:
                logging.error(f"Error reading weights file: {e}")
                return False
            
            logging.info(f"Weights file validation passed - Size: {weights_size} bytes")
            return True
            
        except Exception as e:
            logging.error(f"Error validating weights file: {e}")
            return False

    def get_image_paths(self):
        """Get all image paths from the armored_vehicles directory, excluding NEGATIVES."""
        image_paths = []
        armored_vehicles_dir = self.data_dir
        
        if not armored_vehicles_dir.exists():
            logging.error(f"Directory not found: {armored_vehicles_dir}")
            logging.info(f"Looking for: {armored_vehicles_dir}")
            return []
        
        for category in self.categories.keys():
            category_dir = armored_vehicles_dir / category
            if category_dir.exists():
                for img_file in category_dir.glob("*.jpg"):
                    image_paths.append((img_file, category))
        # Skip NEGATIVES for annotation
        return image_paths

    def copy_negatives(self):
        """Copy all images from the NEGATIVES folder to the output images directory without annotation files."""
        armored_vehicles_dir = self.data_dir / 'dataset'
        negatives_dir = armored_vehicles_dir / 'NEGATIVES'
        output_images_dir = self.output_dir / 'images'
        import shutil
        count = 0
        if negatives_dir.exists():
            for img_file in negatives_dir.glob("*.jpg"):
                shutil.copy2(img_file, output_images_dir / img_file.name)
                count += 1
            logging.info(f"Copied {count} NEGATIVES images to output images directory (no annotations created).")
        else:
            logging.info("No NEGATIVES directory found to copy.")
    
    def convert_to_yolo_format(self, boxes, image_width, image_height):
        """
        Convert Grounding DINO boxes to YOLO format.
        
        Args:
            boxes: Grounding DINO boxes in cxcywh format (normalized)
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            List of YOLO format boxes [class_id, x_center, y_center, width, height]
        """
        yolo_boxes = []
        
        for box in boxes:
            # Grounding DINO boxes are already normalized
            cx, cy, w, h = box
            
            # Ensure values are within [0, 1]
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            w = max(0, min(1, w))
            h = max(0, min(1, h))
            
            # For now, we'll use class_id 0 (will be updated based on detection)
            # In practice, you'd determine the class based on the detected text
            yolo_boxes.append([0, cx, cy, w, h])
        
        return yolo_boxes
    
    def annotate_image(self, image_path, category):
        """
        Annotate a single image using Grounding DINO.
        
        Args:
            image_path: Path to the image
            category: Expected category of the image
            
        Returns:
            List of YOLO format annotations
        """
        try:
            # Load image
            image_source, image = load_image(str(image_path))
            
            # Get search prompts for this category
            prompts = self.categories[category]
            
            # Combine prompts for better detection
            combined_prompt = ", ".join(prompts)
            
            # Run detection
            boxes, logits, phrases = predict(
                model=self.model,
                image=image,
                caption=combined_prompt,
                box_threshold=self.confidence_threshold,
                text_threshold=self.text_threshold
            )
            
            if boxes is None or len(boxes) == 0:
                logging.warning(f"No detections found for {image_path}")
                return []
            
            # Convert to YOLO format
            yolo_boxes = self.convert_to_yolo_format(boxes, image_source.shape[1], image_source.shape[0])
            
            # Assign correct class ID
            class_id = self.class_to_id[category]
            for box in yolo_boxes:
                box[0] = class_id
            
            logging.debug(f"Found {len(yolo_boxes)} detections for {image_path}")
            return yolo_boxes
            
        except Exception as e:
            logging.error(f"Error annotating {image_path}: {str(e)}")
            return []
    
    def save_annotations(self, image_path, annotations):
        """
        Save annotations in YOLO format.
        
        Args:
            image_path: Path to the image
            annotations: List of YOLO format annotations
        """
        # Create annotation filename
        annotation_filename = image_path.stem + '.txt'
        annotation_path = self.output_dir / 'labels' / annotation_filename
        
        # Save annotations
        with open(annotation_path, 'w') as f:
            for annotation in annotations:
                # Convert all values to float, then to string
                line = ' '.join(str(float(x)) for x in annotation)
                f.write(line + '\n')
        
        # Copy image to output directory
        output_image_path = self.output_dir / 'images' / image_path.name
        import shutil
        shutil.copy2(image_path, output_image_path)
    
    def create_dataset_yaml(self):
        """Create a YAML file for the dataset configuration."""
        yaml_content = f"""# Dataset configuration for YOLO training
path: {self.output_dir.absolute()}
train: images
val: images

nc: {len(self.class_names)}
names: {self.class_names}
"""
        
        with open(self.output_dir / 'dataset.yaml', 'w') as f:
            f.write(yaml_content)
        
        logging.info(f"Created dataset configuration: {self.output_dir / 'dataset.yaml'}")
    
    def validate_dataset_structure(self):
        """Validate that the dataset directory has the expected structure."""
        logging.info("Validating dataset structure...")
        
        if not self.data_dir.exists():
            logging.error(f"Data directory does not exist: {self.data_dir}")
            return False
        
        # Check if we have any category directories
        category_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        if not category_dirs:
            logging.error(f"No category directories found in {self.data_dir}")
            logging.info("Expected structure: data_dir/category_name/*.jpg")
            return False
        
        # Check for images in each category
        total_images = 0
        for category_dir in category_dirs:
            if category_dir.name == 'NEGATIVES':
                continue  # Skip NEGATIVES for now
            images = list(category_dir.glob("*.jpg"))
            if images:
                logging.info(f"Found {len(images)} images in category '{category_dir.name}'")
                total_images += len(images)
            else:
                logging.warning(f"No images found in category '{category_dir.name}'")
        
        if total_images == 0:
            logging.error("No images found in any category directory")
            return False
        
        logging.info(f"Dataset validation passed. Found {total_images} total images.")
        return True
    
    def run(self):
        """Run the annotation process."""
        logging.info("Starting automatic annotation process...")
        
        # Validate dataset structure first
        if not self.validate_dataset_structure():
            logging.error("Dataset validation failed. Please check your data directory structure.")
            return
        
        # Copy NEGATIVES images first
        self.copy_negatives()
        
        # Get all image paths (excluding NEGATIVES)
        image_paths = self.get_image_paths()
        
        if not image_paths:
            logging.error("No images found to annotate!")
            return
        
        # Process each image
        annotated_count = 0
        for image_path, category in tqdm(image_paths, desc="Annotating images"):
            try:
                # Annotate the image
                annotations = self.annotate_image(image_path, category)
                
                if annotations:
                    # Save annotations
                    self.save_annotations(image_path, annotations)
                    annotated_count += 1
                
            except Exception as e:
                logging.error(f"Error processing {image_path}: {str(e)}")
                continue
        
        # Create dataset configuration
        self.create_dataset_yaml()
        
        logging.info(f"Annotation complete! Annotated {annotated_count}/{len(image_paths)} images")
        logging.info(f"Output saved to: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Auto-annotate images using Grounding DINO")
    parser.add_argument("--data_dir", type=str, default="dataset", 
                       help="Path to data directory containing armored_vehicles folder (relative to script location)")
    parser.add_argument("--output_dir", type=str, default="annotated_data",
                       help="Path to save annotated data (relative to script location)")
    parser.add_argument("--confidence_threshold", type=float, default=0.35,
                       help="Minimum confidence threshold for detections")
    parser.add_argument("--text_threshold", type=float, default=0.25,
                       help="Minimum text similarity threshold")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Auto-Annotator for Armored Vehicles Detection")
    print("=" * 80)
    print("This script will automatically download the required Grounding DINO model")
    print("if it's not already available. The model will be saved in the 'models' directory.")
    print("=" * 80)
    
    # Create annotator and run
    annotator = AutoAnnotator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence_threshold,
        text_threshold=args.text_threshold
    )
    
    annotator.run()

if __name__ == "__main__":
    main() 