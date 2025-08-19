import sys
import logging
from pathlib import Path
import argparse
from tqdm import tqdm
from groundingdino.util.inference import load_model, load_image, predict
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
        
        # Check if model files exist in models directory
        script_dir = Path(__file__).parent
        model_dir = script_dir / "models"
        config_path = model_dir / "groundingdino_swint_ogc.py"
        weights_path = model_dir / "groundingdino_swint_ogc.pth"
        
        if not config_path.exists() or not weights_path.exists():
            logging.error("Model files not found. Please run setup_annotation.py first to download the required model files.")
            logging.error(f"Expected files:")
            logging.error(f"  - {config_path}")
            logging.error(f"  - {weights_path}")
            raise FileNotFoundError("Model files not found. Run setup_annotation.py first.")
        
        self.model = load_model(str(config_path), str(weights_path))
        logging.info("Model loaded successfully!")
        
    def get_image_paths(self):
        """Get all image paths from the armored_vehicles directory, excluding NEGATIVES."""
        image_paths = []
        armored_vehicles_dir = self.data_dir / 'dataset'
        
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
        yaml_content = ""
        """# Dataset configuration for YOLO training
        path: {self.output_dir.absolute()}
        train: images
        val: images

        nc: {len(self.class_names)}
        names: {self.class_names}
        """
        
        with open(self.output_dir / 'dataset.yaml', 'w') as f:
            f.write(yaml_content)
        
        logging.info(f"Created dataset configuration: {self.output_dir / 'dataset.yaml'}")
    
    def run(self):
        """Run the annotation process."""
        logging.info("Starting automatic annotation process...")
        
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
    parser.add_argument("--data_dir", type=str, default="data", 
                       help="Path to data directory containing armored_vehicles folder (relative to script location)")
    parser.add_argument("--output_dir", type=str, default="annotated_data",
                       help="Path to save annotated data (relative to script location)")
    parser.add_argument("--confidence_threshold", type=float, default=0.35,
                       help="Minimum confidence threshold for detections")
    parser.add_argument("--text_threshold", type=float, default=0.25,
                       help="Minimum text similarity threshold")
    
    args = parser.parse_args()
    
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