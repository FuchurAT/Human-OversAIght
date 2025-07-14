#!/usr/bin/env python3
"""
YOLO Training Script for Armored Vehicles Detection

This script trains a custom YOLO model on the annotated armored vehicles dataset.
It includes data splitting, model configuration, and comprehensive training options.
"""

import os
import sys
import logging
import argparse
import shutil
import random
from pathlib import Path
from typing import List, Tuple
import yaml
import time
import torch

print(f"CUDA: {torch.cuda.is_available()}")
print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)

class ArmoredVehiclesTrainer:
    def __init__(self, 
                 dataset_path: str = "annotated_data",
                 output_dir: str = "runs/train",
                 model_size: str = "n",
                 epochs: int = 100,
                 batch_size: int = 320,
                 img_size: int = 640,
                 split_ratio: float = 0.8):
        """
        Initialize the trainer.
        
        Args:
            dataset_path: Path to the annotated dataset
            output_dir: Directory to save training results
            model_size: YOLO model size (n, s, m, l, x)
            epochs: Number of training epochs
            batch_size: Batch size for training
            img_size: Input image size
            split_ratio: Train/validation split ratio
        """
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.model_size = model_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.split_ratio = split_ratio
        
        # Vehicle categories
        self.classes = [    
            "ARMED_POLICEMEN",
            "CAR_FIRE",
            "FIRE",
            "FIRE_FIREFIGHTER",
            "FIRE_TRUCK",
            "HEALTH_AMBULANCE",
            "IMMIGRANT",
            "MILITARY_OFFICER",
            "MILITARY_SOLDIER",
            "MILITARY_VEHICLE",
            "POLICE",
            "MILITARY_VEHICLE",
            "POLICE_MAN",
            "PRISON",
            "PROTEST",
            "RIOT",
            "RIOT_POLICE",
            "TEARGAS"
        ]
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Initialized trainer with {len(self.classes)} classes")
        logging.info(f"Classes: {self.classes}")
    
    def split_dataset(self) -> Tuple[List[Path], List[Path]]:
        """
        Split the dataset into train and validation sets.
        
        Returns:
            Tuple of (train_images, val_images)
        """
        images_dir = self.dataset_path / "images"
        labels_dir = self.dataset_path / "labels"
        
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        # Get all image files
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png"))
        
        if not image_files:
            raise FileNotFoundError(f"No image files found in {images_dir}")
        
        # Filter images that have corresponding label files
        valid_images = []
        for img_path in image_files:
            label_path = labels_dir / f"{img_path.stem}.txt"
            # Accept images with an empty label file or no label file at all
            if not label_path.exists() or os.stat(label_path).st_size == 0:
                label_path.touch()
            valid_images.append(img_path)
        
        if not valid_images:
            raise FileNotFoundError(f"No valid image-label pairs found")
        
        # Shuffle and split
        random.shuffle(valid_images)
        split_idx = int(len(valid_images) * self.split_ratio)
        train_images = valid_images[:split_idx]
        val_images = valid_images[split_idx:]
        
        logging.info(f"Dataset split: {len(train_images)} train, {len(val_images)} validation")
        
        return train_images, val_images
    
    def create_split_directories(self, train_images: List[Path], val_images: List[Path]):
        """Create train and validation directories with split data."""
        # Create directories
        train_dir = self.dataset_path / "train"
        val_dir = self.dataset_path / "val"
        
        for dir_path in [train_dir, val_dir]:
            dir_path.mkdir(exist_ok=True)
            (dir_path / "images").mkdir(exist_ok=True)
            (dir_path / "labels").mkdir(exist_ok=True)
        
        # Copy train images and labels
        for img_path in train_images:
            # Copy image
            shutil.copy2(img_path, train_dir / "images" / img_path.name)
            # Copy label
            label_path = self.dataset_path / "labels" / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, train_dir / "labels" / f"{img_path.stem}.txt")
        
        # Copy validation images and labels
        for img_path in val_images:
            # Copy image
            shutil.copy2(img_path, val_dir / "images" / img_path.name)
            # Copy label
            label_path = self.dataset_path / "labels" / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, val_dir / "labels" / f"{img_path.stem}.txt")
        
        logging.info(f"Created train/val split directories")
    
    def create_dataset_yaml(self):
        """Create the dataset YAML file for YOLO training."""
        yaml_content = {
            'path': str(self.dataset_path.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(self.classes),
            'names': self.classes
        }
        
        yaml_path = self.dataset_path / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        logging.info(f"Created dataset YAML: {yaml_path}")
        return yaml_path
    
    def create_training_config(self) -> Path:
        """Create a custom training configuration file."""
        config_content = f"""# YOLO Training Configuration for Armored Vehicles
        # Model configuration
        model: yolov8{self.model_size}.pt
        data: {self.dataset_path / "dataset.yaml"}

        # Training parameters
        epochs: {self.epochs}
        batch_size: {self.batch_size}
        imgsz: {self.img_size}

        # Optimization
        lr0: 0.01
        lrf: 0.01
        momentum: 0.937
        weight_decay: 0.0005
        warmup_epochs: 3.0
        warmup_momentum: 0.8
        warmup_bias_lr: 0.1

        # Augmentation
        hsv_h: 0.015
        hsv_s: 0.7
        hsv_v: 0.4
        degrees: 0.0
        translate: 0.1
        scale: 0.5
        shear: 0.0
        perspective: 0.0
        flipud: 0.0
        fliplr: 0.5
        mosaic: 1.0
        mixup: 0.0
        copy_paste: 0.0

        # Loss coefficients
        box: 7.5
        cls: 0.5
        dfl: 1.5
        pose: 12.0
        kobj: 1.0
        label_smoothing: 0.0

        # Save and validation
        save_period: -1
        save_dir: {self.output_dir}
        cache: False
        device: device=1    
        workers: 0
        project: armored_vehicles_detection
        name: exp
        exist_ok: False
        pretrained: True
        optimizer: auto
        verbose: True
        seed: 0
        deterministic: True
        single_cls: False
        rect: False
        cos_lr: False
        close_mosaic: 10
        resume: False
        amp: True
        fraction: 1.0
        profile: False
        freeze: None
        multi_scale: False

        # Validation
        val: True
        plots: True
"""
        
        config_path = self.output_dir / "train_config.yaml"
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        logging.info(f"Created training config: {config_path}")
        return config_path
    
    def train_model(self):
        """Train the YOLO model."""
        try:
            # Import YOLO (will fail if ultralytics not installed)
            from ultralytics import YOLO
            
            start_time = time.time()
            # Split dataset
            train_images, val_images = self.split_dataset()
            self.create_split_directories(train_images, val_images)
            
            # Create dataset YAML
            dataset_yaml = self.create_dataset_yaml()
            
            # Create training config
            config_path = self.create_training_config()
            
            # Load model
            model = YOLO(f'yolov8{self.model_size}.pt')
            
            logging.info("Starting training...")
            logging.info(f"Model: yolov8{self.model_size}.pt")
            logging.info(f"Dataset: {dataset_yaml}")
            logging.info(f"Epochs: {self.epochs}")
            logging.info(f"Batch size: {self.batch_size}")
            logging.info(f"Image size: {self.img_size}")
            
            # Start training
            results = model.train(
                data=str(dataset_yaml),
                epochs=self.epochs,
                batch=self.batch_size,
                imgsz=self.img_size,
                project=str(self.output_dir.parent),
                name=self.output_dir.name,
                exist_ok=True,
                verbose=True,
                save=True,
                save_period=10,  # Save every 10 epochs
                plots=True,
                device='device=1',
                workers=16
            )
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            elapsed_str = self._format_time(elapsed_time)
            logging.info("Training completed successfully!")
            logging.info(f"Results saved to: {self.output_dir}")
            logging.info(f"Total training time: {elapsed_str}")
            print(f"Total training time: {elapsed_str}")
            
            # Save training summary
            self.save_training_summary(results, elapsed_str)
            
            return results
            
        except ImportError:
            logging.error("ultralytics package not found. Please install it:")
            logging.error("pip install ultralytics")
            return None
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            return None
    
    def save_training_summary(self, results, elapsed_str=None):
        """Save a summary of the training results."""
        summary_path = self.output_dir / "training_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("Armored Vehicles YOLO Training Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: yolov8{self.model_size}.pt\n")
            f.write(f"Classes: {', '.join(self.classes)}\n")
            f.write(f"Number of classes: {len(self.classes)}\n")
            f.write(f"Epochs: {self.epochs}\n")
            f.write(f"Batch size: {self.batch_size}\n")
            f.write(f"Image size: {self.img_size}\n")
            f.write(f"Train/Val split ratio: {self.split_ratio}\n")
            f.write(f"Output directory: {self.output_dir}\n\n")
            if elapsed_str:
                f.write(f"Total training time: {elapsed_str}\n\n")
            if hasattr(results, 'results_dict'):
                f.write("Final Metrics:\n")
                for key, value in results.results_dict.items():
                    f.write(f"  {key}: {value}\n")
        
        logging.info(f"Training summary saved to: {summary_path}")
    
    def validate_model(self, model_path: str = None):
        """Validate the trained model."""
        try:
            from ultralytics import YOLO
            
            if model_path is None:
                # Find the best model
                best_model_path = self.output_dir / "weights" / "best.pt"
                if not best_model_path.exists():
                    logging.error("Best model not found. Please train first.")
                    return None
                model_path = str(best_model_path)
            
            model = YOLO(model_path)
            dataset_yaml = self.dataset_path / "dataset.yaml"
            
            logging.info("Running validation...")
            results = model.val(data=str(dataset_yaml))
            
            logging.info("Validation completed!")
            return results
            
        except ImportError:
            logging.error("ultralytics package not found. Please install it:")
            logging.error("pip install ultralytics")
            return None
        except Exception as e:
            logging.error(f"Validation failed: {str(e)}")
            return None
    
    def _format_time(self, seconds):
        """Format seconds into H:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}:{minutes:02d}:{secs:02d}"

def main():
    parser = argparse.ArgumentParser(description="Train YOLO model for armored vehicles detection")
    parser.add_argument("--dataset_path", type=str, default="annotated_data",
                       help="Path to the annotated dataset")
    parser.add_argument("--output_dir", type=str, default="runs/train",
                       help="Directory to save training results")
    parser.add_argument("--model_size", type=str, default="n", choices=['n', 's', 'm', 'l', 'x'],
                       help="YOLO model size (n=nano, s=small, m=medium, l=large, x=xlarge)")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for training")
    parser.add_argument("--img_size", type=int, default=512,
                       help="Input image size")
    parser.add_argument("--split_ratio", type=float, default=0.8,
                       help="Train/validation split ratio")
    parser.add_argument("--validate_only", action="store_true",
                       help="Only validate an existing model")
    parser.add_argument("--model_path", type=str,
                       help="Path to model for validation (if not using best.pt)")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create trainer
    trainer = ArmoredVehiclesTrainer(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        split_ratio=args.split_ratio
    )
    
    if args.validate_only:
        # Only validate
        trainer.validate_model(args.model_path)
    else:
        # Train and validate
        results = trainer.train_model()
        if results:
            trainer.validate_model()

if __name__ == "__main__":
    main() 