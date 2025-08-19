"""
Grad-CAM processing utilities for the object detection application.
"""

import os
import logging
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
from config.config import DEFAULT_FRAME_SKIP_THRESHOLD, DEFAULT_GRADCAM_CONF_THRESHOLD

from application.models import Detection


class GradCAMProcessor:
    """Handles Grad-CAM processing and visualization"""
    
    def __init__(self, model_path: str, temp_frame_path: str = "temp_frame.jpg"):
        self.temp_frame_path = temp_frame_path
        self.frame_buffer = None
        self.gradcam_buffer = None
        self.last_gradcam_img = None
        self.last_frame_shape = None
        self.frame_skip_counter = 0
        self.frame_skip_threshold = DEFAULT_FRAME_SKIP_THRESHOLD
        
        # Initialize buffers with default values
        self._initialize_buffers()
        
        # Initialize Grad-CAM explainer
        try:
            from YOLOv8_Explainer import yolov8_heatmap
            self.cam_model = yolov8_heatmap(
                weight=model_path,
                conf_threshold=DEFAULT_GRADCAM_CONF_THRESHOLD,
                method="GradCAM",
                show_box=False, 
                renormalize=False
            )
            self.model_loaded = True
            logging.info("Grad-CAM model loaded successfully")
        except ImportError:
            logging.warning("YOLOv8_Explainer not available, Grad-CAM disabled")
            self.model_loaded = False
        except Exception as e:
            logging.warning(f"Failed to load Grad-CAM model: {e}")
            self.model_loaded = False
    
    def _initialize_buffers(self) -> None:
        """Initialize buffer arrays with default values"""
        try:
            # Initialize with a default size that can be resized later
            default_shape = (480, 640, 3)
            self.frame_buffer = np.zeros(default_shape, dtype=np.uint8)
            self.gradcam_buffer = np.zeros(default_shape, dtype=np.uint8)
            self.last_gradcam_img = np.zeros(default_shape, dtype=np.uint8)
            self.last_frame_shape = default_shape
            logging.debug("Grad-CAM buffers initialized with default size")
        except Exception as e:
            logging.warning(f"Failed to initialize Grad-CAM buffers: {e}")
            # Fallback to None, will be handled in processing methods
            self.frame_buffer = None
            self.gradcam_buffer = None
            self.last_gradcam_img = None
            self.last_frame_shape = None
    
    def _ensure_buffer_compatibility(self, frame: np.ndarray) -> None:
        """Ensure all buffers are compatible with the current frame size"""
        if frame is None:
            return
            
        current_shape = frame.shape
        
        # Check if we need to resize buffers
        if (self.frame_buffer is None or 
            self.frame_buffer.shape != current_shape or
            self.last_frame_shape != current_shape):
            
            logging.debug(f"Resizing GradCAM buffers from {self.last_frame_shape} to {current_shape}")
            
            # Resize frame buffer
            if self.frame_buffer is None or self.frame_buffer.shape != current_shape:
                self.frame_buffer = np.zeros_like(frame)
            
            # Resize gradcam buffer
            if self.gradcam_buffer is None or self.gradcam_buffer.shape != current_shape:
                self.gradcam_buffer = np.zeros_like(frame)
            
            # Resize last gradcam image
            if self.last_gradcam_img is None or self.last_gradcam_img.shape != current_shape:
                self.last_gradcam_img = np.zeros_like(frame)
            
            # Update last frame shape
            self.last_frame_shape = current_shape
    
    def get_gradcam_image(self, frame: np.ndarray) -> np.ndarray:
        """Generate Grad-CAM visualization for a frame"""
        if not self.model_loaded:
            logging.debug("Grad-CAM model not loaded, returning original frame")
            return frame.copy() if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Check if frame is valid
        if frame is None:
            logging.warning("Received None frame in get_gradcam_image")
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Ensure buffer compatibility
        self._ensure_buffer_compatibility(frame)
        
        try:
            # Ensure temp directory exists
            temp_dir = os.path.dirname(self.temp_frame_path)
            if temp_dir and not os.path.exists(temp_dir):
                os.makedirs(temp_dir, exist_ok=True)
            
            # Save frame to temp file
            cv2.imwrite(self.temp_frame_path, frame)
            
            # Check if cam_model is available
            if not hasattr(self, 'cam_model') or self.cam_model is None:
                logging.warning("GradCAM model not available")
                return frame.copy()
            
            try:
                cam_images = self.cam_model(img_path=self.temp_frame_path)
            except Exception as e:
                logging.warning(f"GradCAM model inference failed: {e}")
                return frame.copy()
            
            gradcam_img = frame.copy()
            
            if isinstance(cam_images, list) and len(cam_images) > 0:
                img_candidate = cam_images[0]
                if isinstance(img_candidate, Image.Image):
                    gradcam_img = np.array(img_candidate.convert("RGB"))
                elif isinstance(img_candidate, np.ndarray):
                    gradcam_img = img_candidate
                
                # Ensure the GradCAM image matches the frame dimensions
                if isinstance(gradcam_img, np.ndarray) and gradcam_img.ndim >= 2:
                    if gradcam_img.shape[:2] != frame.shape[:2]:
                        gradcam_img = cv2.resize(gradcam_img, (frame.shape[1], frame.shape[0]))
                else:
                    gradcam_img = frame.copy()
            else:
                gradcam_img = frame.copy()
                
        except Exception as e:
            logging.warning(f"Grad-CAM generation failed: {e}")
            gradcam_img = frame.copy()
        
        return gradcam_img
    
    def process_gradcam(self, frame: np.ndarray, detections: List[Detection], 
                       in_box_only: bool = True) -> np.ndarray:
        """Process Grad-CAM for multiple detections with improved synchronization"""
        if not self.model_loaded:
            logging.debug("Grad-CAM model not loaded, returning original frame")
            return frame.copy() if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Check if frame is valid
        if frame is None:
            logging.warning("Received None frame in process_gradcam")
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Ensure buffer compatibility
        self._ensure_buffer_compatibility(frame)
        
        self.frame_skip_counter += 1
        
        # Process GradCAM more frequently for better synchronization
        if self.frame_skip_counter % self.frame_skip_threshold == 0:
            logging.debug(f"Processing GradCAM for frame (counter: {self.frame_skip_counter}, threshold: {self.frame_skip_threshold})")
            try:
                if in_box_only and detections:
                    # Process GradCAM for the entire frame first
                    full_gradcam = self.get_gradcam_image(frame)
                    
                    # Create output image starting with the original frame
                    gradcam_img = frame.copy()
                    
                    # Apply GradCAM only to detection regions
                    for detection in detections:
                        if detection.box is not None and len(detection.box) == 4:
                            x1, y1, x2, y2 = detection.box
                            
                            # Ensure coordinates are within bounds
                            x1 = max(0, min(x1, frame.shape[1] - 1))
                            y1 = max(0, min(y1, frame.shape[0] - 1))
                            x2 = max(x1 + 1, min(x2, frame.shape[1]))
                            y2 = max(y1 + 1, min(y2, frame.shape[0]))
                            
                            # Extract regions
                            frame_roi = frame[y1:y2, x1:x2]
                            gradcam_roi = full_gradcam[y1:y2, x1:x2]
                            
                            # Ensure ROI shapes match
                            if frame_roi.shape == gradcam_roi.shape and frame_roi.size > 0:
                                gradcam_img[y1:y2, x1:x2] = gradcam_roi
                                logging.debug(f"Applied GradCAM to detection box: ({x1}, {y1}) to ({x2}, {y2})")
                else:
                    # Use full GradCAM image
                    gradcam_img = self.get_gradcam_image(frame)
                
                # Update the last processed image
                self.last_gradcam_img = gradcam_img.copy()
                logging.debug(f"Updated last GradCAM image with shape: {self.last_gradcam_img.shape}")
                
            except Exception as e:
                logging.warning(f"Grad-CAM processing failed: {e}")
                gradcam_img = self.last_gradcam_img if self.last_gradcam_img is not None else frame.copy()
        else:
            # Use the last processed GradCAM image, but ensure it matches current frame
            if (self.last_gradcam_img is not None and 
                self.last_gradcam_img.shape == frame.shape):
                gradcam_img = self.last_gradcam_img.copy()
                logging.debug(f"Using cached GradCAM image (frame {self.frame_skip_counter})")
            else:
                # If shapes don't match, fall back to original frame
                gradcam_img = frame.copy()
                logging.debug(f"Shape mismatch, using original frame (frame {self.frame_skip_counter})")
        
        return gradcam_img
    
    def set_frame_skip_threshold(self, threshold: int) -> None:
        """Set the frame skip threshold for performance tuning"""
        self.frame_skip_threshold = max(1, threshold)  # Ensure minimum of 1
        logging.info(f"Grad-CAM frame skip threshold set to {self.frame_skip_threshold}")
    
    def is_model_loaded(self) -> bool:
        """Check if the Grad-CAM model is successfully loaded"""
        return self.model_loaded
    
    def cleanup(self) -> None:
        """Clean up Grad-CAM resources"""
        try:
            if os.path.exists(self.temp_frame_path):
                try:
                    os.remove(self.temp_frame_path)
                except Exception as e:
                    logging.debug(f"Could not remove temp file: {e}")
        except Exception as e:
            logging.debug(f"Error during temp file cleanup: {e}")
        
        try:
            self.frame_buffer = None
            self.gradcam_buffer = None
            self.last_gradcam_img = None
            self.last_frame_shape = None
            self.cam_model = None
        except Exception as e:
            logging.debug(f"Error during buffer cleanup: {e}")
        
        logging.debug("Grad-CAM resources cleaned up") 