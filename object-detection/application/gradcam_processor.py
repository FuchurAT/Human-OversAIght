"""
Grad-CAM processing utilities for the object detection application.
"""

import os
import logging
import cv2
import numpy as np
from PIL import Image
from typing import List
from config.config import DEFAULT_FRAME_SKIP_THRESHOLD, DEFAULT_GRADCAM_CONF_THRESHOLD

from application.models import Detection


class GradCAMProcessor:
    """Handles Grad-CAM processing and visualization"""
    
    def __init__(self, model_path: str, temp_frame_path: str = "temp_frame.jpg"):
        self.temp_frame_path = temp_frame_path
        self.frame_buffer = None
        self.gradcam_buffer = None
        self.last_gradcam_img = None
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
                show_box=True,
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
            logging.debug("Grad-CAM buffers initialized with default size")
        except Exception as e:
            logging.warning(f"Failed to initialize Grad-CAM buffers: {e}")
            # Fallback to None, will be handled in processing methods
            self.frame_buffer = None
            self.gradcam_buffer = None
            self.last_gradcam_img = None
    
    def get_gradcam_image(self, frame: np.ndarray) -> np.ndarray:
        """Generate Grad-CAM visualization for a frame"""
        if not self.model_loaded:
            logging.debug("Grad-CAM model not loaded, returning original frame")
            return frame.copy() if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Check if frame is valid
        if frame is None:
            logging.warning("Received None frame in get_gradcam_image")
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        try:
            if self.frame_buffer is None or self.frame_buffer.shape != frame.shape:
                logging.debug(f"Initializing frame buffer with shape {frame.shape}")
                self.frame_buffer = np.zeros_like(frame)
                # Also update other buffers to match
                if self.gradcam_buffer is None or self.gradcam_buffer.shape != frame.shape:
                    self.gradcam_buffer = np.zeros_like(frame)
                if self.last_gradcam_img is None or self.last_gradcam_img.shape != frame.shape:
                    self.last_gradcam_img = np.zeros_like(frame)
            
            # Ensure temp directory exists
            temp_dir = os.path.dirname(self.temp_frame_path)
            if temp_dir and not os.path.exists(temp_dir):
                os.makedirs(temp_dir, exist_ok=True)
            
            cv2.imwrite(self.temp_frame_path, frame)
            
            # Check if cam_model is available
            if not hasattr(self, 'cam_model') or self.cam_model is None:
                logging.warning("GradCAM model not available")
                return self.frame_buffer.copy() if self.frame_buffer is not None else np.zeros_like(frame)
            
            try:
                cam_images = self.cam_model(img_path=self.temp_frame_path)
            except Exception as e:
                logging.warning(f"GradCAM model inference failed: {e}")
                return self.frame_buffer.copy() if self.frame_buffer is not None else np.zeros_like(frame)
            
            gradcam_img = self.frame_buffer.copy() if self.frame_buffer is not None else np.zeros_like(frame)
            
            if isinstance(cam_images, list) and len(cam_images) > 0:
                img_candidate = cam_images[0]
                if isinstance(img_candidate, Image.Image):
                    gradcam_img = np.array(img_candidate.convert("RGB"))
                elif isinstance(img_candidate, np.ndarray):
                    gradcam_img = img_candidate
                
                if isinstance(gradcam_img, np.ndarray) and gradcam_img.ndim >= 2:
                    if gradcam_img.shape[:2] != frame.shape[:2]:
                        gradcam_img = cv2.resize(gradcam_img, (frame.shape[1], frame.shape[0]))
                else:
                    gradcam_img = self.frame_buffer.copy() if self.frame_buffer is not None else np.zeros_like(frame)
        except Exception as e:
            logging.warning(f"Grad-CAM generation failed: {e}")
            gradcam_img = self.frame_buffer.copy() if self.frame_buffer is not None else np.zeros_like(frame)
        
        return gradcam_img
    
    def process_gradcam(self, frame: np.ndarray, detections: List[Detection], 
                       in_box_only: bool = True) -> np.ndarray:
        """Process Grad-CAM for multiple detections"""
        if not self.model_loaded:
            logging.debug("Grad-CAM model not loaded, returning original frame")
            return frame.copy() if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Check if frame is valid
        if frame is None:
            logging.warning("Received None frame in process_gradcam")
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        self.frame_skip_counter += 1
        
        if self.frame_skip_counter % self.frame_skip_threshold == 0:
            try:
                if in_box_only:
                    gradcam_img = frame.copy()
                    for detection in detections:
                        single_gradcam = self.get_gradcam_image(frame)
                        x1, y1, x2, y2 = detection.box
                        gradcam_img[y1:y2, x1:x2] = single_gradcam[y1:y2, x1:x2]
                else:
                    gradcam_img = np.zeros_like(frame)
                    for detection in detections:
                        single_gradcam = self.get_gradcam_image(frame)
                        gradcam_img = np.maximum(gradcam_img, single_gradcam)
                
                self.last_gradcam_img = gradcam_img
            except Exception as e:
                logging.warning(f"Grad-CAM processing failed: {e}")
                gradcam_img = self.last_gradcam_img if self.last_gradcam_img is not None else (self.gradcam_buffer.copy() if self.gradcam_buffer is not None else np.zeros_like(frame))
        else:
            gradcam_img = self.last_gradcam_img if self.last_gradcam_img is not None else (self.gradcam_buffer.copy() if self.gradcam_buffer is not None else np.zeros_like(frame))
        
        return gradcam_img
    
    def set_frame_skip_threshold(self, threshold: int) -> None:
        """Set the frame skip threshold for performance tuning"""
        self.frame_skip_threshold = threshold
        logging.info(f"Grad-CAM frame skip threshold set to {threshold}")
    
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
            self.cam_model = None
        except Exception as e:
            logging.debug(f"Error during buffer cleanup: {e}")
        
        logging.debug("Grad-CAM resources cleaned up") 