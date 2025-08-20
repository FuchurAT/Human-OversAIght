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
            # Patch torch.load to handle PyTorch 2.6+ security restrictions
            import torch
            original_load = torch.load
            
            def patched_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            
            torch.load = patched_load
            
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
        
        try:
            # Save frame to temp file
            cv2.imwrite(self.temp_frame_path, frame)
            
            # Get GradCAM output
            cam_images = self.cam_model(img_path=self.temp_frame_path)
            
            # Start with original frame
            gradcam_img = frame.copy()
            
            if isinstance(cam_images, list) and len(cam_images) > 0:
                img_candidate = cam_images[0]
                
                if isinstance(img_candidate, Image.Image):
                    # Convert PIL image to numpy array
                    gradcam_heatmap = np.array(img_candidate.convert("RGB"))
                elif isinstance(img_candidate, np.ndarray):
                    gradcam_heatmap = img_candidate
                else:
                    return frame.copy()
                
                # Ensure the GradCAM heatmap matches the frame dimensions
                if gradcam_heatmap.shape[:2] != frame.shape[:2]:
                    gradcam_heatmap = cv2.resize(gradcam_heatmap, (frame.shape[1], frame.shape[0]))
                
                # Convert to float for proper blending
                frame_float = frame.astype(np.float32) / 255.0
                heatmap_float = gradcam_heatmap.astype(np.float32) / 255.0
                
                # Blend the heatmap with the original frame
                # Use alpha blending: result = alpha * heatmap + (1 - alpha) * frame
                alpha = 0.6  # Adjust this value to control overlay intensity
                blended = alpha * heatmap_float + (1 - alpha) * frame_float
                
                # Convert back to uint8
                gradcam_img = (blended * 255).astype(np.uint8)
                
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
        
        # Convert to grayscale and back to BGR (this was in the working code)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Use gradcam buffer to avoid repeated allocations
        if self.gradcam_buffer is None or self.gradcam_buffer.shape != frame.shape:
            self.gradcam_buffer = np.zeros_like(frame)
        
        gradcam_img = self.gradcam_buffer.copy()
        
        # Skip Grad-CAM processing for performance (every N frames)
        self.frame_skip_counter += 1
        if self.frame_skip_counter % self.frame_skip_threshold == 0:
            try:
                # Sort detections by confidence, descending
                sorted_detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
                selected_boxes = []
                selected_confs = []
                selected_classes = []
                
                for detection in detections:
                    if detection.box is not None and len(detection.box) == 4:
                        box = detection.box
                        conf = detection.confidence
                        class_id = detection.class_id
                        
                        # Check overlap with already selected boxes
                        overlap = False
                        for sel_box in selected_boxes:
                            # Simple IoU check (you may need to implement this)
                            if self._iou(box, sel_box) > 0.5:  # Default IoU threshold
                                overlap = True
                                break
                        if not overlap:
                            selected_boxes.append(box)
                            selected_confs.append(conf)
                            selected_classes.append(class_id)
                
                # Overlay Grad-CAM for each selected box
                if in_box_only:
                    gradcam_img = frame.copy()
                    for box, conf, class_id in zip(selected_boxes, selected_confs, selected_classes):
                        single_gradcam = self.get_gradcam_image(frame)
                        x1, y1, x2, y2 = box
                        gradcam_img[y1:y2, x1:x2] = single_gradcam[y1:y2, x1:x2]
                else:
                    gradcam_img = np.zeros_like(frame)
                    for box, conf, class_id in zip(selected_boxes, selected_confs, selected_classes):
                        single_gradcam = self.get_gradcam_image(frame)
                        gradcam_img = np.maximum(gradcam_img, single_gradcam)
                
                self.last_gradcam_img = gradcam_img
                
            except Exception as e:
                logging.warning(f"Grad-CAM failed: {e}")
                gradcam_img = self.last_gradcam_img if self.last_gradcam_img is not None else self.gradcam_buffer.copy()
        else:
            gradcam_img = self.last_gradcam_img if self.last_gradcam_img is not None else self.gradcam_buffer.copy()
        
        return gradcam_img
    
    def set_frame_skip_threshold(self, threshold: int) -> None:
        """Set the frame skip threshold for performance tuning"""
        self.frame_skip_threshold = max(1, threshold)  # Ensure minimum of 1
        #logging.info(f"Grad-CAM frame skip threshold set to {self.frame_skip_threshold}")
    
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
    
    def _iou(self, box1, box2):
        """Calculate Intersection over Union between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection coordinates
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        # Check if there is intersection
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        # Calculate areas
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0 