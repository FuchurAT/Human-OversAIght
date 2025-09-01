"""
Main application class for video inference with YOLO.
"""

import cv2
import logging
import os
import time
import numpy as np
import torch
import gc
import io
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from config.config import (
    DEFAULT_BOX_THRESHOLD, DEFAULT_GRADCAM_ALPHA, LEGEND_WINDOW_WIDTH, 
    LEGEND_WINDOW_HEIGHT, LEGEND_FONT_SCALE, LEGEND_LINE_HEIGHT, LEGEND_BOX_PADDING,
    DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT, MIN_WINDOW_SIZE, DEFAULT_FPS,
    DEFAULT_VIDEO_WIDTH, DEFAULT_VIDEO_HEIGHT, DEFAULT_VIDEO_SCALE_MODE,
    DEFAULT_VIDEO_MAINTAIN_ASPECT_RATIO, DEFAULT_VIDEO_CENTER_ON_SCREEN, DEFAULT_VIDEO_SCALE_MULTIPLIER,
    SCREEN_CONFIG, DEFAULT_NDI_ENABLED, DEFAULT_NDI_SOURCE_NAME, DEFAULT_NDI_GROUP_NAME,
    DEFAULT_NDI_VIDEO_FORMAT, DEFAULT_NDI_FRAME_RATE, 
    DEFAULT_NDI_VIDEO_WIDTH, DEFAULT_NDI_VIDEO_HEIGHT, COUNTER_CONFIG
)

from application.models import Detection, DisplayConfig
from application.memory_manager import MemoryManager
from application.box_manager import BoxManager
from application.gradcam_processor import GradCAMProcessor
from application.visualizer import DetectionVisualizer
from application.color_manager import ColorManager
from application.button_handler import ButtonHandler
from application.ndi_sender import create_ndi_sender
from application.count_handler import CountHandler

try:
    from screeninfo import get_monitors
    logging.info("Successfully imported screeninfo.get_monitors")
except ImportError:
    get_monitors = None
    logging.warning("screeninfo package not found. Multi-monitor support will be limited.")
    logging.warning("Install with: pip install screeninfo")

from config.classes import CLASSES


class VideoInferenceApp:
    """Main application class for video inference with YOLO"""
    
    def __init__(self, video_path: str, model_path: str, box_threshold: float = DEFAULT_BOX_THRESHOLD, 
                 app_id: str = None, screen_id: int = None):
        # Handle both single path (backward compatibility) and multiple paths
        if isinstance(video_path, list):
            self.video_folders = video_path
            self.current_folder_index = 0
            self.video_path = video_path[0]  # Current active folder
        else:
            # Backward compatibility: convert single path to list
            self.video_folders = [video_path]
            self.current_folder_index = 0
            self.video_path = video_path
        
        # Ensure video_folders is always a list and contains valid paths
        if not self.video_folders:
            raise ValueError("No video folders provided")
        
        # Validate that all folders exist
        for i, folder in enumerate(self.video_folders):
            if not os.path.exists(folder):
                logging.warning(f"Video folder {i} does not exist: {folder}")
        
        # Set current folder to first valid folder
        for i, folder in enumerate(self.video_folders):
            if os.path.exists(folder):
                self.current_folder_index = i
                self.video_path = folder
                break
        else:
            raise ValueError("No valid video folders found")
        
        logging.info(f"Initialized with {len(self.video_folders)} video folders")
        logging.info(f"Current folder: {self.current_folder_index + 1}/{len(self.video_folders)} - {self.video_path}")
        
        self.model_path = model_path
        self.output_path = ""
        self.box_threshold = box_threshold
        self.max_frames = 0
        self.app_id = app_id or 'default'
        self.screen_id = screen_id or 0
        
        # Initialize components
        self.display_config = DisplayConfig()
        # Enable key features by default for better user experience
        self.display_config.gradcam_enabled = True
        self.display_config.show_legend = True
        self.display_config.show_fps_info = True
        self.display_config.enable_glitches = False
        self.display_config.blur_boxes = True
        self.display_config.show_enemy = False
        self.display_config.solid_border = False
        
        self.memory_manager = MemoryManager()
        self.box_manager = BoxManager()
        
        # Initialize GradCAM processor with error handling
        try:
            self.gradcam_processor = GradCAMProcessor(model_path)
            logging.info("GradCAM processor initialized successfully")
        except Exception as e:
            logging.warning(f"Failed to initialize GradCAM processor: {e}")
            # Create a fallback processor that just returns the original frame
            self.gradcam_processor = None
        
        # Initialize model
        self._initialize_model()
        
        # Initialize visualizer
        self.visualizer = DetectionVisualizer(self.display_config, list(CLASSES.keys()))
        
        # Get app-specific NDI configuration
        ndi_config = self._get_ndi_config()
        
        # Initialize NDI sender with app-specific configuration
        self.ndi_sender = create_ndi_sender(
            source_name=ndi_config['source_name'],
            group_name=ndi_config['group_name'],
            video_format=ndi_config['video_format'],
            frame_rate=ndi_config['frame_rate'],
            video_width=ndi_config['video_width'],
            video_height=ndi_config['video_height']
        )
        
        # Update display config with NDI settings
        self.display_config.ndi_enabled = ndi_config['enabled']
        self.display_config.ndi_source_name = ndi_config['source_name']
        self.display_config.ndi_group_name = ndi_config['group_name']
        self.display_config.ndi_video_format = ndi_config['video_format']
        self.display_config.ndi_frame_rate = ndi_config['frame_rate']
        self.display_config.ndi_video_width = ndi_config['video_width']
        self.display_config.ndi_video_height = ndi_config['video_height']
        
        # Initialize button handler (will be set up by multi-app manager)
        self.button_handler = None
        
        # Initialize count handler with global count file
        self.count_handler = CountHandler("count.txt")
        
        # Button action flags
        self._button_next_video_signal = False
        self._button_next_folder_signal = False
        
        # Screen configuration
        self.screen_config = self._get_screen_config()
        
        # Verify model is loaded
        if hasattr(self, 'model') and self.model is not None:
            logging.info(f"Model successfully loaded and ready for inference")
        else:
            logging.error("Model failed to load during initialization")
        
        # Log initialization status
        logging.info(f"VideoInferenceApp initialized successfully")
        logging.info(f"  App ID: {self.app_id}")
        logging.info(f"  Screen ID: {self.screen_id}")
        logging.info(f"  Model: {self.model_path}")
        logging.info(f"  Video folders: {len(self.video_folders)} folders available")
        logging.info(f"  Current folder: {self.video_path}")
        if len(self.video_folders) > 1:
            logging.info(f"  All folders: {self.video_folders}")
        logging.info(f"  Box threshold: {self.box_threshold}")
        logging.info(f"  GradCAM enabled: {self.gradcam_processor.is_model_loaded() if self.gradcam_processor is not None else False}")
        logging.info(f"  Screen config: {self.screen_config}")
        logging.info(f"  NDI enabled: {self.display_config.ndi_enabled}")
        if self.display_config.ndi_enabled:
            logging.info(f"  NDI source: {self.display_config.ndi_source_name}")
            logging.info(f"  NDI group: {self.display_config.ndi_group_name}")
            logging.info(f"  NDI format: {self.display_config.ndi_video_format}")
            logging.info(f"  NDI resolution: {self.display_config.ndi_video_width}x{self.display_config.ndi_video_height}")
            logging.info(f"  NDI frame rate: {self.display_config.ndi_frame_rate}")
    
    def _get_screen_config(self) -> dict:
        """Get screen configuration for this application"""
        logging.info(f"Getting screen config for screen_id: {self.screen_id}")
        logging.info(f"Available screen configs: {list(SCREEN_CONFIG.keys())}")
        
        if self.screen_id in SCREEN_CONFIG:
            config = SCREEN_CONFIG[self.screen_id].copy()
            logging.info(f"Found screen config for screen_id {self.screen_id}: {config}")
            return config
        else:
            # Fallback to default screen config
            logging.warning(f"Screen ID {self.screen_id} not found in SCREEN_CONFIG, using defaults")
            fallback_config = {
                'width': DEFAULT_WINDOW_WIDTH,
                'height': DEFAULT_WINDOW_HEIGHT,
                'x_offset': 0,
                'y_offset': 0,
                'scale_mode': DEFAULT_VIDEO_SCALE_MODE,
                'scale_multiplier': DEFAULT_VIDEO_SCALE_MULTIPLIER,
                'maintain_aspect_ratio': DEFAULT_VIDEO_MAINTAIN_ASPECT_RATIO,
                'center_video': DEFAULT_VIDEO_CENTER_ON_SCREEN
            }
            logging.info(f"Using fallback config: {fallback_config}")
            return fallback_config
    
    def _get_ndi_config(self) -> dict:
        """Get NDI configuration for this application"""
        from config.config import APPLICATIONS
        
        if self.app_id in APPLICATIONS and 'ndi' in APPLICATIONS[self.app_id]:
            ndi_config = APPLICATIONS[self.app_id]['ndi'].copy()
            logging.info(f"Found NDI config for app {self.app_id}: {ndi_config}")
            return ndi_config
        else:
            # Fallback to default NDI config
            logging.warning(f"NDI config not found for app {self.app_id}, using defaults")
            fallback_config = {
                'enabled': DEFAULT_NDI_ENABLED,
                'source_name': f"{DEFAULT_NDI_SOURCE_NAME}-{self.app_id}",
                'group_name': DEFAULT_NDI_GROUP_NAME,
                'video_format': DEFAULT_NDI_VIDEO_FORMAT,
                'frame_rate': DEFAULT_NDI_FRAME_RATE,
                'video_width': DEFAULT_NDI_VIDEO_WIDTH,
                'video_height': DEFAULT_NDI_VIDEO_HEIGHT
            }
            logging.info(f"Using fallback NDI config: {fallback_config}")
            return fallback_config
    
    def set_button_handler(self, button_handler: ButtonHandler) -> None:
        """Set the button handler for this application instance"""
        self.button_handler = button_handler
        logging.info(f"Button handler set for app {self.app_id}")
    
    def _initialize_button_handler(self) -> None:
        """Initialize button handler for backward compatibility (single app mode)"""
        if self.button_handler is None:
            from application.button_handler import ButtonHandler
            self.button_handler = ButtonHandler(self)
            self.button_handler.start_serial_monitoring()
            logging.info(f"Button handler initialized for app {self.app_id} (backward compatibility mode)")
            logging.info(f"Button handler app instances: {list(self.button_handler.app_instances.keys())}")
        else:
            logging.info(f"Button handler already exists for app {self.app_id}")
            logging.info(f"Button handler app instances: {list(self.button_handler.app_instances.keys())}")
    
    def signal_next_video(self) -> None:
        """Signal that the application should move to the next video"""
        self._button_next_video_signal = True
        logging.info(f"Next video signal set for app {self.app_id}")
    
    def signal_next_folder(self) -> None:
        """Signal that the application should move to the next folder"""
        logging.info(f"signal_next_folder called for app {self.app_id}")
        logging.info(f"Current folder index: {self.current_folder_index}")
        logging.info(f"Total folders: {len(self.video_folders)}")
        logging.info(f"Available folders: {self.video_folders}")
        
        self._button_next_folder_signal = True
        logging.info(f"Next folder signal set for app {self.app_id}")
        
        # Also try to switch immediately for testing
        if len(self.video_folders) > 1:
            logging.info("Attempting immediate folder switch for testing...")
            self.switch_to_next_folder()
        else:
            logging.info("Only one folder available, no switching needed")
    
    def switch_to_next_folder(self) -> None:
        """Switch to the next video folder in the list"""
        if len(self.video_folders) <= 1:
            logging.info(f"Only one folder available, no switching needed")
            return
        
        old_folder = self.video_path
        old_index = self.current_folder_index
        old_video_files = self.refresh_video_files() if os.path.exists(old_folder) else []
        
        logging.info(f"Switching from folder {old_index + 1}/{len(self.video_folders)}: {old_folder}")
        logging.info(f"Old folder had {len(old_video_files)} video files")
        
        # Find next valid folder
        attempts = 0
        max_attempts = len(self.video_folders)
        
        while attempts < max_attempts:
            # Move to next folder index
            self.current_folder_index = (self.current_folder_index + 1) % len(self.video_folders)
            new_folder = self.video_folders[self.current_folder_index]
            
            logging.info(f"Trying folder {self.current_folder_index + 1}/{len(self.video_folders)}: {new_folder}")
            
            # Check if this folder exists and has videos
            if os.path.exists(new_folder):
                video_files = [f for f in os.listdir(new_folder) if f.endswith('.mp4')]
                if video_files:
                    self.video_path = new_folder
                    logging.info(f"Successfully switched to folder {self.current_folder_index + 1}/{len(self.video_folders)}: {self.video_path}")
                    logging.info(f"New folder has {len(video_files)} video files")
                    logging.info(f"New video files: {video_files[:5]}")
                    
                    # Verify this is actually a different folder
                    if new_folder == old_folder:
                        logging.warning("Switched to same folder - this shouldn't happen")
                    else:
                        logging.info("Successfully switched to different folder")
                    
                    # Trigger visual feedback for folder switch
                    if hasattr(self, 'visualizer') and self.visualizer:
                        self.visualizer.trigger_key_feedback(ord('f'))
                    
                    # After switching folders, immediately signal next video to show content from new folder
                    logging.info(f"Auto-triggering next video after folder switch from {old_folder} to {self.video_path}")
                    self._button_next_video_signal = True
                    # Also call the signal method directly to ensure it's processed
                    self.signal_next_video()
                    return
                else:
                    logging.warning(f"Folder {new_folder} exists but contains no .mp4 files")
            else:
                logging.warning(f"Folder {new_folder} does not exist")
            
            attempts += 1
            
            # If we've tried all folders and none work, revert to original
            if attempts >= max_attempts:
                logging.error(f"Could not find any valid folder after {max_attempts} attempts, reverting to original")
                self.current_folder_index = old_index
                self.video_path = old_folder
                return
    
    def force_refresh_video_files(self) -> list:
        """Force refresh the video file list and return the new list"""
        logging.info(f"Force refreshing video files from current folder: {self.video_path}")
        new_files = self.refresh_video_files()
        logging.info(f"Force refresh complete: {len(new_files)} files found")
        return new_files
    
    def log_current_state(self) -> None:
        """Log the current state for debugging purposes"""
        logging.info(f"=== Current State for App {self.app_id} ===")
        logging.info(f"Current folder index: {self.current_folder_index}")
        logging.info(f"Total folders: {len(self.video_folders)}")
        logging.info(f"Current video path: {self.video_path}")
        logging.info(f"All video folders: {self.video_folders}")
        
        # Check current folder contents
        try:
            if os.path.exists(self.video_path):
                video_files = [f for f in os.listdir(self.video_path) if f.endswith('.mp4')]
                logging.info(f"Current folder contains {len(video_files)} video files")
                if video_files:
                    logging.info(f"Sample video files: {video_files[:3]}")
            else:
                logging.warning(f"Current video path does not exist: {self.video_path}")
        except Exception as e:
            logging.error(f"Error checking current folder contents: {e}")
        
        logging.info(f"Next video signal: {getattr(self, '_button_next_video_signal', False)}")
        logging.info(f"Next folder signal: {getattr(self, '_button_next_folder_signal', False)}")
        logging.info("=== End State Log ===")
    
    def refresh_video_files(self) -> list:
        """Refresh the list of video files from the current folder"""
        try:
            logging.info(f"Refreshing video files from: {self.video_path}")
            if os.path.exists(self.video_path):
                all_files = os.listdir(self.video_path)
                logging.info(f"All files in folder: {len(all_files)} files")
                logging.info(f"Sample files: {all_files[:5] if all_files else 'None'}")
                
                video_files = [f for f in all_files if f.endswith('.mp4')]
                logging.info(f"Video files found: {len(video_files)} .mp4 files")
                if video_files:
                    logging.info(f"Video file names: {video_files[:5]}")
                else:
                    logging.warning(f"No .mp4 files found in {self.video_path}")
                return video_files
            else:
                logging.warning(f"Current video path does not exist: {self.video_path}")
                return []
        except Exception as e:
            logging.error(f"Error refreshing video files: {e}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def get_current_folder_info(self) -> dict:
        """Get information about the current folder and available folders"""
        return {
            'current_index': self.current_folder_index,
            'total_folders': len(self.video_folders),
            'current_path': self.video_path,
            'all_paths': self.video_folders.copy()
        }
    
    def get_total_video_count(self) -> int:
        """Get the total number of videos across all folders"""
        total_count = 0
        for folder in self.video_folders:
            try:
                if os.path.exists(folder):
                    mp4_files = [f for f in os.listdir(folder) if f.endswith('.mp4')]
                    total_count += len(mp4_files)
            except Exception as e:
                logging.warning(f"Error counting videos in folder {folder}: {e}")
        return total_count
    
    def get_folder_video_counts(self) -> dict:
        """Get video counts for each folder"""
        folder_counts = {}
        for i, folder in enumerate(self.video_folders):
            try:
                if os.path.exists(folder):
                    mp4_files = [f for f in os.listdir(folder) if f.endswith('.mp4')]
                    folder_counts[i] = {
                        'path': folder,
                        'count': len(mp4_files),
                        'is_current': i == self.current_folder_index
                    }
                else:
                    folder_counts[i] = {
                        'path': folder,
                        'count': 0,
                        'is_current': i == self.current_folder_index,
                        'error': 'Folder not found'
                    }
            except Exception as e:
                folder_counts[i] = {
                    'path': folder,
                    'count': 0,
                    'is_current': i == self.current_folder_index,
                    'error': str(e)
                }
        return folder_counts
    
    def _initialize_model(self) -> None:
        """Initialize the YOLO model with proper error handling"""
        try:
            from ultralytics import YOLO
            logging.info("Successfully imported ultralytics.YOLO")
        except ImportError:
            logging.error("ultralytics package not found. Please install it: pip install ultralytics")
            raise
        
        if not Path(self.model_path).exists():
            logging.error(f"Model not found: {self.model_path}")
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        logging.info(f"Model file exists: {self.model_path}")
        logging.info(f"Model file size: {Path(self.model_path).stat().st_size} bytes")
        
        # Add safe globals for PyTorch serialization
        self._add_safe_globals()
        
        # Create YOLO model with verbose=False to suppress output
        logging.info("Creating YOLO model instance...")
        self.model = YOLO(self.model_path, verbose=False)
        logging.info(f"Successfully loaded model: {self.model_path}")
        
        # Test the model with a dummy frame
        try:
            logging.info("Testing model with dummy frame...")
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            test_result = self.model(dummy_frame, conf=0.15, verbose=False)
            logging.info(f"Model test successful, result type: {type(test_result)}")
            if hasattr(test_result, 'boxes'):
                logging.info(f"Model has boxes attribute: {test_result.boxes is not None}")
        except Exception as e:
            logging.warning(f"Model test failed: {e}")
            # Don't raise here, just log the warning
    
    def _add_safe_globals(self) -> None:
        """Add safe globals for PyTorch serialization"""
        from torch.nn.modules.container import Sequential, ModuleList
        from torch.nn.modules.conv import Conv2d
        from torch.nn.modules.batchnorm import BatchNorm2d
        from torch.nn.modules.activation import SiLU
        from torch.nn.modules.pooling import MaxPool2d
        from torch.nn.modules.upsampling import Upsample
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.nn.modules.conv import Conv, Concat
        from ultralytics.nn.modules.block import C2f, SPPF, Bottleneck, DFL
        from ultralytics.nn.modules.head import Detect

        torch.serialization.add_safe_globals([
            DetectionModel, Detect, Sequential, Conv, Conv2d, MaxPool2d, 
            BatchNorm2d, SiLU, C2f, Bottleneck, DFL, ModuleList, SPPF, Upsample, Concat
        ])
    
    def _process_frame(self, frame: np.ndarray) -> Tuple[List[Detection], float]:
        """Process a single frame and return detections and inference time"""
        # Check if frame is valid
        if frame is None:
            logging.warning("Received None frame in _process_frame")
            return [], 0.0
        
        # Check if model is loaded
        if not hasattr(self, 'model') or self.model is None:
            logging.error("Model not initialized in _process_frame")
            return [], 0.0
        
        # Memory cleanup
        self.memory_manager.cleanup_memory()
        
        # Run inference with output suppression
        inf_start = time.time()
        try:
            # Temporarily redirect stdout and stderr to suppress YOLO output
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                results = self.model(frame, conf=0.15, verbose=False)
            inf_time = (time.time() - inf_start) * 1000  # ms
            
            # Debug: Check results
            if results is None or len(results) == 0:
                logging.debug("Model returned no results")
                return [], inf_time
            
            # Extract detections
            boxes = results[0].boxes
            detections = []
            
            if boxes is not None and len(boxes) > 0:
                logging.debug(f"Found {len(boxes)} raw detections")
                for box in boxes:
                    conf = float(box.conf[0])
                    if conf < self.box_threshold:
                        continue
                    class_id = int(box.cls[0])
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    detections.append(Detection(tuple(xyxy), conf, class_id))
                
                logging.debug(f"Filtered to {len(detections)} detections above threshold {self.box_threshold}")
            else:
                logging.debug("No boxes found in results")
            
            return detections, inf_time
            
        except Exception as e:
            logging.error(f"Error during model inference: {e}")
            import traceback
            logging.error(f"Inference traceback: {traceback.format_exc()}")
            return [], 0.0
    
    def _validate_frame(self, frame: np.ndarray, name: str = "frame") -> bool:
        """Validate that a frame is valid and usable"""
        if frame is None:
            logging.error(f"{name} is None")
            return False
        if not hasattr(frame, 'shape'):
            logging.error(f"{name} has no shape attribute")
            return False
        if len(frame.shape) != 3:
            logging.error(f"{name} has invalid shape: {frame.shape}")
            return False
        if frame.shape[2] != 3:
            logging.error(f"{name} has invalid channels: {frame.shape[2]}")
            return False
        if frame.size == 0:
            logging.error(f"{name} has size 0")
            return False
        return True
    
    def _prepare_display_frame(self, frame: np.ndarray, detections: List[Detection], 
                               gradcam_img: np.ndarray) -> np.ndarray:
        """Prepare the frame for display with all overlays"""
        # Check if frame is valid
        if not self._validate_frame(frame, "input frame"):
            logging.warning("Received invalid frame in _prepare_display_frame")
            # Return a black frame as fallback
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Debug: Log detection count and frame info
        logging.debug(f"Preparing display frame with {len(detections)} detections")
        logging.debug(f"Input frame shape: {frame.shape}, dtype: {frame.dtype}")
        logging.debug(f"GradCAM image: {'None' if gradcam_img is None else f'shape: {gradcam_img.shape}, dtype: {gradcam_img.dtype}'}")
        
        # Use original frame instead of converting to grayscale to preserve colors for detection boxes
        frame_for_display = frame.copy()
        logging.debug(f"Frame for display copied - shape: {frame_for_display.shape}, dtype: {frame_for_display.dtype}")
        
        # Filter overlapping detections
        filtered_detections = self.box_manager.filter_overlapping_boxes(detections)
        logging.debug(f"After filtering: {len(filtered_detections)} detections")
        
        # Draw detection overlays
        frame_with_overlays = frame_for_display.copy()
        logging.debug(f"Frame with overlays copied - shape: {frame_with_overlays.shape}, dtype: {frame_with_overlays.dtype}")
        
        try:
            # Debug: Check if we have detections and if visualizer exists
            logging.debug(f"Drawing detection overlays: {len(filtered_detections)} detections, visualizer: {self.visualizer is not None}")
         
            frame_with_overlays = self.visualizer.draw_detection_overlays(frame_with_overlays, filtered_detections)
            logging.debug(f"Successfully drew detection overlays for {len(filtered_detections)} detections")
            logging.debug(f"Frame after detection overlays - shape: {frame_with_overlays.shape}, dtype: {frame_with_overlays.dtype}")
            
            # Verify frame is still valid after drawing overlays
            if frame_with_overlays is None or frame_with_overlays.size == 0:
                logging.error("Frame became invalid after drawing detection overlays!")
                frame_with_overlays = frame_for_display.copy()
                if filtered_detections:
                    frame_with_overlays = self.visualizer.draw_detection_overlays(frame_with_overlays, filtered_detections)
                
        except Exception as e:
            logging.error(f"Error drawing detection overlays: {e}")
            import traceback
            logging.error(f"Overlay drawing traceback: {traceback.format_exc()}")
            # Fallback: use original frame
            frame_with_overlays = frame_for_display.copy()
        
        # Apply Grad-CAM if enabled and if it's different from the original frame
        if (self.display_config.gradcam_enabled and gradcam_img is not None and 
            not np.array_equal(gradcam_img, frame)):
            frame_with_overlays = self._apply_gradcam_overlay(frame_with_overlays, filtered_detections, gradcam_img)
            
            # When gradcam_in_box_only is False, redraw detection overlays on top of GradCAM
            # to ensure they remain visible
            if not self.display_config.gradcam_in_box_only and filtered_detections:
                logging.debug("Redrawing detection overlays on top of full GradCAM overlay")
                frame_with_overlays = self.visualizer.draw_detection_overlays(frame_with_overlays, filtered_detections)
        elif self.display_config.gradcam_enabled and gradcam_img is None:
            # GradCAM is enabled but no image available - this is normal when GradCAM fails
            # The frame_with_overlays already contains the detection boxes, so no action needed
            # This ensures that when gradcam_in_box_only = False and no GradCAM is available,
            # the system displays the normal frame with detection boxes (default behavior)
            logging.debug("GradCAM enabled but no image available - displaying normal frame with detections")
            # Ensure we have a valid frame to return
            if frame_with_overlays is None or frame_with_overlays.size == 0:
                logging.warning("Frame with overlays is invalid, using original frame")
                frame_with_overlays = frame_for_display.copy()
                # Redraw detection overlays on the fresh frame
                if filtered_detections:
                    frame_with_overlays = self.visualizer.draw_detection_overlays(frame_with_overlays, filtered_detections)
        elif not self.display_config.gradcam_enabled:
            # GradCAM is disabled - frame_with_overlays already contains detection boxes
            logging.debug("GradCAM disabled - displaying normal frame with detections")
        
        # Draw glitches if enabled
        if self.display_config.enable_glitches:
            self.visualizer.draw_random_glitches(frame_with_overlays)
        
        # Draw counter on frame if enabled
        if (COUNTER_CONFIG.get('enabled', True) and 
            hasattr(self, 'count_handler') and self.count_handler):
            position = COUNTER_CONFIG.get('position', 'top_right')
            frame_with_overlays = self.count_handler.draw_counter_on_frame(frame_with_overlays, position)
        
        # Final debug logging before return
        logging.debug(f"Final frame shape: {frame_with_overlays.shape}, dtype: {frame_with_overlays.dtype}")
        logging.debug(f"Frame is None: {frame_with_overlays is None}, Frame size: {frame_with_overlays.size if frame_with_overlays is not None else 'N/A'}")
        
        # Final safety check - ensure we return a valid frame
        if frame_with_overlays is None or frame_with_overlays.size == 0:
            logging.error("Final frame is invalid, returning original frame as fallback")
            frame_with_overlays = frame_for_display.copy()
            if filtered_detections:
                frame_with_overlays = self.visualizer.draw_detection_overlays(frame_with_overlays, filtered_detections)
        
        return frame_with_overlays
    
    def _apply_gradcam_overlay(self, frame: np.ndarray, detections: List[Detection], 
                               gradcam_img: np.ndarray) -> np.ndarray:
        """Apply Grad-CAM overlay to the frame with improved synchronization"""
        if gradcam_img is None or frame is None:
            return frame
        
        # Ensure both images have the same dimensions
        if gradcam_img.shape != frame.shape:
            logging.warning(f"GradCAM shape mismatch: frame {frame.shape} vs gradcam {gradcam_img.shape}")
            # Resize GradCAM to match frame
            gradcam_img = cv2.resize(gradcam_img, (frame.shape[1], frame.shape[0]))
        
        if self.display_config.gradcam_in_box_only and detections:
            # Overlay Grad-CAM only inside boxes
            for detection in detections:
                if detection.box is not None and len(detection.box) == 4:
                    x1, y1, x2, y2 = detection.box
                    
                    # Ensure coordinates are within bounds
                    x1 = max(0, min(x1, frame.shape[1] - 1))
                    y1 = max(0, min(y1, frame.shape[0] - 1))
                    x2 = max(x1 + 1, min(x2, frame.shape[1]))
                    y2 = max(y1 + 1, min(y2, frame.shape[0]))
                    
                    # Extract regions
                    roi = frame[y1:y2, x1:x2]
                    grad_roi = gradcam_img[y1:y2, x1:x2]
                    
                    # Ensure ROI shapes match and are valid
                    if roi.shape == grad_roi.shape and roi.size > 0:
                        try:
                            # Apply alpha blending
                            alpha = DEFAULT_GRADCAM_ALPHA
                            blended = cv2.addWeighted(roi, 1 - alpha, grad_roi, alpha, 0)
                            frame[y1:y2, x1:x2] = blended
                        except Exception as e:
                            logging.warning(f"GradCAM blending failed for detection {detection}: {e}")
                            # Fallback: just copy the GradCAM region
                            frame[y1:y2, x1:x2] = grad_roi
        else:
            # Use full Grad-CAM image with alpha blending
            try:
                alpha = DEFAULT_GRADCAM_ALPHA
                frame = cv2.addWeighted(frame, 1 - alpha, gradcam_img, alpha, 0)
            except Exception as e:
                logging.warning(f"Full GradCAM overlay failed: {e}")
                # Fallback: replace frame with GradCAM
                frame = gradcam_img.copy()
        
        return frame
    
    def _create_legend_frame(self, legend_dict: Dict[int, Tuple[float, Tuple[int, int, int]]], 
                            inf_time: float, monitors) -> None:
        """Create and display legend frame on secondary monitor"""
        if not (self.display_config.show_legend or self.display_config.show_fps_info):
            return
        
        text_frame = np.zeros((LEGEND_WINDOW_HEIGHT, LEGEND_WINDOW_WIDTH, 3), dtype=np.uint8)

        if self.display_config.show_legend:
            self.visualizer.draw_legend(
                text_frame, legend_dict, center=True, 
                font_scale=LEGEND_FONT_SCALE, 
                line_height=LEGEND_LINE_HEIGHT, 
                box_padding=LEGEND_BOX_PADDING
            )
        if self.display_config.show_fps_info:
            self.visualizer.draw_fps_info(
                text_frame, inf_time, center=True, 
                font_scale=LEGEND_FONT_SCALE, 
                y_offset=LEGEND_WINDOW_HEIGHT//2 + 60
            )

        # Use screen configuration for legend positioning
        if self.screen_id == 1 and self.screen_config:  # Secondary monitor
            try:
                # Create legend window in normal mode first for positioning
                cv2.namedWindow("Legend Display", cv2.WINDOW_NORMAL)
                
                # Position legend window based on screen configuration
                x_offset = self.screen_config.get('x_offset', 1920)  # Default to right of primary
                y_offset = self.screen_config.get('y_offset', 0)
                
                # Set window size
                cv2.resizeWindow("Legend Display", LEGEND_WINDOW_WIDTH, LEGEND_WINDOW_HEIGHT)
                
                # Move window to correct position
                cv2.moveWindow("Legend Display", x_offset, y_offset)
                logging.info(f"Positioned legend window at ({x_offset}, {y_offset})")
                
                # Now set fullscreen properties without showing dummy content
                cv2.setWindowProperty("Legend Display", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                
                # Show the actual legend content
                cv2.imshow("Legend Display", text_frame)
                
            except Exception as e:
                logging.warning(f"Failed to position legend window: {e}")
                # Fallback to simple fullscreen
                cv2.namedWindow("Legend Display", cv2.WND_PROP_FULLSCREEN)
                cv2.imshow("Legend Display", text_frame)
        elif monitors and len(monitors) > 1:
            # Fallback to monitor-based positioning if screen config not available
            monitor_2 = monitors[1]
            cv2.namedWindow("Legend Display", cv2.WND_PROP_FULLSCREEN)
            cv2.moveWindow("Legend Display", monitor_2.x, monitor_2.y)
            cv2.setWindowProperty("Legend Display", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Legend Display", text_frame)
        else:
            # Single monitor or no screen config
            cv2.namedWindow("Legend Display", cv2.WINDOW_NORMAL)
            cv2.imshow("Legend Display", text_frame)
    
    def _cleanup_legend_window(self) -> None:
        """Clean up the legend window specifically to prevent black squares"""
        try:
            if cv2.getWindowProperty("Legend Display", cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyWindow("Legend Display")
                logging.info("Destroyed legend window to prevent black squares")
        except Exception as e:
            logging.debug(f"Legend window cleanup: {e}")
    
    def _cleanup_windows(self) -> None:
        """Clean up all OpenCV windows to prevent black squares"""
        try:
            # Get all window names
            window_names = []
            
            # Clean up main window
            try:
                if cv2.getWindowProperty("Object detection Main", cv2.WND_PROP_VISIBLE) >= 0:
                    window_names.append("Object detection Main")
                    cv2.destroyWindow("Object detection Main")
            except:
                pass
            
            # Clean up legend window
            try:
                if cv2.getWindowProperty("Legend Display", cv2.WND_PROP_VISIBLE) >= 0:
                    window_names.append("Legend Display")
                    cv2.destroyWindow("Legend Display")
            except:
                pass
            
            # Destroy any remaining windows
            cv2.destroyAllWindows()
            logging.info(f"Destroyed {len(window_names)} OpenCV windows: {window_names}")
            
            # Force a small delay to ensure windows are fully destroyed
            import time
            time.sleep(0.1)
            
        except Exception as e:
            logging.warning(f"Error during window cleanup: {e}")
    
    def _setup_fullscreen_window(self, window_name: str) -> None:
        """Setup fullscreen window with proper properties and screen positioning"""
        # Log screen configuration for debugging
        logging.info(f"Setting up fullscreen window '{window_name}' for app {self.app_id}")
        logging.info(f"Screen ID: {self.screen_id}")
        logging.info(f"Screen config: {self.screen_config}")
        
        # Create window in normal mode first for positioning
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Position window based on screen configuration BEFORE going fullscreen
        if self.screen_config and 'x_offset' in self.screen_config and 'y_offset' in self.screen_config:
            try:
                # Set window size first to ensure proper positioning
                window_width = self.screen_config.get('width', DEFAULT_WINDOW_WIDTH)
                window_height = self.screen_config.get('height', DEFAULT_WINDOW_HEIGHT)
                cv2.resizeWindow(window_name, window_width, window_height)
                logging.info(f"Resized window to {window_width}x{window_height}")
                
                # Move window to correct position
                x_offset = self.screen_config['x_offset']
                y_offset = self.screen_config['y_offset']
                cv2.moveWindow(window_name, x_offset, y_offset)
                logging.info(f"Positioned window '{window_name}' at ({x_offset}, {y_offset})")
                
                # Additional positioning verification
                logging.info(f"Window '{window_name}' should now be at position ({x_offset}, {y_offset})")
                
            except Exception as e:
                logging.warning(f"Failed to position window: {e}")
                logging.warning(f"Screen config: {self.screen_config}")
        else:
            logging.warning(f"No screen configuration found for screen_id {self.screen_id}")
            logging.warning(f"Available screen configs: {list(SCREEN_CONFIG.keys())}")
        
        # Now set fullscreen properties
        try:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_BORDERLESS, 1)
        except Exception:
            pass
        
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Final fullscreen setup - no dummy windows needed
        try:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_BORDERLESS, 1)
        except Exception:
            pass
        
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        logging.info(f"Fullscreen window '{window_name}' setup complete")
        
        # Small delay to ensure window is properly set up
        import time
        time.sleep(0.05)
    
    def _handle_key_input(self, key: int, cap, out_writer) -> Tuple[bool, bool]:
        """Handle keyboard input and return (should_exit, should_next_video)"""
        # Only increment key press counter for actual user key presses (not programmatic actions)
        # Check if this is a real user key press by looking for specific key codes
        if (key != 0 and COUNTER_CONFIG.get('enabled', True) and 
            hasattr(self, 'count_handler') and self.count_handler):
            # Only count actual user input keys, not programmatic or system keys
            # Also check if this is a valid ASCII key (to avoid counting system/control keys)
            if (key in [ord('q'), ord('n'), ord('b'), ord('l'), ord('f'), ord('d'), 27] and  # ESC
                key < 256):  # Only count valid ASCII keys
                self.count_handler.increment_key_press()
                logging.debug(f"User key press detected: {chr(key) if key < 128 else key}")
        
        # Trigger visual feedback for key press
        if key != 0 and hasattr(self, 'visualizer') and self.visualizer:
            self.visualizer.trigger_key_feedback(key)
        
        if key == ord('q') or key == 27:  # 'q' or ESC
            return True, False
        elif key == ord('n'):  # Next video
            return False, True
        elif key == ord('b'):  # Toggle Grad-CAM box mode
            self.display_config.gradcam_in_box_only = not self.display_config.gradcam_in_box_only
            logging.info(f"Grad-CAM in box only: {self.display_config.gradcam_in_box_only}")
        elif key == ord('l'):  # Toggle legend
            self.display_config.show_legend = not self.display_config.show_legend
            logging.info(f"Show legend: {self.display_config.show_legend}")
        elif key == ord('f'):  # Next folder
            logging.info("'f' key pressed - switching to next folder")
            self.log_current_state()  # Log state before switching
            self.switch_to_next_folder()
            self.log_current_state()  # Log state after switching
            logging.info(f"Switched to next folder: {self.video_path}")
            # After switching folders, trigger next video to show content from new folder
            should_next_video = True
        elif key == ord('t'):  # Toggle FPS info (changed from 'f' to 't')
            self.display_config.show_fps_info = not self.display_config.show_fps_info
            logging.info(f"Show FPS info: {self.display_config.show_fps_info}")
        elif key == ord('d'):  # Toggle NDI output
            self.display_config.ndi_enabled = not self.display_config.ndi_enabled
            if self.display_config.ndi_enabled:
                logging.info(f"NDI output enabled for app {self.app_id}")
                # Update NDI sender configuration
                ndi_config = self._get_ndi_config()
                self.ndi_sender.update_config(
                    source_name=ndi_config['source_name'],
                    group_name=ndi_config['group_name'],
                    video_format=ndi_config['video_format'],
                    frame_rate=ndi_config['frame_rate'],
                    video_width=ndi_config['video_width'],
                    video_height=ndi_config['video_height']
                )
            else:
                logging.info(f"NDI output disabled for app {self.app_id}")
        else:
            self.visualizer.handle_key_press(key, self)
        
        return False, False
    
    def _reset_gradcam_state(self) -> None:
        """Reset GradCAM processor state for new video"""
        if self.gradcam_processor is not None:
            try:
                # Reset frame skip counter and clear buffers
                self.gradcam_processor.frame_skip_counter = 0
                self.gradcam_processor.last_gradcam_img = None
                self.gradcam_processor.last_frame_shape = None
                logging.debug("GradCAM state reset for new video")
            except Exception as e:
                logging.warning(f"Failed to reset GradCAM state: {e}")
    
    def _cleanup_video_resources(self) -> None:
        """Clean up video-specific resources when switching videos"""
        try:
            # Only clean up video-related resources, not core components
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            
            gc.collect()
            
            # Clean up visualizer audio resources for new video
            if hasattr(self, 'visualizer'):
                self.visualizer.cleanup_audio()
            
            # Clean up NDI sender for new video
            if hasattr(self, 'ndi_sender'):
                self.ndi_sender.cleanup()
                # Reinitialize NDI sender for new video
                ndi_config = self._get_ndi_config()
                self.ndi_sender = create_ndi_sender(
                    source_name=ndi_config['source_name'],
                    group_name=ndi_config['group_name'],
                    video_format=ndi_config['video_format'],
                    frame_rate=ndi_config['frame_rate'],
                    video_width=ndi_config['video_width'],
                    video_height=ndi_config['video_height']
                )
            
            logging.debug("Video resources cleaned up for next video")
            
        except Exception as e:
            logging.warning(f"Video resource cleanup failed: {e}")
    
    def _cleanup_resources(self) -> None:
        """Clean up all resources when stopping the application"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            
            gc.collect()
            
            if self.gradcam_processor is not None:
                self.gradcam_processor.cleanup()
            
            # Clean up visualizer audio resources
            if hasattr(self, 'visualizer'):
                self.visualizer.cleanup_audio()
            
            # Clean up NDI sender
            if hasattr(self, 'ndi_sender'):
                self.ndi_sender.cleanup()
            
            # Clean up button handler
            if hasattr(self, 'button_handler'):
                self.button_handler.stop()
            
            logging.info("Resource cleanup completed")
            
        except Exception as e:
            logging.warning(f"Resource cleanup failed: {e}")
    
    def run(self, stop_event=None) -> None:
        """Main application loop"""
        # Initialize button handler for backward compatibility (single app mode)
        self._initialize_button_handler()
        
        # Get all .mp4 files in the current folder
        mp4_files = self.refresh_video_files()
        
        if not mp4_files:
            logging.error(f"No video files found in current folder: {self.video_path}")
            return

        while True:  # Loop forever over the videos
            # Check stop event if provided (for multi-app mode)
            if stop_event and stop_event.is_set():
                logging.info(f"Stop signal received for app {self.app_id}")
                break
            
            # Check if we need to refresh video files (e.g., after folder switch)
            if hasattr(self, '_button_next_folder_signal') and self._button_next_folder_signal:
                logging.info("Folder switch detected, refreshing video file list...")
                old_mp4_files = mp4_files.copy() if mp4_files else []
                mp4_files = self.force_refresh_video_files()
                if not mp4_files:
                    logging.error(f"No video files found in new folder: {self.video_path}")
                    return
                self._button_next_folder_signal = False  # Reset the flag
                logging.info(f"Video file list refreshed: {len(mp4_files)} files in {self.video_path}")
                logging.info(f"Old list had {len(old_mp4_files)} files, new list has {len(mp4_files)} files")
                if old_mp4_files != mp4_files:
                    logging.info("Video file list successfully updated")
                else:
                    logging.warning("Video file list unchanged - this might indicate an issue")
            
            # Also check if we need to refresh due to next video signal after folder switch
            if hasattr(self, '_button_next_video_signal') and self._button_next_video_signal:
                logging.info("Next video signal detected, checking if video file list needs refresh...")
                # Check if current mp4_files list matches current folder
                current_folder_files = self.force_refresh_video_files()
                logging.info(f"Current mp4_files list: {len(mp4_files)} files")
                logging.info(f"Refreshed folder files: {len(current_folder_files)} files")
                logging.info(f"Current folder: {self.video_path}")
                
                if current_folder_files != mp4_files:
                    logging.info("Video file list mismatch detected, updating...")
                    mp4_files = current_folder_files
                    logging.info(f"Updated video file list: {len(mp4_files)} files in {self.video_path}")
                else:
                    logging.info("Video file list is up to date")
                self._button_next_video_signal = False  # Reset the flag
            
            # Log current state for debugging
            logging.info(f"Starting video loop with {len(mp4_files)} files from folder: {self.video_path}")
            logging.info(f"mp4_files list: {mp4_files[:3] if mp4_files else 'None'}")
            
            for mp4 in mp4_files:
                # Check stop event before processing each video
                if stop_event and stop_event.is_set():
                    logging.info(f"Stop signal received for app {self.app_id}")
                    return
                    
                print(f"Processing: {mp4}")
                video_file_path = os.path.join(self.video_path, mp4)
                logging.info(f"Processing video: {mp4} from path: {video_file_path}")
                logging.info(f"Current video_path: {self.video_path}")
                
                # Verify the video file actually exists
                if not os.path.exists(video_file_path):
                    logging.error(f"Video file not found: {video_file_path}")
                    logging.error(f"Current folder: {self.video_path}")
                    logging.error(f"mp4_files list: {mp4_files}")
                    continue
                
                cap = cv2.VideoCapture(video_file_path)
                if not cap.isOpened():
                    logging.error(f"Could not open video file: {mp4}")
                    continue
                
                # Validate video file properties
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames <= 0:
                    logging.warning(f"Video file {mp4} appears to have no frames or is corrupted")
                    cap.release()
                    continue
                
                logging.info(f"Video {mp4}: {total_frames} frames, {cap.get(cv2.CAP_PROP_FPS):.2f} FPS")

                # Reset GradCAM state for new video
                self._reset_gradcam_state()
                
                # Ensure audio is initialized for new video
                if hasattr(self, 'visualizer'):
                    self.visualizer.reinitialize_audio()

                # Setup video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = DEFAULT_FPS
                wait_ms = int(1000 / fps)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Setup output video writer
                out_writer = None
                if self.output_path:
                    output_dir = str(Path(self.output_path).parent)
                    if not Path(output_dir).exists():
                        Path(output_dir).mkdir(parents=True, exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out_writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
                    logging.info(f"Saving output video to: {self.output_path}")

                logging.info("Press SPACE to toggle border color and show 'ENEMY'. Press 'q' or ESC to quit.")
                logging.info("Press 'd' to toggle NDI output. Press 'b' to toggle GradCAM box mode.")
                logging.info("Press 'l' to toggle legend. Press 't' to toggle FPS info.")
                if len(self.video_folders) > 1:
                    logging.info(f"Press 'f' to switch to next folder ({self.current_folder_index + 1}/{len(self.video_folders)}).")
                    logging.info(f"Current folder: {self.video_path}")

                # Setup fullscreen window
                window_name = 'Object detection Main'
                
                # Check if window already exists and destroy it to prevent black square
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 0:
                    cv2.destroyWindow(window_name)
                    logging.info("Destroyed existing window to prevent black square")
                
                # Use the new window cleanup method to ensure clean fullscreen setup
                self._cleanup_windows()
                
                self._setup_fullscreen_window(window_name)

                # Main video processing loop
                frame_count = 0
                first_frame = True
                fullscreen_size = None

                try:
                    while True:
                        # Check stop event before processing each frame
                        if stop_event and stop_event.is_set():
                            logging.info(f"Stop signal received for app {self.app_id}")
                            # Clean up current video resources
                            if out_writer:
                                out_writer.release()
                            cap.release()
                            # Close OpenCV windows
                            cv2.destroyAllWindows()
                            # Clean up resources
                            self._cleanup_resources()
                            return
                            
                        ret, frame = cap.read()
                        frame_count += 1
                        
                        if frame_count == self.max_frames and frame_count > 0:
                            break
                        
                        #rint(f"FRAME: {frame_count}")
                        if not ret:
                            logging.info(f"Failed to grab frame {frame_count}: ret={ret}")
                            break
                        if frame is None:
                            logging.warning(f"Frame {frame_count} is None despite ret={ret}")
                            break
                        
                        # Check stop event again after frame read
                        if stop_event and stop_event.is_set():
                            logging.info(f"Stop signal received for app {self.app_id} after frame read")
                            if out_writer:
                                out_writer.release()
                            cap.release()
                            cv2.destroyAllWindows()
                            self._cleanup_resources()
                            return
                        
                        # Debug: Check frame properties
                        if frame_count == 1:
                            print(f"First frame: shape={frame.shape}, dtype={frame.dtype}")
                        
                        # Process frame
                        detections, inf_time = self._process_frame(frame)
                        
                        # Debug: Log detection results
                        if frame_count % 30 == 0:  # Every 30 frames
                            logging.info(f"Frame {frame_count}: {len(detections)} detections, inference time: {inf_time:.1f}ms")
                            if detections:
                                for i, det in enumerate(detections[:3]):  # Log first 3 detections
                                    logging.info(f"  Detection {i}: class={det.class_id}, conf={det.confidence:.3f}, box={det.box}")
                        
                        # Check stop event after frame processing
                        if stop_event and stop_event.is_set():
                            logging.info(f"Stop signal received for app {self.app_id} after frame processing")
                            if out_writer:
                                out_writer.release()
                            cap.release()
                            cv2.destroyAllWindows()
                            self._cleanup_resources()
                            return
                        
                        # Generate Grad-CAM (with additional safety check)
                        gradcam_img = None  # Initialize gradcam_img
                        if frame is not None and self.gradcam_processor is not None:
                            logging.debug(f"Processing frame {frame_count} with shape {frame.shape}")
                            try:
                                # Check if GradCAM processor needs reloading
                                if self.gradcam_processor.needs_reload():
                                    logging.info("GradCAM processor needs reloading, attempting...")
                                    if self.gradcam_processor.reload_model(self.model_path):
                                        logging.info("GradCAM processor reloaded successfully")
                                    else:
                                        logging.warning("Failed to reload GradCAM processor")
                                        # Don't set gradcam_img here - let it remain None to show normal frame
                                        gradcam_img = None
                                
                                # Frame skip threshold adjustment removed to fix synchronization issues
                                # GradCAM now processes every frame for better sync
                                
                                gradcam_img = self.gradcam_processor.process_gradcam(
                                    frame, detections, self.display_config.gradcam_in_box_only
                                )
                                
                                # Check if GradCAM actually produced a meaningful result
                                if gradcam_img is not None and np.array_equal(gradcam_img, frame):
                                    # GradCAM returned the original frame (no meaningful result)
                                    if not self.display_config.gradcam_in_box_only:
                                        # For full-frame mode, if no meaningful GradCAM, show normal frame
                                        logging.debug("GradCAM returned original frame - showing normal frame with detections")
                                        gradcam_img = None
                                
                            except Exception as e:
                                logging.warning(f"GradCAM processing failed: {e}, showing normal frame")
                                gradcam_img = None  # Don't apply overlay, show normal frame
                        elif frame is not None:
                            logging.debug("GradCAM processor not available, showing normal frame")
                            gradcam_img = None  # Don't apply overlay, show normal frame
                        else:
                            logging.warning("Frame is None, creating blank image")
                            gradcam_img = np.zeros((480, 640, 3), dtype=np.uint8)
                        
                        # Prepare display frame (with additional safety check)
                        if frame is not None:
                            logging.debug(f"Preparing display frame - frame shape: {frame.shape}, detections: {len(detections)}")
                            display_img = self._prepare_display_frame(frame, detections, gradcam_img)
                            logging.debug(f"Display frame prepared - shape: {display_img.shape}, dtype: {display_img.dtype}")
                        else:
                            logging.warning("Frame is None, creating blank display image")
                            display_img = np.zeros((480, 640, 3), dtype=np.uint8)
                        
                        # Create legend frame
                        legend_dict = self._create_legend_dict(detections)
                        monitors = get_monitors() if get_monitors is not None else None
                        self._create_legend_frame(legend_dict, inf_time, monitors)
                        
                        # Display frame
                        try:
                            # Ensure the main window exists before displaying
                            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 0:
                                logging.warning("Main window not visible, recreating...")
                                self._cleanup_windows()
                                self._setup_fullscreen_window(window_name)
                            
                            # Display the frame without creating temporary windows
                            first_frame, fullscreen_size = self._display_frame_fullscreen(window_name, display_img, first_frame, fullscreen_size)
                            
                            # Small delay to ensure frame is displayed properly
                            cv2.waitKey(1)
                            
                        except Exception as e:
                            logging.warning(f"Error displaying frame: {e}")
                            # Continue with next frame
                        
                        # Send frame via NDI if enabled
                        if self.display_config.ndi_enabled and self.ndi_sender.is_available():
                            try:
                                self.ndi_sender.send_frame(display_img)
                            except Exception as e:
                                logging.warning(f"Error sending frame via NDI: {e}")
                        
                        # Debug: Print frame info
                        if frame_count % 30 == 0:  # Every 30 frames
                            print(f"Frame {frame_count}: shape={display_img.shape}, detections={len(detections)}, inf_time={inf_time:.1f}ms")
                            print(f"Window state: first_frame={first_frame}, fullscreen_size={fullscreen_size}")
                        
                        # Handle key input with proper timing and stop event check
                        try:
                            # Use shorter wait time to be more responsive to stop events
                            key_wait_time = min(wait_ms, 50)  # Max 50ms wait
                            key = cv2.waitKey(key_wait_time) & 0xFF
                        except Exception as e:
                            logging.warning(f"Error handling key input: {e}")
                            key = 0
                        
                        # Check stop event again before key handling
                        if stop_event and stop_event.is_set():
                            logging.info(f"Stop signal received for app {self.app_id} during key handling")
                            if out_writer:
                                out_writer.release()
                            cap.release()
                            cv2.destroyAllWindows()
                            self._cleanup_resources()
                            return
                        
                        should_exit, should_next_video = self._handle_key_input(key, cap, out_writer)
                        
                        # Check button signals
                        if self._button_next_video_signal:
                            should_next_video = True
                            self._button_next_video_signal = False  # Reset the flag
                            logging.info(f"Button next video signal processed for app {self.app_id}")
                        
                        if self._button_next_folder_signal:
                            logging.info("Processing next folder signal...")
                            self.log_current_state()  # Log state before switching
                            self.switch_to_next_folder()
                            self.log_current_state()  # Log state after switching
                            self._button_next_folder_signal = False  # Reset the flag
                            logging.info(f"Button next folder signal processed for app {self.app_id}")
                            
                            # After switching folders, we need to break out of the current video loop
                            # and refresh the video file list to start fresh with the new folder
                            logging.info("Breaking out of current video loop to refresh video files from new folder")
                            # Force the next video to start immediately
                            should_next_video = True
                            # Use the new window cleanup method to prevent black squares
                            self._cleanup_windows()
                            logging.info("About to break out of video loop...")
                            break
                        
                        if should_exit:
                            # Use the new window cleanup method to prevent black squares
                            self._cleanup_windows()
                            
                            self._cleanup_resources()
                            if out_writer:
                                out_writer.release()
                            cap.release()
                            return
                        elif should_next_video:
                            # Clean up current video resources but keep core components
                            if out_writer:
                                out_writer.release()
                            cap.release()
                            
                            # Use the new window cleanup method to prevent black squares
                            self._cleanup_windows()
                            
                            self._cleanup_video_resources()
                            # Reset GradCAM state for next video
                            self._reset_gradcam_state()
                            # Reinitialize audio for next video
                            if hasattr(self, 'visualizer'):
                                self.visualizer.reinitialize_audio()
                            # Reinitialize button handler for next video
                            if hasattr(self, 'button_handler'):
                                if not self.button_handler.get_running_status():
                                    logging.info("Button handler not running, restarting...")
                                    self.button_handler.restart()
                                else:
                                    logging.debug("Button handler already running")
                            # Reload GradCAM processor if needed
                            if (self.gradcam_processor is not None and 
                                self.gradcam_processor.needs_reload()):
                                self.gradcam_processor.reload_model(self.model_path)
                            elif self.gradcam_processor is None:
                                # Reinitialize GradCAM processor if it was cleaned up
                                try:
                                    self.gradcam_processor = GradCAMProcessor(self.model_path)
                                    logging.info("GradCAM processor reinitialized for new video")
                                except Exception as e:
                                    logging.warning(f"Failed to reinitialize GradCAM processor: {e}")
                            break
                        
                        self.visualizer.frame_idx += 1
                        
                        # Ensure ambient sounds are playing and check for cycling
                        self.visualizer.ensure_ambient_playing()
                        self.visualizer.check_ambient_cycle_timer()
                        
                        # Write to output video
                        if out_writer:
                            try:
                                out_writer.write(display_img)
                            except Exception as e:
                                logging.warning(f"Error writing to output video: {e}")
                                # Try to recreate the writer
                                try:
                                    out_writer.release()
                                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                    out_writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
                                    logging.info("Recreated output video writer")
                                except Exception as e2:
                                    logging.error(f"Failed to recreate output video writer: {e2}")
                                    out_writer = None
                        
                        first_frame = False
                    
                    # Clean up video resources
                    cap.release()
                    if out_writer:
                        out_writer.release()
                        
                except Exception as e:
                    logging.error(f"Error during video processing: {e}")
                    logging.error(f"Frame count: {frame_count}")
                    if frame is not None:
                        logging.error(f"Frame shape: {frame.shape}")
                    else:
                        logging.error("Frame is None")
                    # Clean up resources on error
                    cap.release()
                    if out_writer:
                        out_writer.release()
                    raise
            
            # Clean up resources after processing all videos
            self._cleanup_resources()
            if out_writer:
                out_writer.release()
            # Don't return here - let it continue looping to the first video
            logging.info("Completed all videos, looping back to first video...")
    
    def _create_legend_dict(self, detections: List[Detection]) -> Dict[int, Tuple[float, Tuple[int, int, int]]]:
        """Create legend dictionary from detections"""
        legend_dict = {}
        for detection in detections:
            if detection.class_id not in legend_dict or detection.confidence > legend_dict[detection.class_id][0]:
                color = ColorManager.get_state_color(self.display_config.color_state, detection.confidence)
                legend_dict[detection.class_id] = (detection.confidence, color)
        return legend_dict
    
    def _display_frame_fullscreen(self, window_name: str, display_img: np.ndarray, 
                                 first_frame: bool, fullscreen_size: Optional[Tuple[int, int]]) -> Tuple[bool, Optional[Tuple[int, int]]]:
        """Display frame in fullscreen with configurable sizing using screen configuration"""
        debug = False
        if first_frame:
            # Use screen configuration for dimensions
            ww = self.screen_config.get('width', DEFAULT_WINDOW_WIDTH)
            wh = self.screen_config.get('height', DEFAULT_WINDOW_HEIGHT)
            logging.info(f"Using screen config size: {ww}x{wh}")
            
            fullscreen_size = (ww, wh)
            first_frame = False
            logging.info(f"Final fullscreen size set to: {ww}x{wh}")
        
        # Always display the frame, even if fullscreen_size is not set yet
        if fullscreen_size:
            ww, wh = fullscreen_size
            ih, iw = display_img.shape[:2]
            
            # Get scaling configuration from screen config
            scale_mode = self.screen_config.get('scale_mode', DEFAULT_VIDEO_SCALE_MODE)
            scale_multiplier = self.screen_config.get('scale_multiplier', DEFAULT_VIDEO_SCALE_MULTIPLIER)
            maintain_aspect_ratio = self.screen_config.get('maintain_aspect_ratio', DEFAULT_VIDEO_MAINTAIN_ASPECT_RATIO)
            center_video = self.screen_config.get('center_video', DEFAULT_VIDEO_CENTER_ON_SCREEN)
            
            # Debug logging for scaling
            logging.info(f"Scaling: video={iw}x{ih}, screen={ww}x{wh}, mode={scale_mode}") if debug else None
            
            # Apply scaling based on configuration
            if scale_mode == 'original':
                # Display original size
                new_w, new_h = iw, ih
                logging.info(f"Original mode: keeping video size {iw}x{ih}") if debug else None
            elif scale_mode == 'stretch':
                # Stretch to fill entire window (may distort video)
                new_w, new_h = ww, wh
                logging.info(f"Stretch mode: stretching to screen size {ww}x{wh}") if debug else None
            else:  # 'fit' mode (default)
                # Fit to fill screen while maintaining aspect ratio
                if maintain_aspect_ratio:
                    scale = min(ww / iw, wh / ih)
                else:
                    scale = max(ww / iw, wh / ih)
                # Apply multiplier to use more screen space
                scale = scale * scale_multiplier
                new_w, new_h = int(iw * scale), int(ih * scale)
                logging.info(f"Fit mode: scale={scale:.3f}, multiplier={scale_multiplier}, final={new_w}x{new_h}") if debug else None
            
            # Resize the image
            resized_img = cv2.resize(display_img, (new_w, new_h))
            
            # Debug logging for final dimensions
            logging.info(f"Final dimensions: original={iw}x{ih}, scaled={new_w}x{new_h}, screen={ww}x{wh}")if debug else None
            
            if center_video:
                # Center on black background
                fullscreen_img = np.zeros((wh, ww, 3), dtype=np.uint8)
                y_offset = (wh - new_h) // 2
                x_offset = (ww - new_w) // 2
                
                # Debug logging for positioning
                logging.info(f"Positioning: screen={ww}x{wh}, video={new_w}x{new_h}, offsets=({x_offset}, {y_offset})") if debug else None
                
                # Ensure the video fits within the screen bounds
                if x_offset < 0 or y_offset < 0:
                    logging.warning(f"Video too large for screen: video={new_w}x{new_h}, screen={ww}x{wh}") if debug else None
                    # Force video to fit within screen
                    if new_w > ww:
                        new_w = ww
                        x_offset = 0
                    if new_h > wh:
                        new_h = wh
                        y_offset = 0
                    # Resize again if needed
                    if new_w != display_img.shape[1] or new_h != display_img.shape[0]:
                        resized_img = cv2.resize(display_img, (new_w, new_h))
                
                fullscreen_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
                cv2.imshow(window_name, fullscreen_img)
            else:
                # Display resized image directly
                cv2.imshow(window_name, resized_img)
        else:
            # Fallback: display the frame directly if fullscreen size not set
            cv2.imshow(window_name, display_img)
        
        return first_frame, fullscreen_size
    
    def set_display_config(self, config: DisplayConfig) -> None:
        """Update the display configuration"""
        self.display_config = config
        self.visualizer.update_display_config(config)
    
    def get_display_config(self) -> DisplayConfig:
        """Get the current display configuration"""
        return self.display_config
    
    def set_box_threshold(self, threshold: float) -> None:
        """Set the detection confidence threshold"""
        self.box_threshold = threshold
        logging.info(f"Box threshold set to {threshold}")
    
    def set_video_size_config(self, width: Optional[int] = None, height: Optional[int] = None,
                             scale_mode: str = 'fit', maintain_aspect_ratio: bool = True,
                             center_video: bool = True, scale_multiplier: float = 0.9) -> None:
        """Set video size configuration"""
        global DEFAULT_VIDEO_WIDTH, DEFAULT_VIDEO_HEIGHT, DEFAULT_VIDEO_SCALE_MODE
        global DEFAULT_VIDEO_MAINTAIN_ASPECT_RATIO, DEFAULT_VIDEO_CENTER_ON_SCREEN, DEFAULT_VIDEO_SCALE_MULTIPLIER
        
        if width is not None:
            DEFAULT_VIDEO_WIDTH = width
        if height is not None:
            DEFAULT_VIDEO_HEIGHT = height
        
        DEFAULT_VIDEO_SCALE_MODE = scale_mode
        DEFAULT_VIDEO_MAINTAIN_ASPECT_RATIO = maintain_aspect_ratio
        DEFAULT_VIDEO_CENTER_ON_SCREEN = center_video
        DEFAULT_VIDEO_SCALE_MULTIPLIER = scale_multiplier
        
        logging.info(f"Video size config updated: {DEFAULT_VIDEO_WIDTH}x{DEFAULT_VIDEO_HEIGHT}, "
                    f"scale_mode={DEFAULT_VIDEO_SCALE_MODE}, "
                    f"maintain_aspect_ratio={DEFAULT_VIDEO_MAINTAIN_ASPECT_RATIO}, "
                    f"center_video={DEFAULT_VIDEO_CENTER_ON_SCREEN}, "
                    f"scale_multiplier={DEFAULT_VIDEO_SCALE_MULTIPLIER}")
    
    def get_box_threshold(self) -> float:
        """Get the current detection confidence threshold"""
        return self.box_threshold
    
    def get_ndi_status(self) -> dict:
        """Get current NDI status and configuration"""
        return {
            'enabled': self.display_config.ndi_enabled,
            'available': self.ndi_sender.is_available() if hasattr(self, 'ndi_sender') else False,
            'source_name': self.display_config.ndi_source_name,
            'group_name': self.display_config.ndi_group_name,
            'video_format': self.display_config.ndi_video_format,
            'frame_rate': self.display_config.ndi_frame_rate,
            'video_width': self.display_config.ndi_video_width,
            'video_height': self.display_config.ndi_video_height,
            'sending': self.ndi_sender.is_sending if hasattr(self, 'ndi_sender') else False
        }
    
    def set_ndi_config(self, enabled: bool = None, source_name: str = None, 
                       group_name: str = None, video_format: str = None,
                       frame_rate: int = None, video_width: int = None, 
                       video_height: int = None) -> None:
        """Update NDI configuration"""
        if enabled is not None:
            self.display_config.ndi_enabled = enabled
        
        if source_name is not None:
            self.display_config.ndi_source_name = source_name
        if group_name is not None:
            self.display_config.ndi_group_name = group_name
        if video_format is not None:
            self.display_config.ndi_video_format = video_format
        if frame_rate is not None:
            self.display_config.ndi_frame_rate = frame_rate
        if video_width is not None:
            self.display_config.ndi_video_width = video_width
        if video_height is not None:
            self.display_config.ndi_video_height = video_height
        
        # Update NDI sender configuration
        if hasattr(self, 'ndi_sender'):
            self.ndi_sender.update_config(
                source_name=self.display_config.ndi_source_name,
                group_name=self.display_config.ndi_group_name,
                video_format=self.display_config.ndi_video_format,
                frame_rate=self.display_config.ndi_frame_rate,
                video_width=self.display_config.ndi_video_width,
                video_height=self.display_config.ndi_video_height
            )
        
        logging.info(f"NDI configuration updated for app {self.app_id}: enabled={self.display_config.ndi_enabled}, "
                    f"source={self.display_config.ndi_source_name}, "
                    f"format={self.display_config.ndi_video_format}, "
                    f"resolution={self.display_config.ndi_video_width}x{self.display_config.ndi_video_height}")
    
    def toggle_ndi(self) -> bool:
        """Toggle NDI output on/off"""
        self.display_config.ndi_enabled = not self.display_config.ndi_enabled
        if self.display_config.ndi_enabled:
            logging.info(f"NDI output enabled for app {self.app_id}")
        else:
            logging.info(f"NDI output disabled for app {self.app_id}")
        return self.display_config.ndi_enabled 

    def _cleanup_windows(self) -> None:
        """Clean up all OpenCV windows to prevent black squares"""
        try:
            # Get all window names
            window_names = []
            
            # Clean up main window
            try:
                if cv2.getWindowProperty("Object detection Main", cv2.WND_PROP_VISIBLE) >= 0:
                    window_names.append("Object detection Main")
                    cv2.destroyWindow("Object detection Main")
            except:
                pass
            
            # Clean up legend window
            try:
                if cv2.getWindowProperty("Legend Display", cv2.WND_PROP_VISIBLE) >= 0:
                    window_names.append("Legend Display")
                    cv2.destroyWindow("Legend Display")
            except:
                pass
            
            # Destroy any remaining windows
            cv2.destroyAllWindows()
            logging.info(f"Destroyed {len(window_names)} OpenCV windows: {window_names}")
            
            # Force a small delay to ensure windows are fully destroyed
            import time
            time.sleep(0.1)
            
        except Exception as e:
            logging.warning(f"Error during window cleanup: {e}") 