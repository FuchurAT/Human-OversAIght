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
    SCREEN_CONFIG
)

from application.models import Detection, DisplayConfig
from application.memory_manager import MemoryManager
from application.box_manager import BoxManager
from application.gradcam_processor import GradCAMProcessor
from application.visualizer import DetectionVisualizer
from application.color_manager import ColorManager
from application.button_handler import ButtonHandler

try:
    from screeninfo import get_monitors
except ImportError:
    get_monitors = None

from config.classes import CLASSES


class VideoInferenceApp:
    """Main application class for video inference with YOLO"""
    
    def __init__(self, video_path: str, model_path: str, box_threshold: float = DEFAULT_BOX_THRESHOLD, 
                 app_id: str = None, screen_id: int = None):
        self.video_path = video_path
        self.model_path = model_path
        self.output_path = ""
        self.box_threshold = box_threshold
        self.max_frames = 0
        self.app_id = app_id or 'default'
        self.screen_id = screen_id or 0
        
        # Initialize components
        self.display_config = DisplayConfig()
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
        
        # Initialize button handler (will be set up by multi-app manager)
        self.button_handler = None
        
        # Screen configuration
        self.screen_config = self._get_screen_config()
        
        # Log initialization status
        logging.info(f"VideoInferenceApp initialized successfully")
        logging.info(f"  App ID: {self.app_id}")
        logging.info(f"  Screen ID: {self.screen_id}")
        logging.info(f"  Model: {self.model_path}")
        logging.info(f"  Video path: {self.video_path}")
        logging.info(f"  Box threshold: {self.box_threshold}")
        logging.info(f"  GradCAM enabled: {self.gradcam_processor.is_model_loaded() if self.gradcam_processor is not None else False}")
        logging.info(f"  Screen config: {self.screen_config}")
    
    def _get_screen_config(self) -> dict:
        """Get screen configuration for this application"""
        if self.screen_id in SCREEN_CONFIG:
            return SCREEN_CONFIG[self.screen_id].copy()
        else:
            # Fallback to default screen config
            return {
                'width': DEFAULT_WINDOW_WIDTH,
                'height': DEFAULT_WINDOW_HEIGHT,
                'x_offset': 0,
                'y_offset': 0,
                'scale_mode': DEFAULT_VIDEO_SCALE_MODE,
                'scale_multiplier': DEFAULT_VIDEO_SCALE_MULTIPLIER,
                'maintain_aspect_ratio': DEFAULT_VIDEO_MAINTAIN_ASPECT_RATIO,
                'center_video': DEFAULT_VIDEO_CENTER_ON_SCREEN
            }
    
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
    
    def _initialize_model(self) -> None:
        """Initialize the YOLO model with proper error handling"""
        try:
            from ultralytics import YOLO
        except ImportError:
            logging.error("ultralytics package not found. Please install it: pip install ultralytics")
            raise
        
        if not Path(self.model_path).exists():
            logging.error(f"Model not found: {self.model_path}")
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Add safe globals for PyTorch serialization
        self._add_safe_globals()
        
        # Create YOLO model with verbose=False to suppress output
        self.model = YOLO(self.model_path, verbose=False)
        logging.info(f"Loaded model: {self.model_path}")
    
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
        
        # Memory cleanup
        self.memory_manager.cleanup_memory()
        
        # Run inference with output suppression
        inf_start = time.time()
        # Temporarily redirect stdout and stderr to suppress YOLO output
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            results = self.model(frame, conf=0.15, verbose=False)
        inf_time = (time.time() - inf_start) * 1000  # ms
        
        # Extract detections
        boxes = results[0].boxes
        detections = []
        
        if boxes is not None:
            for box in boxes:
                conf = float(box.conf[0])
                if conf < self.box_threshold:
                    continue
                class_id = int(box.cls[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                detections.append(Detection(tuple(xyxy), conf, class_id))
        
        return detections, inf_time
    
    def _prepare_display_frame(self, frame: np.ndarray, detections: List[Detection], 
                              gradcam_img: np.ndarray) -> np.ndarray:
        """Prepare the frame for display with all overlays"""
        # Check if frame is valid
        if frame is None:
            logging.warning("Received None frame in _prepare_display_frame")
            # Return a black frame as fallback
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Convert to grayscale and back to BGR (create a copy to avoid modifying original)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_for_display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Filter overlapping detections
        filtered_detections = self.box_manager.filter_overlapping_boxes(detections)
        
        # Draw detection overlays
        frame_with_overlays = frame_for_display.copy()
        self.visualizer.draw_detection_overlays(frame_with_overlays, filtered_detections)
        
        # Apply Grad-CAM if enabled
        if self.display_config.gradcam_enabled:
            frame_with_overlays = self._apply_gradcam_overlay(frame_with_overlays, filtered_detections, gradcam_img)
        
        # Draw glitches if enabled
        if self.display_config.enable_glitches:
            self.visualizer.draw_random_glitches(frame_with_overlays)
        
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

        if monitors and len(monitors) > 1:
            monitor_2 = monitors[1]
            cv2.namedWindow("Legend Display", cv2.WND_PROP_FULLSCREEN)
            cv2.moveWindow("Legend Display", monitor_2.x, monitor_2.y)
            cv2.setWindowProperty("Legend Display", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Legend Display", text_frame)
        else:
            cv2.namedWindow("Legend Display", cv2.WINDOW_NORMAL)
            cv2.imshow("Legend Display", text_frame)
    
    def _setup_fullscreen_window(self, window_name: str) -> None:
        """Setup fullscreen window with proper properties and screen positioning"""
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        
        try:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_BORDERLESS, 1)
        except Exception:
            pass
        
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Position window based on screen configuration
        if self.screen_config and 'x_offset' in self.screen_config and 'y_offset' in self.screen_config:
            try:
                cv2.moveWindow(window_name, self.screen_config['x_offset'], self.screen_config['y_offset'])
                logging.info(f"Positioned window '{window_name}' at ({self.screen_config['x_offset']}, {self.screen_config['y_offset']})")
            except Exception as e:
                logging.warning(f"Failed to position window: {e}")
        
        # Force window to appear and go fullscreen
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imshow(window_name, dummy)
        cv2.waitKey(1)
        
        try:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_BORDERLESS, 1)
        except Exception:
            pass
        
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    def _handle_key_input(self, key: int, cap, out_writer) -> Tuple[bool, bool]:
        """Handle keyboard input and return (should_exit, should_next_video)"""
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
        elif key == ord('f'):  # Toggle FPS info
            self.display_config.show_fps_info = not self.display_config.show_fps_info
            logging.info(f"Show FPS info: {self.display_config.show_fps_info}")
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
            
            # Clean up button handler
            if hasattr(self, 'button_handler'):
                self.button_handler.stop()
            
            logging.info("Resource cleanup completed")
            
        except Exception as e:
            logging.warning(f"Resource cleanup failed: {e}")
    
    def run(self) -> None:
        """Main application loop"""
        # Initialize button handler for backward compatibility (single app mode)
        self._initialize_button_handler()
        
        # Get all .mp4 files in the folder
        mp4_files = [file for file in os.listdir(self.video_path) if file.endswith('.mp4')]

        while True:  # Loop forever over the videos
            for mp4 in mp4_files:
                print(f"Processing: {mp4}")
                video_file_path = os.path.join(self.video_path, mp4)
                
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

                # Setup fullscreen window
                window_name = 'Object detection Main'
                self._setup_fullscreen_window(window_name)

                # Main video processing loop
                frame_count = 0
                first_frame = True
                fullscreen_size = None

                try:
                    while True:
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
                        
                        # Debug: Check frame properties
                        if frame_count == 1:
                            print(f"First frame: shape={frame.shape}, dtype={frame.dtype}")
                        
                        # Process frame
                        detections, inf_time = self._process_frame(frame)
                        
                        # Generate Grad-CAM (with additional safety check)
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
                                        gradcam_img = frame.copy()
                                        continue
                                
                                # Dynamically adjust frame skip threshold based on performance
                                if inf_time > 100:  # If inference is slow (>100ms)
                                    self.gradcam_processor.set_frame_skip_threshold(10)
                                elif inf_time < 50:  # If inference is fast (<50ms)
                                    self.gradcam_processor.set_frame_skip_threshold(3)
                                
                                gradcam_img = self.gradcam_processor.process_gradcam(
                                    frame, detections, self.display_config.gradcam_in_box_only
                                )
                            except Exception as e:
                                logging.warning(f"GradCAM processing failed: {e}, using original frame")
                                gradcam_img = frame.copy()
                        elif frame is not None:
                            logging.debug("GradCAM processor not available, using original frame")
                            gradcam_img = frame.copy()
                        else:
                            logging.warning("Frame is None, creating blank image")
                            gradcam_img = np.zeros((480, 640, 3), dtype=np.uint8)
                        
                        # Prepare display frame (with additional safety check)
                        if frame is not None:
                            display_img = self._prepare_display_frame(frame, detections, gradcam_img)
                        else:
                            logging.warning("Frame is None, creating blank display image")
                            display_img = np.zeros((480, 640, 3), dtype=np.uint8)
                        
                        # Create legend frame
                        legend_dict = self._create_legend_dict(detections)
                        monitors = get_monitors() if get_monitors is not None else None
                        self._create_legend_frame(legend_dict, inf_time, monitors)
                        
                        # Display frame
                        try:
                            first_frame, fullscreen_size = self._display_frame_fullscreen(window_name, display_img, first_frame, fullscreen_size)
                        except Exception as e:
                            logging.warning(f"Error displaying frame: {e}")
                            # Continue with next frame
                        
                        # Debug: Print frame info
                        if frame_count % 30 == 0:  # Every 30 frames
                            print(f"Frame {frame_count}: shape={display_img.shape}, detections={len(detections)}, inf_time={inf_time:.1f}ms")
                            print(f"Window state: first_frame={first_frame}, fullscreen_size={fullscreen_size}")
                        
                        # Handle key input with proper timing
                        try:
                            key = cv2.waitKey(max(1, wait_ms)) & 0xFF
                        except Exception as e:
                            logging.warning(f"Error handling key input: {e}")
                            key = 0
                        should_exit, should_next_video = self._handle_key_input(key, cap, out_writer)
                        
                        if should_exit:
                            self._cleanup_resources()
                            if out_writer:
                                out_writer.release()
                            cap.release()
                            return
                        elif should_next_video:
                            # Clean up current video resources but keep core components
                            self._cleanup_video_resources()
                            # Reset GradCAM state for next video
                            self._reset_gradcam_state()
                            # Reinitialize audio for next video
                            if hasattr(self, 'visualizer'):
                                self.visualizer.reinitialize_audio()
                            # Reinitialize button handler for next video
                            if hasattr(self, 'button_handler'):
                                if not self.button_handler.is_running:
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
            logging.info(f"Scaling: video={iw}x{ih}, screen={ww}x{wh}, mode={scale_mode}")
            
            # Apply scaling based on configuration
            if scale_mode == 'original':
                # Display original size
                new_w, new_h = iw, ih
                logging.info(f"Original mode: keeping video size {iw}x{ih}")
            elif scale_mode == 'stretch':
                # Stretch to fill entire window (may distort video)
                new_w, new_h = ww, wh
                logging.info(f"Stretch mode: stretching to screen size {ww}x{wh}")
            else:  # 'fit' mode (default)
                # Fit to fill screen while maintaining aspect ratio
                if maintain_aspect_ratio:
                    scale = min(ww / iw, wh / ih)
                else:
                    scale = max(ww / iw, wh / ih)
                # Apply multiplier to use more screen space
                scale = scale * scale_multiplier
                new_w, new_h = int(iw * scale), int(ih * scale)
                logging.info(f"Fit mode: scale={scale:.3f}, multiplier={scale_multiplier}, final={new_w}x{new_h}")
            
            # Resize the image
            resized_img = cv2.resize(display_img, (new_w, new_h))
            
            # Debug logging for final dimensions
            logging.info(f"Final dimensions: original={iw}x{ih}, scaled={new_w}x{new_h}, screen={ww}x{wh}")
            
            if center_video:
                # Center on black background
                fullscreen_img = np.zeros((wh, ww, 3), dtype=np.uint8)
                y_offset = (wh - new_h) // 2
                x_offset = (ww - new_w) // 2
                
                # Debug logging for positioning
                logging.info(f"Positioning: screen={ww}x{wh}, video={new_w}x{new_h}, offsets=({x_offset}, {y_offset})")
                
                # Ensure the video fits within the screen bounds
                if x_offset < 0 or y_offset < 0:
                    logging.warning(f"Video too large for screen: video={new_w}x{new_h}, screen={ww}x{wh}")
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