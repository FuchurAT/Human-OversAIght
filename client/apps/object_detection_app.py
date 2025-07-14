import os
import sys
import cv2
from pathlib import Path
import threading
import time
from typing import List
import numpy as np
from PIL import Image
import logging
from utils.screen.screen import ScreenApplication
from utils.log import Log
from ultralytics import YOLO

# Add the object-detection directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'object-detection'))

# Import DetectionVisualizer and GradCAM
from detection_visualizer import DetectionVisualizer
from grad_cam import GradCAM

# Define classes for object detection
CLASSES = [
    "ARMED_POLICEMEN", "CAR_FIRE", "FIRE", "FIRE_FIREFIGHTER", "FIRE_TRUCK",
    "HEALTH_AMBULANCE", "IMMIGRANT", "MILITARY_OFFICER", "MILITARY_SOLDIER",
    "MILITARY_VIECHLE", "POLICE", "POLICE_CAR", "POLICE_TRUCK", "POLICEMAN",
    "PRISON", "PROTEST", "RIOT", "RIOT_POLICE"
]


class ObjectDetectionApp(ScreenApplication):
    """Simplified screen application wrapper for object detection with DetectionVisualizer and GradCAM"""
    
    def __init__(self, screen):
        super().__init__(screen)
        self.video_path = "E:/Projects/human-oversaight/client/apps/object-detection/videos/us_capitol.mp4"
        self.model_path = "E:/Projects/human-oversaight/client/apps/object-detection/runs/train/weights/best.pt"
        self.current_frame_path = None
        self.is_running = False
        self.detection_thread = None
        self.model = None
        self.gradcam_enabled = True
        self.last_gradcam_img = None
        self.last_cleanup_time = time.time()
        self.cleanup_interval = 120  # Clean up every 2 minutes to reduce overhead
        self.pending_deletions = []  # Queue for async deletions
        self.cleanup_thread = None
        
        # Create output directory for frames
        self.output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Find video and model files
        self._find_resources()
        
        # Initialize visualizer and GradCAM
        self._initialize_components()
        
        # Start detection in background thread
        if self.video_path and self.model_path:
            self._start_detection()
            self._start_cleanup_thread()
            Log.info("SimpleObjectDetectionApp started with DetectionVisualizer and GradCAM support")
            Log.info("Available controls: toggle_gradcam(), toggle_visualization_mode()")
        else:
            Log.warning("Video or model not found. App will not start detection.")
    
    def _find_resources(self):
        """Find video and model files"""
        app_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Look for video files in common locations
        video_search_paths = [
            os.path.join(app_dir, "..", "..", "data", "videos"),
            os.path.join(app_dir, "..", "..", "data"),
            os.path.join(app_dir, "object-detection", "data"),
        ]
        
        for path in video_search_paths:
            if os.path.exists(path):
                for file in os.listdir(path):
                    if file.lower().endswith(('.mp4', '.avi', '.mov')):
                        self.video_path = os.path.join(path, file)
                        Log.info(f"Found video: {self.video_path}")
                        break
                if self.video_path:
                    break
        
        # Look for model file
        model_path = os.path.join(app_dir, "object-detection", "runs", "train", "weights", "best.pt")
        
        if os.path.exists(model_path):
            self.model_path = model_path
            Log.info(f"Found model: {self.model_path}")
    
    def _initialize_components(self):
        """Initialize DetectionVisualizer and GradCAM components"""
        try:
            # Initialize visualizer with reference to this app instance
            self.visualizer = DetectionVisualizer(CLASSES, self)
            Log.info("Initialized DetectionVisualizer")
            
            # Initialize GradCAM if model is available
            if self.model_path and os.path.exists(self.model_path):
                try:
                    # Add safe globals for PyTorch 2.6 compatibility
                    import torch
                    from ultralytics.nn.tasks import DetectionModel
                    from ultralytics.nn.modules.conv import Conv, Concat
                    from ultralytics.nn.modules.block import C2f, SPPF, Bottleneck, DFL
                    from ultralytics.nn.modules.head import Detect
                    from torch.nn.modules.container import Sequential, ModuleList
                    from torch.nn.modules.conv import Conv2d
                    from torch.nn.modules.batchnorm import BatchNorm2d
                    from torch.nn.modules.activation import SiLU
                    from torch.nn.modules.pooling import MaxPool2d
                    from torch.nn.modules.upsampling import Upsample
                    
                    torch.serialization.add_safe_globals([
                        DetectionModel, Detect, Sequential, Conv, Conv2d, MaxPool2d, BatchNorm2d, SiLU, C2f, Bottleneck, DFL, ModuleList, SPPF, Upsample, Concat,
                        # Add string-based globals for additional safety
                        'ultralytics.nn.tasks.DetectionModel',
                        'ultralytics.nn.modules.conv.Conv',
                        'ultralytics.nn.modules.conv.Concat',
                        'ultralytics.nn.modules.block.C2f',
                        'ultralytics.nn.modules.block.SPPF',
                        'ultralytics.nn.modules.block.Bottleneck',
                        'ultralytics.nn.modules.block.DFL',
                        'ultralytics.nn.modules.head.Detect'
                    ])
                    
                    # Load the YOLO model first
                    yolo_model = YOLO(self.model_path)
                    
                    # Get the underlying PyTorch model
                    pytorch_model = yolo_model.model
                    
                    # Find a suitable target layer (usually the last convolutional layer)
                    target_layer = None
                    for name, module in pytorch_model.named_modules():
                        if isinstance(module, torch.nn.Conv2d):
                            target_layer = module
                            # Use the last conv layer found
                    
                    if target_layer is None:
                        Log.warning("Could not find suitable target layer for GradCAM")
                        self.cam_model = None
                    else:
                        Log.info(f"Using target layer: {type(target_layer).__name__}")
                        # Pass model_path to try yolov8_heatmap first, fallback to custom implementation
                        self.cam_model = GradCAM(pytorch_model, target_layer, self.model_path)
                        Log.info("Initialized GradCAM with model path")
                        
                except ImportError:
                    Log.warning("Required modules not available, GradCAM will be disabled")
                    self.cam_model = None
                except Exception as e:
                    Log.warning(f"Failed to initialize GradCAM: {e}")
                    Log.warning("GradCAM will be disabled. Continuing with object detection only.")
                    self.cam_model = None
            else:
                self.cam_model = None
                
        except Exception as e:
            Log.error(f"Failed to initialize components: {e}")
            # Ensure visualizer is still available even if GradCAM fails
            if not hasattr(self, 'visualizer'):
                self.visualizer = DetectionVisualizer(CLASSES, self)
    

    def get_gradcam_image(self, frame):
        """Generate GradCAM image for the given frame using external GradCAM class"""
        if not hasattr(self, 'cam_model') or self.cam_model is None:
            return np.zeros_like(frame)
            
        try:
            # Convert frame to tensor format expected by GradCAM
            import torch
            
            # Preprocess frame for YOLO model
            # Convert BGR to RGB and normalize
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            
            # Generate GradCAM using the external GradCAM class
            gradcam_img = self.cam_model(frame_tensor)
            
            # Ensure the output is in the correct format
            if isinstance(gradcam_img, np.ndarray):
                if gradcam_img.shape[:2] != frame.shape[:2]:
                    gradcam_img = cv2.resize(gradcam_img, (frame.shape[1], frame.shape[0]))
                return gradcam_img
            else:
                return np.zeros_like(frame)
            
        except Exception as e:
            Log.warning(f"Grad-CAM generation failed: {e}")
            return np.zeros_like(frame)
    
    def _start_detection(self):
        """Start object detection in background thread"""
        try:
            # Try to import ultralytics
            self.model = YOLO(self.model_path)
            Log.info(f"Loaded YOLO model: {self.model_path}")
            
            # Start detection in background thread
            self.detection_thread = threading.Thread(target=self._run_detection)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            self.is_running = True
            
        except ImportError:
            Log.error("ultralytics package not found. Please install it: pip install ultralytics")
        except Exception as e:
            Log.error(f"Failed to start object detection: {e}")
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        try:
            self.cleanup_thread = threading.Thread(target=self._run_cleanup_loop)
            self.cleanup_thread.daemon = True
            self.cleanup_thread.start()
            Log.info("Started background cleanup thread")
        except Exception as e:
            Log.error(f"Failed to start cleanup thread: {e}")
    
    def _run_cleanup_loop(self):
        """Background thread for async file cleanup"""
        while self.is_running:
            try:
                # Process pending deletions with a small delay to ensure frames are read
                while self.pending_deletions:
                    file_path = self.pending_deletions.pop(0)
                    if os.path.exists(file_path):
                        # Add small delay to ensure screen system has time to read the frame
                        time.sleep(0.1)
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            Log.warning(f"Failed to delete {file_path}: {e}")
                
                # Periodic cleanup
                current_time = time.time()
                if current_time - self.last_cleanup_time > self.cleanup_interval:
                    self._periodic_cleanup()
                    self.last_cleanup_time = current_time
                
                time.sleep(1.0)  # Check every second
                
            except Exception as e:
                Log.warning(f"Error in cleanup thread: {e}")
                time.sleep(5.0)  # Wait longer on error
    
    def _run_detection(self):
        """Run object detection and save frames as images with DetectionVisualizer and GradCAM"""
        if not self.model:
            return
            
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                Log.error(f"Could not open video file: {self.video_path}")
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 15  # Reduced FPS for better performance
            wait_ms = int(1000 / fps)
            frame_count = 0

            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    # Loop video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                # Run inference
                inf_start = time.time()
                results = self.model(frame, conf=0.25)
                inf_time = (time.time() - inf_start) * 1000  # ms
                boxes = results[0].boxes
                
                # Debug: Log inference results
                if frame_count % 30 == 0:  # Log every 30 frames
                    Log.info(f"Frame {frame_count}: Inference time: {inf_time:.1f}ms, Raw boxes: {len(boxes) if boxes is not None else 0}")
                detections = []
                
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        conf = float(box.conf[0])
                        if conf < 0.1:  # box_threshold
                            continue
                        class_id = int(box.cls[0])
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        detections.append((tuple(xyxy), conf, class_id))
                
                # Debug: Log detection count
                if frame_count % 30 == 0:  # Log every 30 frames
                    Log.info(f"Frame {frame_count}: Found {len(detections)} detections")
                
                # Process detections with DetectionVisualizer
                overlay_boxes, legend_dict = self.visualizer.process_detections(detections, 0.1)
                
                # Debug: Log visualization info
                if frame_count % 30 == 0:  # Log every 30 frames
                    Log.info(f"Frame {frame_count}: Overlay boxes: {len(overlay_boxes)}, Legend items: {len(legend_dict)}")
                
                # Generate GradCAM less frequently to improve performance
                gradcam_img = np.zeros_like(frame)
                if self.visualizer.frame_idx % 30 == 0 and hasattr(self, 'cam_model') and self.cam_model and self.gradcam_enabled:
                    try:
                        # Only generate GradCAM if there are detections
                        if detections:
                            single_gradcam = self.get_gradcam_image(frame)
                            gradcam_img = single_gradcam
                            self.last_gradcam_img = gradcam_img
                    except Exception as e:
                        Log.warning(f"Grad-CAM failed on frame {self.visualizer.frame_idx}: {e}")
                        gradcam_img = self.last_gradcam_img if hasattr(self, 'last_gradcam_img') and self.last_gradcam_img is not None else np.zeros_like(frame)
                else:
                    gradcam_img = self.last_gradcam_img if hasattr(self, 'last_gradcam_img') and self.last_gradcam_img is not None else np.zeros_like(frame)
                
                # Draw detections on frame using DetectionVisualizer
                frame_with_overlays = frame.copy()
                self.visualizer.draw_detection_overlays(frame_with_overlays, overlay_boxes, legend_dict)
                self.visualizer.draw_fps_info(frame_with_overlays, inf_time)
                
                # Debug: Add a simple test box if no detections found
                if len(detections) == 0 and frame_count % 60 == 0:  # Every 60 frames
                    # Draw a test rectangle to verify drawing works
                    cv2.rectangle(frame_with_overlays, (50, 50), (150, 150), (0, 255, 0), 2)
                    cv2.putText(frame_with_overlays, "TEST BOX", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Always show detection boxes, optionally overlay with GradCAM
                if hasattr(self, 'cam_model') and self.cam_model:
                    final_frame = self.cam_model.create_final_frame(frame_with_overlays, gradcam_img, self.gradcam_enabled)
                else:
                    final_frame = frame_with_overlays
                
                # Save frame as image
                frame_path = os.path.join(self.output_dir, f"detection_frame_{frame_count:06d}.jpg")
                
                cv2.imwrite(frame_path, final_frame)
                
                # Queue previous frame for async deletion after setting new frame
                old_frame_path = self.current_frame_path
                self.current_frame_path = frame_path
                
                if old_frame_path and os.path.exists(old_frame_path):
                    self.pending_deletions.append(old_frame_path)
                
                # Update the screen
                self.update()
                
                frame_count += 1
                self.visualizer.frame_idx += 1
                
                time.sleep(wait_ms / 1000.0)
            
            cap.release()
            
        except Exception as e:
            Log.error(f"Error in detection thread: {e}")
    
    def handle_key_press(self, key):
        """Handle keyboard input for toggling features"""
        Log.info(f"Key pressed: {key} (char: {chr(key) if key < 128 else 'non-printable'})")
        
        # Handle specific keys
        if key == 32:  # Space bar
            Log.info("Space bar pressed - toggling visualization mode")
            self.toggle_visualization_mode()
        elif key == 71:  # 'g' key
            Log.info("G key pressed - toggling GradCAM")
            self.toggle_gradcam()
        elif key == ord('q'):  # 'q' key
            Log.info("Q key pressed - stopping detection")
            self.stop()
        else:
            # Forward to visualizer if it exists
            if hasattr(self, 'visualizer'):
                Log.info(f"Forwarding key {key} to visualizer")
                self.visualizer.handle_key_press(key)
    
    def toggle_gradcam(self):
        """Toggle GradCAM display mode"""
        if hasattr(self, 'gradcam_enabled'):
            self.gradcam_enabled = not self.gradcam_enabled
            Log.info(f"GradCAM {'enabled' if self.gradcam_enabled else 'disabled'}")
    
    def toggle_visualization_mode(self):
        """Toggle visualization mode (border style, colors, etc.)"""
        if hasattr(self, 'visualizer'):
            # Simulate space bar press
            self.visualizer.handle_key_press(32)
    
    def get_status(self):
        """Get current app status"""
        status = {
            'is_running': self.is_running,
            'gradcam_enabled': self.gradcam_enabled,
            'video_path': self.video_path,
            'model_path': self.model_path,
            'current_frame_path': self.current_frame_path,
            'has_visualizer': hasattr(self, 'visualizer'),
            'has_cam_model': hasattr(self, 'cam_model') and self.cam_model is not None
        }
        
        # Add storage information
        storage_info = self.get_storage_info()
        status.update(storage_info)
        
        if hasattr(self, 'visualizer'):
            status.update({
                'frame_idx': self.visualizer.frame_idx,
                'color_state': self.visualizer.color_state,
                'blur_boxes': self.visualizer.blur_boxes,
                'show_enemy': self.visualizer.show_enemy,
                'solid_border': self.visualizer.solid_border
            })
        return status
    
    def input(self) -> List[str]:
        """Return the current frame path for the screen system to display"""
        if self.current_frame_path and os.path.exists(self.current_frame_path):
            return [self.current_frame_path]
        else:
            # Return a placeholder image or empty list
            return []
    
    def update(self):
        """Update the screen display"""
        super().update()
    
    def stop(self):
        """Stop the detection thread"""
        self.is_running = False
        
        # Stop detection thread
        if self.detection_thread:
            self.detection_thread.join(timeout=1.0)
        
        # Stop cleanup thread
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=1.0)
        
        # Clean up temporary files
        self._cleanup_temp_files()
        
        # Clean up GradCAM if available
        if hasattr(self, 'cam_model') and self.cam_model:
            try:
                self.cam_model.close()
            except Exception as e:
                Log.warning(f"Error closing GradCAM: {e}")
    
    def _cleanup_temp_files(self):
        """Clean up temporary files created during detection"""
        try:
            # Remove current frame if it exists
            if self.current_frame_path and os.path.exists(self.current_frame_path):
                os.remove(self.current_frame_path)
                Log.info(f"Cleaned up frame: {self.current_frame_path}")
            
            # Remove temp frame if it exists
            if hasattr(self, 'temp_frame_path') and os.path.exists(self.temp_frame_path):
                os.remove(self.temp_frame_path)
                Log.info(f"Cleaned up temp frame: {self.temp_frame_path}")
            
            # Clean up any remaining detection frames in output directory
            if os.path.exists(self.output_dir):
                for file in os.listdir(self.output_dir):
                    if file.startswith("detection_frame_") and file.endswith(".jpg"):
                        file_path = os.path.join(self.output_dir, file)
                        try:
                            os.remove(file_path)
                            Log.info(f"Cleaned up detection frame: {file}")
                        except Exception as e:
                            Log.warning(f"Failed to clean up {file}: {e}")
                            
        except Exception as e:
            Log.warning(f"Error during cleanup: {e}")
    
    def _periodic_cleanup(self):
        """Periodic cleanup of old detection frames"""
        try:
            if not os.path.exists(self.output_dir):
                return
                
            current_time = time.time()
            max_age = 300  # Keep frames for 5 minutes max
            
            files_removed = 0
            for file in os.listdir(self.output_dir):
                if file.startswith("detection_frame_") and file.endswith(".jpg"):
                    file_path = os.path.join(self.output_dir, file)
                    
                    # Skip current frame
                    if file_path == self.current_frame_path:
                        continue
                    
                    # Check file age
                    try:
                        file_age = current_time - os.path.getmtime(file_path)
                        if file_age > max_age:
                            os.remove(file_path)
                            files_removed += 1
                    except Exception as e:
                        Log.warning(f"Failed to check/remove old file {file}: {e}")
            
            if files_removed > 0:
                Log.info(f"Periodic cleanup: removed {files_removed} old detection frames")
                
        except Exception as e:
            Log.warning(f"Error during periodic cleanup: {e}")
    
    def get_storage_info(self):
        """Get information about storage usage"""
        try:
            if not os.path.exists(self.output_dir):
                return {"total_files": 0, "total_size_mb": 0}
            
            total_size = 0
            file_count = 0
            
            for file in os.listdir(self.output_dir):
                if file.startswith("detection_frame_") and file.endswith(".jpg"):
                    file_path = os.path.join(self.output_dir, file)
                    try:
                        total_size += os.path.getsize(file_path)
                        file_count += 1
                    except:
                        pass
            
            return {
                "total_files": file_count,
                "total_size_mb": round(total_size / (1024 * 1024), 2)
            }
            
        except Exception as e:
            Log.warning(f"Error getting storage info: {e}")
            return {"total_files": 0, "total_size_mb": 0} 