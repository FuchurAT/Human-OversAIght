"""
Object Detection Inference Package

A modular, refactored object detection application with YOLO integration,
Grad-CAM visualization, and multi-monitor support.
"""

from .models import Detection, DisplayConfig
from .memory_manager import MemoryManager
from .color_manager import ColorManager
from .box_manager import BoxManager
from .gradcam_processor import GradCAMProcessor
from .visualizer import DetectionVisualizer
from .app import VideoInferenceApp
from .utils import (
    spawn_fullscreen_videos_on_other_screens,
    get_monitor_info,
    create_fullscreen_window,
    resize_frame_for_monitor
)

__version__ = "2.0.0"
__author__ = "Human Oversaight Team"

__all__ = [
    # Core classes
    'Detection',
    'DisplayConfig',
    'VideoInferenceApp',
    
    # Utility classes
    'MemoryManager',
    'ColorManager',
    'BoxManager',
    'GradCAMProcessor',
    'DetectionVisualizer',
    
    # Utility functions
    'spawn_fullscreen_videos_on_other_screens',
    'get_monitor_info',
    'create_fullscreen_window',
    'resize_frame_for_monitor',
] 