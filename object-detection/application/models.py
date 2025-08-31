"""
Data models and classes for the object detection application.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Detection:
    """Data class for storing detection information"""
    box: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    process_id: int = 0


@dataclass
class DisplayConfig:
    """Configuration for display settings"""
    show_legend: bool = False
    show_fps_info: bool = False
    gradcam_enabled: bool = False
    enable_glitches: bool = False
    gradcam_in_box_only: bool = False
    blur_boxes: bool = True
    color_state: str = 'blue'
    show_enemy: bool = False
    solid_border: bool = False
    # NDI Settings
    ndi_enabled: bool = False
    ndi_source_name: str = "Human-OversAIght-Detection"
    ndi_group_name: str = ""
    ndi_video_format: str = "BGRX"
    ndi_frame_rate: int = 30
    ndi_video_width: int = 1920
    ndi_video_height: int = 1080 