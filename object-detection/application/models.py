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
    gradcam_in_box_only: bool = True
    blur_boxes: bool = True
    color_state: str = 'blue'
    show_enemy: bool = False
    solid_border: bool = False 