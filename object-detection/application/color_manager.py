"""
Color management utilities for the object detection application.
"""

from typing import Tuple


class ColorManager:
    """Manages color schemes and confidence-based coloring"""
    
    @staticmethod
    def confidence_to_color(conf: float) -> Tuple[int, int, int]:
        """Convert confidence to color (0.0 = orange, 1.0 = yellow)"""
        b = 255
        g = int(165 + (255 - 105) * conf)
        r = 0
        return (b, g, r)
    
    @staticmethod
    def get_state_color(color_state: str, confidence: float) -> Tuple[int, int, int]:
        """Get color based on current state and confidence"""
        if color_state == 'red':
            return (0, 0, 255)
        return ColorManager.confidence_to_color(confidence)
    
    @staticmethod
    def interpolate_color(color1: Tuple[int, int, int], 
                         color2: Tuple[int, int, int], 
                         alpha: float) -> Tuple[int, int, int]:
        """Interpolate between two colors"""
        return tuple([int(a + (b - a) * alpha) for a, b in zip(color1, color2)])
    
    @staticmethod
    def get_contrast_color(color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Get a contrasting color for text overlays"""
        # Simple luminance-based contrast
        luminance = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
        if luminance > 128:
            return (0, 0, 0)  # Black for light backgrounds
        else:
            return (255, 255, 255)  # White for dark backgrounds 