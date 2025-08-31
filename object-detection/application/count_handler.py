"""
Count Handler Module
Handles counting of button presses and key presses, displays the count on the output frame,
and persists the count to a text file for restoration on startup.
"""

import cv2
import logging
import os
from pathlib import Path
from typing import Dict, Optional
from config.config import LOGGING_CONFIG, COUNTER_CONFIG


class CountHandler:
    """
    Manages counting of button presses and key presses.
    Displays the count on the output frame and persists to a text file.
    """
    
    def __init__(self, count_file_path: str = "count.txt"):
        self.count_file_path = Path(count_file_path)
        self.counts = {
            'button_presses': 0,
            'key_presses': 0,
            'total_presses': 0
        }
        
        # Display settings, now using COUNTER_CONFIG
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = COUNTER_CONFIG.get('font_scale', 1.0)
        self.font_thickness = COUNTER_CONFIG.get('font_thickness', 2)
        self.text_color = COUNTER_CONFIG.get('text_color', (255, 255, 255))  # White
        self.bg_color = COUNTER_CONFIG.get('background_color', (0, 0, 0))  # Black
        self.bg_alpha = COUNTER_CONFIG.get('background_alpha', 0.7)
        
        # Position settings, now using COUNTER_CONFIG
        self.x_offset = COUNTER_CONFIG.get('x_offset', 20)
        self.y_offset = COUNTER_CONFIG.get('y_offset', 50)
        self.line_height = COUNTER_CONFIG.get('line_height', 30)
        
        self._load_count()
        
        logging.info(f"CountHandler initialized. Current counts: {self.counts}")
    
    def _load_count(self):
        """Load count from the text file if it exists"""
        try:
            if self.count_file_path.exists():
                with open(self.count_file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = int(value.strip())
                            if key in self.counts:
                                self.counts[key] = value
                
                # Update total
                self.counts['total_presses'] = self.counts['button_presses'] + self.counts['key_presses']
                logging.info(f"Loaded counts from {self.count_file_path}: {self.counts}")
            else:
                logging.info(f"Count file {self.count_file_path} not found. Starting with zero counts.")
        except Exception as e:
            logging.error(f"Error loading count from {self.count_file_path}: {e}")
    
    def _save_count(self):
        """Save current count to the text file"""
        try:
            with open(self.count_file_path, 'w') as f:
                for key, value in self.counts.items():
                    f.write(f"{key}={value}\n")
            logging.debug(f"Saved counts to {self.count_file_path}")
        except Exception as e:
            logging.error(f"Error saving count to {self.count_file_path}: {e}")
    
    def increment_button_press(self):
        """Increment button press count"""
        self.counts['button_presses'] += 1
        self.counts['total_presses'] += 1
        self._save_count()
        logging.debug(f"Button press count incremented: {self.counts['button_presses']}")
    
    def increment_key_press(self):
        """Increment key press count"""
        self.counts['key_presses'] += 1
        self.counts['total_presses'] += 1
        self._save_count()
        logging.debug(f"Key press count incremented: {self.counts['key_presses']}")
    
    def reset_counts(self):
        """Reset all counts to zero"""
        self.counts = {
            'button_presses': 0,
            'key_presses': 0,
            'total_presses': 0
        }
        self._save_count()
        logging.info("All counts reset to zero")
    
    def get_counts(self) -> Dict[str, int]:
        """Get current counts"""
        return self.counts.copy()
    
    def draw_counter_on_frame(self, frame: cv2.Mat, position: str = "top_left") -> cv2.Mat:
        """
        Draw the counter information on the output frame
        
        Args:
            frame: The frame to draw on
            position: Position to draw the counter ("top_left", "top_right", "bottom_left", "bottom_right")
        
        Returns:
            The frame with counter drawn on it
        """
        if frame is None:
            return frame
        
        # Calculate position based on frame size and desired location
        height, width = frame.shape[:2]
        
        if position == "top_right":
            x_start = width - 200
            y_start = self.y_offset
        elif position == "bottom_left":
            x_start = self.x_offset
            y_start = height - 100
        elif position == "bottom_right":
            x_start = width - 200
            y_start = height - 100
        else:  # top_left (default)
            x_start = self.x_offset
            y_start = self.y_offset
        
        # Create background rectangle for better visibility
        bg_rect = frame.copy()
        cv2.rectangle(
            bg_rect,
            (x_start - 10, y_start - 30),
            (x_start + 180, y_start + 80),
            self.bg_color,
            -1
        )
        
        # Blend background with original frame
        cv2.addWeighted(bg_rect, self.bg_alpha, frame, 1 - self.bg_alpha, 0, frame)
        
        # Draw counter text
        lines = [
            f"{self.counts['total_presses']}"
        ]
        
        for i, line in enumerate(lines):
            y_pos = y_start + (i * self.line_height)
            cv2.putText(
                frame,
                line,
                (x_start, y_pos),
                self.font,
                self.font_scale,
                self.text_color,
                self.font_thickness
            )
        
        return frame
    
    def set_display_position(self, x_offset: int, y_offset: int):
        """Set the display position offset"""
        self.x_offset = x_offset
        self.y_offset = y_offset
    
    def set_font_properties(self, font_scale: float = None, thickness: int = None, color: tuple = None):
        """Set font display properties"""
        if font_scale is not None:
            self.font_scale = font_scale
        if thickness is not None:
            self.font_thickness = thickness
        if color is not None:
            self.text_color = color
    
    def set_background_properties(self, color: tuple = None, alpha: float = None):
        """Set background display properties"""
        if color is not None:
            self.bg_color = color
        if alpha is not None:
            self.bg_alpha = max(0.0, min(1.0, alpha))  # Clamp between 0 and 1
