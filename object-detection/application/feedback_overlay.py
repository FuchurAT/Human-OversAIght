"""
Feedback Overlay Module for Visual Feedback
Handles visual feedback overlays when buttons or keys are pressed.
"""

import cv2
import numpy as np
import time
import logging
from typing import Tuple, List, Dict
from config.config import FEEDBACK_CONFIG, ACTION_FEEDBACK_COLORS, KEY_FEEDBACK_COLORS


class FeedbackOverlay:
    """Handles visual feedback overlays for button and key presses"""
    
    def __init__(self):
        self.active_feedbacks = []
        self.config = FEEDBACK_CONFIG.copy()
        self.action_colors = ACTION_FEEDBACK_COLORS.copy()
        self.key_colors = KEY_FEEDBACK_COLORS.copy()
    
    def add_feedback(self, action_name: str = None, key_code: int = None, 
                    position: Tuple[int, int] = None, color: Tuple[int, int, int] = None,
                    radius: int = None, duration: float = None):
        """Add a new feedback overlay"""
        if not self.config['enabled']:
            return
        
        # Determine color
        if color is None:
            if action_name and action_name in self.action_colors:
                color = self.action_colors[action_name]
            elif key_code and key_code in self.key_colors:
                color = self.key_colors[key_code]
            else:
                color = self.config['default_color']
        
        # Determine position
        if position is None:
            position = self._calculate_position()
        
        # Determine radius and duration
        radius = radius or self.config['default_radius']
        duration = duration or self.config['total_duration']
        
        # Create feedback object
        feedback = {
            'action_name': action_name,
            'key_code': key_code,
            'position': position,
            'color': color,
            'radius': radius,
            'start_time': time.time(),
            'duration': duration,
            'fade_in_duration': self.config['fade_in_duration'],
            'fade_out_duration': self.config['fade_out_duration'],
            'min_alpha': self.config['min_alpha'],
            'max_alpha': self.config['max_alpha']
        }
        
        self.active_feedbacks.append(feedback)
        logging.debug(f"Added feedback: {action_name or key_code} at {position} with color {color}")
    
    def _calculate_position(self) -> Tuple[int, int]:
        """Calculate feedback position based on configuration"""
        offset_x = self.config['offset_x']
        offset_y = self.config['offset_y']
        
        return (offset_x, offset_y) 
    
    def update_feedbacks(self, frame_shape: Tuple[int, int, int]):
        """Update feedback states and remove expired ones"""
        current_time = time.time()
        expired_feedbacks = []
        
        for feedback in self.active_feedbacks:
            elapsed = current_time - feedback['start_time']
            
            if elapsed >= feedback['duration']:
                expired_feedbacks.append(feedback)
            else:
                # Calculate current alpha based on fade in/out
                if elapsed < feedback['fade_in_duration']:
                    # Fade in phase
                    progress = elapsed / feedback['fade_in_duration']
                    feedback['current_alpha'] = feedback['min_alpha'] + (feedback['max_alpha'] - feedback['min_alpha']) * progress
                elif elapsed > (feedback['duration'] - feedback['fade_out_duration']):
                    # Fade out phase
                    fade_out_elapsed = elapsed - (feedback['duration'] - feedback['fade_out_duration'])
                    progress = fade_out_elapsed / feedback['fade_out_duration']
                    feedback['current_alpha'] = feedback['max_alpha'] - (feedback['max_alpha'] - feedback['min_alpha']) * progress
                else:
                    # Full opacity phase
                    feedback['current_alpha'] = feedback['max_alpha']
        
        # Remove expired feedbacks
        for feedback in expired_feedbacks:
            self.active_feedbacks.remove(feedback)
    
    def draw_feedbacks(self, frame: np.ndarray) -> np.ndarray:
        """Draw all active feedback overlays on the frame"""
        if not self.active_feedbacks:
            return frame
        
        # Update feedback states
        self.update_feedbacks(frame.shape)
        
        # Create a copy of the frame for overlays
        result_frame = frame.copy()
        
        for feedback in self.active_feedbacks:
            if 'current_alpha' not in feedback:
                continue
                
            position = feedback['position']
            color = feedback['color']
            radius = feedback['radius']
            alpha = feedback['current_alpha']
            
            # Ensure position is within frame bounds
            x, y = position
            if not (0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]):
                continue
            
            # Draw the feedback circle
            result_frame = self._draw_feedback_circle(
                result_frame, position, radius, color, alpha
            )
        
        return result_frame
    
    def _draw_feedback_circle(self, frame: np.ndarray, position: Tuple[int, int], 
                             radius: int, color: Tuple[int, int, int], alpha: float) -> np.ndarray:
        """Draw a single feedback circle with alpha blending"""
        x, y = position
        
        # Create a mask for the circle
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), radius, 255, -1)
        
        # Create colored overlay
        overlay = np.zeros_like(frame)
        overlay[mask > 0] = color
        
        # Apply alpha blending
        if self.config['blend_mode'] == 'alpha':
            # Alpha blending
            alpha_mask = mask.astype(np.float32) / 255.0 * alpha
            alpha_mask = np.stack([alpha_mask] * 3, axis=2)
            
            result = frame.astype(np.float32) * (1 - alpha_mask) + overlay.astype(np.float32) * alpha_mask
            return result.astype(np.uint8)
        else:
            # Additive blending
            alpha_mask = mask.astype(np.float32) / 255.0 * alpha
            alpha_mask = np.stack([alpha_mask] * 3, axis=2)
            
            result = frame.astype(np.float32) + overlay.astype(np.float32) * alpha_mask
            return np.clip(result, 0, 255).astype(np.uint8)
    
    def clear_all_feedbacks(self):
        """Clear all active feedbacks"""
        self.active_feedbacks.clear()
    
    def get_feedback_count(self) -> int:
        """Get the number of active feedbacks"""
        return len(self.active_feedbacks)
    
    def set_config(self, new_config: dict):
        """Update feedback configuration"""
        self.config.update(new_config)
    
    def set_action_colors(self, new_colors: dict):
        """Update action-specific colors"""
        self.action_colors.update(new_colors)
    
    def set_key_colors(self, new_colors: dict):
        """Update key-specific colors"""
        self.key_colors.update(new_colors)
    
    def set_frame_dimensions(self, width: int, height: int):
        """Update frame dimensions for better position calculations"""
        self.frame_width = width
        self.frame_height = height
    
    def _calculate_position(self) -> Tuple[int, int]:
        """Calculate feedback position based on configuration and frame dimensions"""
        position_type = self.config['position']
        offset_x = self.config['offset_x']
        offset_y = self.config['offset_y']
        
        # Get frame dimensions (default to 1280x720 if not set)
        width = getattr(self, 'frame_width', 1280)
        height = getattr(self, 'frame_height', 720)
        
        if position_type == 'center':
            x = width // 2 + offset_x
            y = height // 2 + offset_y
        elif position_type == 'top_left':
            x = 100 + offset_x
            y = 100 + offset_y
        elif position_type == 'top_right':
            x = width - 100 + offset_x
            y = 100 + offset_y
        elif position_type == 'bottom_left':
            x = 100 + offset_x
            y = height - 100 + offset_y
        elif position_type == 'bottom_right':
            x = width - 100 + offset_x
            y = height - 100 + offset_y
        elif position_type == 'random':
            import random
            x = random.randint(100, width - 100) + offset_x
            y = random.randint(100, height - 100) + offset_y
        else:
            # Default to center
            x = width // 2 + offset_x
            y = height // 2 + offset_y
        
        # Ensure position is within bounds
        x = max(50, min(x, width - 50))
        y = max(50, min(y, height - 50))
        
        return (x, y) 