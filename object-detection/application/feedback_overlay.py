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
        self.debug_counter = 0
    
    def add_feedback(self, action_name: str = None, key_code: int = None, 
                    position: Tuple[int, int] = None, color: Tuple[int, int, int] = None,
                    radius: int = None, duration: float = None):
        """Add a new feedback overlay"""
        if not self.config['enabled']:
            return
        
        # Validate inputs - at least one of action_name or key_code should be provided
        if action_name is None and key_code is None:
            logging.warning("Cannot add feedback: neither action_name nor key_code provided")
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
        
        # Validate position is not None
        if position is None:
            logging.warning("Cannot add feedback: invalid position calculated")
            return
        
        # Determine radius and duration
        radius = radius or self.config['default_radius']
        duration = duration or self.config['total_duration']
        
        # Validate duration is positive
        if duration <= 0:
            logging.warning(f"Cannot add feedback: invalid duration {duration}")
            return
        
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
        """Calculate feedback position based on configuration and frame dimensions"""
        offset_x = self.config['offset_x']
        offset_y = self.config['offset_y']
        
        # If frame dimensions are set, use them for better positioning
        if hasattr(self, 'frame_width') and hasattr(self, 'frame_height'):
            # Ensure position is within frame bounds
            x = max(offset_x, 0)
            y = max(offset_y, 0)
            x = min(x, self.frame_width - 1)
            y = min(y, self.frame_height - 1)
            return (x, y)
        
        # Fallback to config offsets
        return (offset_x, offset_y)
    
    def update_feedbacks(self):
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
    
    def draw_feedbacks(self, frame: np.ndarray, debug_save: bool = False, debug_filename: str = None) -> np.ndarray:
        """Draw all active feedback overlays on the frame"""
        if not self.active_feedbacks:
            logging.debug("No active feedbacks to draw")
            return frame
        
        logging.debug(f"Drawing {len(self.active_feedbacks)} active feedbacks")
        
        # Update feedback states
        self.update_feedbacks()
        
        # Create a copy of the frame for overlays
        result_frame = frame.copy()
        
        for feedback in self.active_feedbacks:
            if 'current_alpha' not in feedback:
                logging.warning(f"Feedback missing current_alpha: {feedback}")
                continue
                
            position = feedback['position']
            color = feedback['color']
            radius = feedback['radius']
            alpha = feedback['current_alpha']
            
            # Ensure position is within frame bounds
            x, y = position
            if not (0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]):
                logging.warning(f"Feedback position {position} outside frame bounds {frame.shape}")
                continue
            
            logging.debug(f"Drawing feedback at {position} with color {color}, radius {radius}, alpha {alpha}")
            
            # Draw the feedback circle
            result_frame = self._draw_feedback_circle(
                result_frame, position, radius, color, alpha
            )
        
        # Save debug image if requested
        if debug_save:
            self._save_debug_image(result_frame, debug_filename)
        
        logging.debug(f"Finished drawing feedbacks, result frame shape: {result_frame.shape}")
        return result_frame
    
    def _draw_feedback_circle(self, frame: np.ndarray, position: Tuple[int, int], 
                             radius: int, color: Tuple[int, int, int], alpha: float) -> np.ndarray:
        """Draw a single feedback circle with alpha blending"""
        x, y = position
        
        logging.debug(f"Drawing circle at ({x}, {y}) with radius {radius}, color {color}, alpha {alpha}")
        
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
            logging.debug(f"Applied alpha blending, result shape: {result.shape}")
            return result.astype(np.uint8)
        else:
            # Additive blending
            alpha_mask = mask.astype(np.float32) / 255.0 * alpha
            alpha_mask = np.stack([alpha_mask] * 3, axis=2)
            
            result = frame.astype(np.float32) + overlay.astype(np.float32) * alpha_mask
            logging.debug(f"Applied additive blending, result shape: {result.shape}")
            return np.clip(result, 0, 255).astype(np.uint8)
    
    def _save_debug_image(self, frame: np.ndarray, filename: str = None):
        """Save debug image to file"""
        import os
        from datetime import datetime
        
        # Create debug directory if it doesn't exist
        debug_dir = "debug_images"
        os.makedirs(debug_dir, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"feedback_debug_{timestamp}_{self.debug_counter:04d}.jpg"
            self.debug_counter += 1
        else:
            # Ensure filename has proper extension
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filename += '.jpg'
        
        # Save the image
        filepath = os.path.join(debug_dir, filename)
        success = cv2.imwrite(filepath, frame)
        
        if success:
            logging.debug(f"Debug image saved: {filepath}")
        else:
            logging.warning(f"Failed to save debug image: {filepath}")
    
    def clear_all_feedbacks(self):
        """Clear all active feedbacks"""
        count = len(self.active_feedbacks)
        self.active_feedbacks.clear()
        logging.info(f"Cleared {count} active feedbacks")
    
    def clear_feedbacks_immediately(self):
        """Clear all feedbacks immediately without waiting for duration to expire"""
        count = len(self.active_feedbacks)
        self.active_feedbacks.clear()
        logging.info(f"Immediately cleared {count} active feedbacks")
    
    def get_feedback_info(self) -> List[Dict]:
        """Get detailed information about all active feedbacks for debugging"""
        return [
            {
                'action_name': f.get('action_name'),
                'key_code': f.get('key_code'),
                'position': f.get('position'),
                'elapsed': time.time() - f.get('start_time', 0),
                'duration': f.get('duration'),
                'alpha': f.get('current_alpha', 0)
            }
            for f in self.active_feedbacks
        ]
    
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
    
    def disable_feedback(self):
        """Temporarily disable feedback (circles won't appear)"""
        self.config['enabled'] = False
        logging.info("Feedback overlay disabled")
    
    def enable_feedback(self):
        """Re-enable feedback after disabling"""
        self.config['enabled'] = True
        logging.info("Feedback overlay enabled")
    
    def is_feedback_enabled(self) -> bool:
        """Check if feedback is currently enabled"""
        return self.config['enabled']