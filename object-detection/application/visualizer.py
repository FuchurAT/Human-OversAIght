"""
Visualization utilities for the object detection application.
"""

import cv2
import numpy as np
import time
from typing import Dict, Tuple, List
from config.config import (
    DEFAULT_ANIMATION_ALPHA, DEFAULT_CORNER_LENGTH, DEFAULT_EDGE_MID_LENGTH,
    DEFAULT_GLOW_THICKNESS, DEFAULT_GLOW_ALPHA, DEFAULT_CROSS_LENGTH,
    DEFAULT_CROSS_THICKNESS, DEFAULT_PIXEL_SIZE_DIVISOR, MIN_PIXEL_SIZE,
    FPS_UPDATE_INTERVAL, FPS_Y_OFFSET, FPS_LINE_SPACING
)

from application.models import Detection, DisplayConfig
from application.color_manager import ColorManager


class DetectionVisualizer:
    """Handles visualization of detections and overlays"""
    
    def __init__(self, display_config: DisplayConfig, classes: List[str]):
        self.display_config = display_config
        self.classes = classes
        self.frame_idx = 0
        self.animation_alpha = DEFAULT_ANIMATION_ALPHA
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        self.last_fps_text = "FPS: 0.0"
        self.last_inf_text = "Inference: 0.0 ms"
    
    def draw_detection_overlays(self, frame: np.ndarray, detections: List[Detection]) -> None:
        """Draw detection overlays on the frame"""
        for detection in detections:
            color = ColorManager.get_state_color(self.display_config.color_state, detection.confidence)
            
            if self.display_config.solid_border:
                x1, y1, x2, y2 = detection.box
                corner_length = DEFAULT_CORNER_LENGTH
                thickness = 4
            else:
                x1, y1, x2, y2 = detection.box
                corner_length = DEFAULT_CORNER_LENGTH
                thickness = 3
                
                if self.display_config.blur_boxes:
                    self._apply_blur_effect(frame, detection.box)
            
            self._draw_box_corners(frame, (x1, y1, x2, y2), color, corner_length, thickness)
            self._draw_box_edges(frame, (x1, y1, x2, y2), color, thickness)
            self._apply_glow_effect(frame, detection.box, color)
            
            if self.display_config.show_enemy:
                self._draw_enemy_indicators(frame, detection.box, color)
    
    def _apply_blur_effect(self, frame: np.ndarray, box: Tuple[int, int, int, int]) -> None:
        """Apply pixelation effect to the detection box"""
        x1, y1, x2, y2 = box
        roi = frame[y1:y2, x1:x2]
        h, w = roi.shape[:2]
        if h > 0 and w > 0:
            pixel_size = max(MIN_PIXEL_SIZE, min(h, w) // DEFAULT_PIXEL_SIZE_DIVISOR)
            temp = cv2.resize(roi, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
            pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
            frame[y1:y2, x1:x2] = pixelated
    
    def _draw_box_corners(self, frame: np.ndarray, box: Tuple[int, int, int, int], 
                          color: Tuple[int, int, int], corner_length: int, thickness: int) -> None:
        """Draw corner indicators for the detection box"""
        x1, y1, x2, y2 = box
        
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, thickness)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, thickness)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, thickness)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, thickness)
    
    def _draw_box_edges(self, frame: np.ndarray, box: Tuple[int, int, int, int], 
                        color: Tuple[int, int, int], thickness: int) -> None:
        """Draw edge indicators for the detection box"""
        x1, y1, x2, y2 = box
        mid_length = DEFAULT_EDGE_MID_LENGTH
        
        # Top edge
        mid_top_x1 = x1 + (x2 - x1) // 2 - mid_length // 2
        mid_top_x2 = mid_top_x1 + mid_length
        cv2.line(frame, (mid_top_x1, y1), (mid_top_x2, y1), color, thickness)
        
        # Bottom edge
        mid_bot_x1 = x1 + (x2 - x1) // 2 - mid_length // 2
        mid_bot_x2 = mid_bot_x1 + mid_length
        cv2.line(frame, (mid_bot_x1, y2), (mid_bot_x2, y2), color, thickness)
        
        # Left edge
        mid_left_y1 = y1 + (y2 - y1) // 2 - mid_length // 2
        mid_left_y2 = mid_left_y1 + mid_length
        cv2.line(frame, (x1, mid_left_y1), (x1, mid_left_y2), color, thickness)
        
        # Right edge
        mid_right_y1 = y1 + (y2 - y1) // 2 - mid_length // 2
        mid_right_y2 = mid_right_y1 + mid_length
        cv2.line(frame, (x2, mid_right_y1), (x2, mid_right_y2), color, thickness)
    
    def _apply_glow_effect(self, frame: np.ndarray, box: Tuple[int, int, int], 
                           color: Tuple[int, int, int]) -> None:
        """Apply glow effect around the detection box"""
        overlay = frame.copy()
        glow_color = (255, 255, 0) if self.display_config.color_state != 'red' else (0, 0, 255)
        cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), glow_color, DEFAULT_GLOW_THICKNESS)
        alpha = DEFAULT_GLOW_ALPHA
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    def _draw_enemy_indicators(self, frame: np.ndarray, box: Tuple[int, int, int], 
                              color: Tuple[int, int, int]) -> None:
        """Draw enemy indicators (TARGET text and cross)"""
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, "TARGET", (box[0], box[1] - 10), font, 0.8, color, 2)
        
        # Draw target cross
        cx = (box[0] + box[2]) // 2
        cy = (box[1] + box[3]) // 2
        cross_length = DEFAULT_CROSS_LENGTH
        cross_thickness = DEFAULT_CROSS_THICKNESS
        cv2.line(frame, (cx - cross_length // 2, cy), (cx + cross_length // 2, cy), color, cross_thickness)
        cv2.line(frame, (cx, cy - cross_length // 2), (cx, cy + cross_length // 2), color, cross_thickness)
    
    def draw_random_glitches(self, frame: np.ndarray) -> None:
        """Draw random glitch effects on the frame"""
        h, w = frame.shape[:2]
        y = np.random.randint(h // 16, 3 * h // 4)
        
        points = []
        for x in range(0, w, 1):
            offset = np.random.randint(-5, 6)
            points.append((x, min(max(y + offset, 0), h - 1)))
        
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], (0, 0, 0), 3)
    
    def draw_legend(self, frame: np.ndarray, legend_dict: Dict[int, Tuple[float, Tuple[int, int, int]]], 
                    center: bool = False, font_scale: float = 0.7, line_height: int = 25, 
                    box_padding: int = 10) -> None:
        """Draw legend on the frame"""
        legend_lines = []
        legend_colors = []
        
        for class_id, (conf, color) in legend_dict.items():
            class_name = self.classes[class_id] if class_id < len(self.classes) else f"Class_{class_id}"
            legend_lines.append(f"{class_name}: {conf:.2f}")
            legend_colors.append(color)
        
        if legend_lines:
            font = cv2.FONT_HERSHEY_DUPLEX
            if center:
                text_x = frame.shape[1] // 2 - (len(legend_lines) * line_height * font_scale * 0.7 + box_padding * 2) // 2
                text_y = frame.shape[0] // 2 - (len(legend_lines) * line_height * font_scale * 0.7 + box_padding * 2) // 2
            else:
                text_x = box_padding
                text_y = box_padding

            for idx, line in enumerate(legend_lines):
                lcolor = legend_colors[idx]
                cv2.putText(
                    frame,
                    line,
                    (int(text_x), int(text_y + idx * line_height * font_scale)),
                    font,
                    font_scale,
                    lcolor,
                    2,
                    cv2.LINE_AA
                )
    
    def draw_fps_info(self, frame: np.ndarray, inf_time: float, center: bool = False, 
                      font_scale: float = 0.8, y_offset: int = FPS_Y_OFFSET) -> None:
        """Draw FPS and inference time information"""
        self.frame_count += 1
        if self.frame_idx % FPS_UPDATE_INTERVAL == 0:
            now = time.time()
            if now - self.last_fps_time > 1.0:
                self.current_fps = self.frame_count / (now - self.last_fps_time)
                self.last_fps_time = now
                self.frame_count = 0
            fps_text = f"FPS: {self.current_fps:.1f}"
            inf_text = f"Inference: {inf_time:.1f} ms"
            self.last_fps_text = fps_text
            self.last_inf_text = inf_text
        else:
            fps_text = self.last_fps_text
            inf_text = self.last_inf_text

        if center:
            parts = fps_text.split(':')
            text_x = frame.shape[1] // 2 - int((len(parts[0]) + len(parts[1]) + 2) * font_scale * 0.7 // 2)
            text_y = y_offset
        else:
            text_x = 15
            text_y = y_offset

        cv2.putText(frame, fps_text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0,0,255), 2)
        cv2.putText(frame, inf_text, (text_x, text_y + FPS_LINE_SPACING), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0,255,255), 2)
    
    def handle_key_press(self, key: int, app_instance) -> None:
        """Handle keyboard input for visualization controls"""
        from config.config import ENABLE_UNMASK, ENABLE_GRAD_CAM_VIEW
        
        if key == 32 and ENABLE_UNMASK:  # Space bar
            self.display_config.blur_boxes = not self.display_config.blur_boxes
            if self.display_config.color_state == 'red':
                self.display_config.color_state = 'yellow_orange'
            else:
                self.display_config.color_state = 'red'
            self.display_config.show_enemy = not self.display_config.show_enemy
            self.display_config.solid_border = not self.display_config.solid_border
        elif key == ord('g') and ENABLE_GRAD_CAM_VIEW:  # 'g' key to toggle Grad-CAM
            if app_instance:
                app_instance.display_config.gradcam_enabled = not app_instance.display_config.gradcam_enabled
                print(f"Grad-CAM {'enabled' if app_instance.display_config.gradcam_enabled else 'disabled'}")
        elif key == ord('w'):
            app_instance.display_config.enable_glitches = not app_instance.display_config.enable_glitches
            print(f"Glitches {'enabled' if app_instance.display_config.enable_glitches else 'disabled'}")
    
    def update_display_config(self, new_config: DisplayConfig) -> None:
        """Update the display configuration"""
        self.display_config = new_config
    
    def get_current_fps(self) -> float:
        """Get the current FPS value"""
        return self.current_fps
    
    def reset_fps_counter(self) -> None:
        """Reset the FPS counter"""
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0 