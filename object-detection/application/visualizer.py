"""
Visualization utilities for the object detection application.
"""

import cv2
import numpy as np
import time
import pygame
import os
from pathlib import Path
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
        
        # Initialize pygame mixer for audio playback
        self.audio_available = False
        self.vocal_sound = None
        self.ambient_sounds = {}
        self.current_ambient = None
        self.ambient_volume = 0.3  # Default volume for ambient sounds
        self.ambient_cycle_timer = 0
        self.ambient_cycle_interval = 30  # Cycle every 30 seconds
        self.ambient_sound_index = 0
        
        try:
            pygame.mixer.init()
            self.audio_available = True
            print("Pygame mixer initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize pygame mixer: {e}")
            # Try alternative audio drivers
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                self.audio_available = True
                print("Pygame mixer initialized with alternative settings")
            except Exception as e2:
                print(f"Alternative initialization also failed: {e2}")
                self.audio_available = False
        
        # Get the path to vocal.wav relative to the project root
        project_root = Path(__file__).parent.parent.parent
        self.vocal_sound_path = project_root / "assets" / "sounds" / "buttons" / "vocal.wav"
        
        # Initialize button sound (only if audio is available)
        if self.audio_available and self.vocal_sound_path.exists():
            try:
                self.vocal_sound = pygame.mixer.Sound(str(self.vocal_sound_path))
                print("Button sound (vocal.wav) loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load button sound: {e}")
                self.vocal_sound = None
        
        # Initialize ambient sound paths
        self.ambient_sounds_dir = project_root / "assets" / "sounds" / "ambient"
        
        # Load ambient sound files (only if audio is available)
        if self.audio_available:
            self._load_ambient_sounds()
    
    def _load_ambient_sounds(self) -> None:
        """Load all available ambient sound files"""
        print(f"Loading ambient sounds from: {self.ambient_sounds_dir}")
        print(f"Directory exists: {self.ambient_sounds_dir.exists()}")
        print(f"Audio available: {self.audio_available}")
        
        if not self.audio_available:
            print("Audio not available, skipping ambient sound loading")
            return
        
        # Additional safety check
        if not self.is_audio_safe():
            print("Audio not safe, disabling audio functionality")
            self.disable_audio()
            return
            
        if not self.ambient_sounds_dir.exists():
            print(f"Ambient sounds directory does not exist: {self.ambient_sounds_dir}")
            return
            
        try:
            # List all files in the directory
            all_files = list(self.ambient_sounds_dir.iterdir())
            print(f"All files in ambient directory: {[f.name for f in all_files]}")
            
            # Load all .wav files from the ambient directory
            wav_files = list(self.ambient_sounds_dir.glob("*.wav"))
            print(f"Found WAV files: {[f.name for f in wav_files]}")
            
            for sound_file in wav_files:
                try:
                    sound_name = sound_file.stem
                    print(f"Attempting to load: {sound_file}")
                    # Load as pygame.mixer.Sound for compatibility
                    self.ambient_sounds[sound_name] = pygame.mixer.Sound(str(sound_file))
                    print(f"Successfully loaded ambient sound: {sound_name}")
                except Exception as e:
                    print(f"Warning: Could not load ambient sound {sound_file.name}: {e}")
            
            print(f"Loaded {len(self.ambient_sounds)} ambient sound(s)")
            print(f"Ambient sounds loaded: {list(self.ambient_sounds.keys())}")
            
            # Automatically start playing the first available ambient sound
            if self.ambient_sounds:
                self.ambient_sound_index = 0
                first_sound = list(self.ambient_sounds.keys())[0]
                print(f"Starting first ambient sound: {first_sound}")
                self.start_ambient_sound(first_sound)
            else:
                print("No ambient sounds loaded, cannot start playback")
            
        except Exception as e:
            print(f"Warning: Error loading ambient sounds: {e}")
            import traceback
            traceback.print_exc()
    
    def start_ambient_sound(self, sound_name: str = None, volume: float = None) -> bool:
        """Start playing ambient sound in background loop"""
        if not self.audio_available or not self.ambient_sounds:
            print("No ambient sounds available")
            return False
        
        def _start_sound():
            # Stop any currently playing ambient sound
            self.stop_ambient_sound()
            
            # If no specific sound specified, use the first available one
            if sound_name is None:
                if not self.ambient_sounds:
                    print("No ambient sounds loaded")
                    return False
                sound_name = sound_name or list(self.ambient_sounds.keys())[0]
            
            if sound_name not in self.ambient_sounds:
                print(f"Ambient sound '{sound_name}' not found. Available: {list(self.ambient_sounds.keys())}")
                return False
            
            sound_object = self.ambient_sounds[sound_name]
            
            # Set volume if specified
            if volume is not None:
                self.ambient_volume = max(0.0, min(1.0, volume))
            
            # Set volume for the sound object
            sound_object.set_volume(self.ambient_volume)
            
            # Play the ambient sound in a loop
            sound_object.play(-1)  # -1 means loop indefinitely
            
            self.current_ambient = sound_name
            # Update the index to match the current sound
            if sound_name in self.ambient_sounds:
                sound_names = list(self.ambient_sounds.keys())
                self.ambient_sound_index = sound_names.index(sound_name)
            
            print(f"Started ambient sound: {sound_name} (volume: {self.ambient_volume:.2f})")
            return True
        
        return self._safe_audio_operation("start_ambient_sound", _start_sound)
    
    def stop_ambient_sound(self) -> None:
        """Stop the currently playing ambient sound"""
        if self.audio_available and self.current_ambient and self.is_audio_safe():
            try:
                # Stop the current ambient sound
                if self.current_ambient in self.ambient_sounds:
                    self.ambient_sounds[self.current_ambient].stop()
                print(f"Stopped ambient sound: {self.current_ambient}")
                self.current_ambient = None
            except Exception as e:
                print(f"Error stopping ambient sound: {e}")
    
    def set_ambient_volume(self, volume: float) -> None:
        """Set the volume of ambient sounds (0.0 to 1.0)"""
        if not self.audio_available or not self.is_audio_safe():
            return
            
        try:
            self.ambient_volume = max(0.0, min(1.0, volume))
            # Update volume for all ambient sounds
            for sound_object in self.ambient_sounds.values():
                sound_object.set_volume(self.ambient_volume)
            print(f"Ambient sound volume set to: {self.ambient_volume:.2f}")
        except Exception as e:
            print(f"Error setting ambient sound volume: {e}")
    
    def get_available_ambient_sounds(self) -> List[str]:
        """Get list of available ambient sound names"""
        return list(self.ambient_sounds.keys())
    
    def get_current_ambient_sound(self) -> str:
        """Get the name of the currently playing ambient sound"""
        return self.current_ambient
    
    def is_ambient_playing(self) -> bool:
        """Check if ambient sound is currently playing"""
        if not self.audio_available or not self.current_ambient or not self.is_audio_safe():
            return False
        try:
            # Check if the current ambient sound is playing
            if self.current_ambient in self.ambient_sounds:
                return self.ambient_sounds[self.current_ambient].get_num_channels() > 0
            return False
        except:
            return False
    
    def cycle_to_next_ambient_sound(self) -> bool:
        """Manually cycle to the next ambient sound in the sequence"""
        if not self.audio_available or not self.ambient_sounds or not self.is_audio_safe():
            return False
            
        try:
            sound_names = list(self.ambient_sounds.keys())
            if len(sound_names) > 1:
                # Move to next sound, wrap around to beginning
                self.ambient_sound_index = (self.ambient_sound_index + 1) % len(sound_names)
                next_sound = sound_names[self.ambient_sound_index]
                print(f"Manually cycling to next ambient sound: {next_sound}")
                return self.start_ambient_sound(next_sound)
            else:
                print("Only one ambient sound available, cannot cycle")
                return False
        except Exception as e:
            print(f"Error cycling to next ambient sound: {e}")
            return False
    
    def auto_cycle_ambient_sounds(self) -> None:
        """Automatically cycle to the next ambient sound after a certain duration"""
        if not self.audio_available or not self.ambient_sounds or not self.is_audio_safe():
            return
            
        try:
            sound_names = list(self.ambient_sounds.keys())
            if len(sound_names) > 1:
                # Move to next sound, wrap around to beginning
                self.ambient_sound_index = (self.ambient_sound_index + 1) % len(sound_names)
                next_sound = sound_names[self.ambient_sound_index]
                print(f"Auto-cycling to next ambient sound: {next_sound}")
                self.start_ambient_sound(next_sound)
        except Exception as e:
            print(f"Error auto-cycling ambient sounds: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
    
    def check_ambient_cycle_timer(self, delta_time: float = 1.0) -> None:
        """Check if it's time to cycle ambient sounds based on timer"""
        if not self.audio_available or not self.ambient_sounds:
            return
            
        # Check and fix audio state if needed
        if not self.is_audio_safe():
            return
            
        try:
            self.ambient_cycle_timer += delta_time
            if self.ambient_cycle_timer >= self.ambient_cycle_interval:
                self.ambient_cycle_timer = 0
                self.auto_cycle_ambient_sounds()
        except Exception as e:
            print(f"Error in ambient cycle timer: {e}")
            self.ambient_cycle_timer = 0
    
    def get_ambient_sound_info(self) -> dict:
        """Get information about the current ambient sound sequence"""
        if not self.audio_available or not self.ambient_sounds:
            return {}
            
        sound_names = list(self.ambient_sounds.keys())
        return {
            'total_sounds': len(sound_names),
            'current_index': getattr(self, 'ambient_sound_index', 0),
            'current_sound': self.current_ambient,
            'next_sound': sound_names[(getattr(self, 'ambient_sound_index', 0) + 1) % len(sound_names)] if len(sound_names) > 1 else None,
            'all_sounds': sound_names
        }
    
    def ensure_ambient_playing(self) -> None:
        """Ensure ambient sound is playing, restart if it stopped"""
        if not self.audio_available or not self.ambient_sounds:
            return
            
        # Check and fix audio state if needed
        if not self.is_audio_safe():
            return
            
        if not self.is_ambient_playing() and self.current_ambient:
            # Restart the current ambient sound
            print(f"Ambient sound stopped, restarting: {self.current_ambient}")
            try:
                self.start_ambient_sound(self.current_ambient)
            except Exception as e:
                print(f"Error restarting ambient sound: {e}")
                self.current_ambient = None
    
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
            if self.audio_available and self.vocal_sound and self.is_audio_safe():
                try:
                    self.vocal_sound.play()
                except Exception as e:
                    print(f"Error playing vocal sound: {e}")
                    self.disable_audio()
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
    
    def cleanup_audio(self) -> None:
        """Clean up audio resources"""
        if hasattr(self, 'audio_available') and self.audio_available:
            try:
                # Stop ambient sounds first
                self.stop_ambient_sound()
                # Don't quit the mixer completely - just stop sounds
                # pygame.mixer.quit()  # Commented out to allow reuse
            except Exception as e:
                print(f"Warning: Error cleaning up audio: {e}")
    
    def reinitialize_audio(self) -> bool:
        """Reinitialize pygame mixer after cleanup"""
        try:
            # Check if mixer is already initialized
            if hasattr(pygame.mixer, 'get_init') and pygame.mixer.get_init():
                print("Pygame mixer already initialized")
                self.audio_available = True
                return True
            
            # Try to initialize mixer
            try:
                pygame.mixer.init()
                self.audio_available = True
                print("Pygame mixer reinitialized successfully")
            except Exception as e:
                print(f"Standard reinitialization failed: {e}")
                # Try alternative settings
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                self.audio_available = True
                print("Pygame mixer reinitialized with alternative settings")
            
            # Reload ambient sounds
            if self.ambient_sounds_dir.exists():
                self._load_ambient_sounds()
            
            return True
            
        except Exception as e:
            print(f"Warning: Could not reinitialize pygame mixer: {e}")
            self.audio_available = False
            return False
    
    def _safe_audio_operation(self, operation_name: str, operation_func, *args, **kwargs):
        """Safely execute audio operations with automatic recovery"""
        try:
            if not self.is_audio_safe():
                if not self.reinitialize_audio():
                    print(f"Cannot perform {operation_name} - audio not available")
                    return False
            return operation_func(*args, **kwargs)
        except Exception as e:
            print(f"Error in {operation_name}: {e}")
            # Try to recover
            if self.reinitialize_audio():
                try:
                    return operation_func(*args, **kwargs)
                except Exception as e2:
                    print(f"Recovery failed for {operation_name}: {e2}")
                    self.disable_audio()
                    return False
            else:
                self.disable_audio()
                return False
    
    def disable_audio(self) -> None:
        """Safely disable audio functionality to prevent crashes"""
        print("Disabling audio functionality to prevent crashes")
        self.audio_available = False
        self.current_ambient = None
        self.ambient_sounds = {}
        self.vocal_sound = None
        try:
            pygame.mixer.quit()
        except Exception as e:
            print(f"Warning: Error quitting pygame mixer: {e}")
    
    def is_audio_safe(self) -> bool:
        """Check if audio is safe to use"""
        if not self.audio_available:
            return False
        try:
            # Test if pygame mixer is still working
            if hasattr(pygame.mixer, 'get_init') and pygame.mixer.get_init():
                return True
            else:
                print("Pygame mixer not properly initialized")
                # Try to reinitialize automatically
                if self.reinitialize_audio():
                    return True
                return False
        except Exception as e:
            print(f"Audio safety check failed: {e}")
            return False 