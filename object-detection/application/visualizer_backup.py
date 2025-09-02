"""
Visualization utilities for the object detection application.
"""

import cv2
import numpy as np
import time
import sounddevice as sd
import soundfile as sf
import threading
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
from application.feedback_overlay import FeedbackOverlay


def get_audio_devices() -> Tuple[str, ...]:
    """Get available audio output devices using sounddevice"""
    try:
        devices = sd.query_devices()
        output_devices = []
        
        # Use default device with 6 channels
        try:
            default_device = sd.default.device
            if hasattr(default_device, '__getitem__'):  # Handle _InputOutputPair
                default_output = default_device[1]
            else:
                default_output = default_device
            
            if default_output is not None:
                # Try to get device info for default device
                if default_output < len(devices):
                    device_info = devices[default_output]
                    output_devices.append(f"{default_output}: {device_info['name']} (default - 6 channels)")
                else:
                    output_devices.append(f"{default_output}: Default Audio Device (6 channels)")
        except Exception as e:
            print(f"Could not get default audio device: {e}")
        
        return tuple(output_devices)
    except Exception as e:
        print(f"Error getting audio devices: {e}")
        return tuple()

# Initialize audio devices lazily to prevent import-time issues
audio_devices = None
selected_device = None

def _initialize_audio_devices():
    """Initialize audio devices on first use"""
    global audio_devices, selected_device
    if audio_devices is None:
        audio_devices = get_audio_devices()
        
        if not audio_devices:
            print("Warning: No audio output devices found - audio will be disabled")
            selected_device = None
        else:
            # Extract device index from the first available device string
            sound_device = audio_devices[0]
            try:
                selected_device = int(sound_device.split(':')[0])
            except (ValueError, IndexError):
                selected_device = 0  # Fallback to device 0
            print(f"Available audio devices: {audio_devices}")
            print(f"Selected audio device: {selected_device} ({sound_device})")
    
    return audio_devices, selected_device


class AudioPlayer:
    """Sounddevice-based audio player for button and ambient sounds with multi-channel support"""
    
    def __init__(self, device_index: int = None):
        # Initialize audio devices if not already done
        _, default_device = _initialize_audio_devices()
        self.device_index = device_index or default_device
        self.audio_available = False
        self.ambient_playing = False
        self.ambient_stop_event = threading.Event()
        self.ambient_thread = None
        self.ambient_stream = None
        self._lock = threading.Lock()  # Add thread safety
        
        # Audio settings
        self.sample_rate = 44100
        self.channels = 6  # Use 6 channels for surround sound
        
        self._initialize_audio()
    
    def _initialize_audio(self) -> None:
        """Initialize sounddevice for audio playback"""
        if self.device_index is None:
            print("✗ No audio device available - audio disabled")
            self.audio_available = False
            return
            
        try:
            # Test if we can open a stream
            test_stream = sd.OutputStream(
                device=self.device_index,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32
            )
            test_stream.close()
            
            self.audio_available = True
            print("✓ Sounddevice initialization successful")
            
        except Exception as e:
            print(f"✗ Sounddevice initialization failed: {e}")
            # Try with different settings for 6-channel audio
            try:
                print("Trying alternative audio settings for 6-channel device...")
                test_stream = sd.OutputStream(
                    device=self.device_index,
                    samplerate=48000,  # Try different sample rate
                    channels=6,  # Try 6 channels
                    dtype=np.float32
                )
                test_stream.close()
                self.audio_available = True
                self.sample_rate = 48000  # Update sample rate
                self.channels = 6  # Update channels
                print("✓ Sounddevice initialization successful with 6-channel settings")
            except Exception as e2:
                # Try with 2 channels as fallback
                try:
                    print("Trying 2-channel fallback...")
                    test_stream = sd.OutputStream(
                        device=self.device_index,
                        samplerate=44100,
                        channels=2,
                        dtype=np.float32
                    )
                    test_stream.close()
                    self.audio_available = True
                    self.channels = 2  # Fallback to 2 channels
                    print("✓ Sounddevice initialization successful with 2-channel fallback")
                except Exception as e3:
                    print(f"✗ All audio initialization attempts failed: {e3}")
                    self.audio_available = False
    
    def play_sound_file(self, file_path: str, volume: float = 1.0) -> bool:
        """Play a single sound file with multi-channel support"""
        if not self.audio_available:
            return False
        
        try:
            with self._lock:  # Thread safety
                # Load audio file
                audio_data, sample_rate = sf.read(file_path)
                
                # Convert to appropriate channel count
                if len(audio_data.shape) == 1:
                    # Mono to multi-channel
                    if self.channels == 6:
                        # Convert mono to 6-channel (front L/R, center, LFE, rear L/R)
                        audio_data = np.column_stack((audio_data, audio_data, audio_data, audio_data, audio_data, audio_data))
                    else:
                        audio_data = np.column_stack((audio_data, audio_data))
                elif audio_data.shape[1] == 1:
                    # Mono to multi-channel
                    if self.channels == 6:
                        audio_data = np.column_stack((audio_data[:, 0], audio_data[:, 0], audio_data[:, 0], audio_data[:, 0], audio_data[:, 0], audio_data[:, 0]))
                    else:
                        audio_data = np.column_stack((audio_data[:, 0], audio_data[:, 0]))
                elif audio_data.shape[1] == 2 and self.channels == 6:
                    # Stereo to 6-channel (duplicate channels)
                    audio_data = np.column_stack((audio_data[:, 0], audio_data[:, 1], audio_data[:, 0], audio_data[:, 0], audio_data[:, 0], audio_data[:, 1]))
                
                # Apply volume
                if volume != 1.0:
                    audio_data = audio_data * volume
                
                # Play audio
                sd.play(audio_data, sample_rate, device=self.device_index)
                sd.wait()  # Wait for playback to finish
                
                return True
                
        except Exception as e:
            print(f"Error playing sound file {file_path}: {e}")
            return False
    
    def start_ambient_sound(self, file_path: str, volume: float = 1.0) -> bool:
        """Start playing ambient sound in a loop with multi-channel support"""
        if not self.audio_available:
            return False
        
        with self._lock:  # Thread safety
            # Stop any currently playing ambient sound
            self.stop_ambient_sound()
            
            try:
                self.ambient_stop_event.clear()
                self.ambient_thread = threading.Thread(
                    target=self._ambient_sound_loop,
                    args=(file_path, volume),
                    daemon=True
                )
                self.ambient_thread.start()
                self.ambient_playing = True
                return True
                
            except Exception as e:
                print(f"Error starting ambient sound: {e}")
                return False
    
    def _ambient_sound_loop(self, file_path: str, volume: float):
        """Loop ambient sound in background thread"""
        try:
            # Load audio file once
            audio_data, sample_rate = sf.read(file_path)
            
            # Convert to appropriate channel count
            if len(audio_data.shape) == 1:
                # Mono to multi-channel
                if self.channels == 6:
                    # Convert mono to 6-channel (front L/R, center, LFE, rear L/R)
                    audio_data = np.column_stack((audio_data, audio_data, audio_data, audio_data, audio_data, audio_data))
                else:
                    audio_data = np.column_stack((audio_data, audio_data))
            elif audio_data.shape[1] == 1:
                # Mono to multi-channel
                if self.channels == 6:
                    audio_data = np.column_stack((audio_data[:, 0], audio_data[:, 0], audio_data[:, 0], audio_data[:, 0], audio_data[:, 0], audio_data[:, 0]))
                else:
                    audio_data = np.column_stack((audio_data[:, 0], audio_data[:, 0]))
            elif audio_data.shape[1] == 2 and self.channels == 6:
                # Stereo to 6-channel (duplicate channels)
                audio_data = np.column_stack((audio_data[:, 0], audio_data[:, 1], audio_data[:, 0], audio_data[:, 0], audio_data[:, 0], audio_data[:, 1]))
            
            # Apply volume
            if volume != 1.0:
                audio_data = audio_data * volume
            
            while not self.ambient_stop_event.is_set():
                try:
                    with self._lock:  # Thread safety
                        if self.ambient_stop_event.is_set():
                            break
                        # Play audio
                        sd.play(audio_data, sample_rate, device=self.device_index)
                        sd.wait()
                        
                except Exception as e:
                    print(f"Error in ambient sound loop: {e}")
                    break
                    
        except Exception as e:
            print(f"Error loading ambient sound: {e}")
        finally:
            with self._lock:
                self.ambient_playing = False
    
    def stop_ambient_sound(self) -> None:
        """Stop the currently playing ambient sound"""
        with self._lock:  # Thread safety
            if self.ambient_playing:
                self.ambient_stop_event.set()
                try:
                    sd.stop()  # Stop all audio playback
                except:
                    pass  # Ignore errors during stop
                
                if self.ambient_thread and self.ambient_thread.is_alive():
                    self.ambient_thread.join(timeout=2.0)  # Increased timeout
                self.ambient_playing = False
    
    def is_ambient_playing(self) -> bool:
        """Check if ambient sound is currently playing"""
        with self._lock:
            return self.ambient_playing and not self.ambient_stop_event.is_set()
    
    def get_device_info(self) -> dict:
        """Get detailed information about the current audio device"""
        if not self.audio_available:
            return {}
        
        try:
            devices = sd.query_devices()
            if self.device_index < len(devices):
                device_info = devices[self.device_index]
                return {
                    'name': device_info['name'],
                    'max_outputs': device_info.get('max_outputs', 0),
                    'default_samplerate': device_info.get('default_samplerate', 44100),
                    'hostapi': device_info.get('hostapi', 'unknown')
                }
        except Exception as e:
            print(f"Error getting device info: {e}")
        return {}
    
    def cleanup(self) -> None:
        """Clean up audio resources"""
        with self._lock:  # Thread safety
            self.stop_ambient_sound()
            self.audio_available = False


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
        
        # Initialize feedback overlay system
        self.feedback_overlay = FeedbackOverlay()
        
        self.audio_player = AudioPlayer()
        self.audio_available = self.audio_player.audio_available
        
        # Audio file paths
        project_root = Path(__file__).parent.parent.parent
        self.button_sound_path = project_root / "assets" / "sounds" / "buttons"
        self.ambient_sounds_dir = project_root / "assets" / "sounds" / "ambient"
        
        # Load button sounds
        self.button_sounds = {}
        if self.audio_available and self.button_sound_path.exists():
            self._load_button_sounds()
        
        # Load ambient sounds
        self.ambient_sounds = {}
        self.current_ambient = None
        self.ambient_volume = 1.0
        self.volume_amplification = 1.0
        self.ambient_cycle_timer = 0
        self.ambient_cycle_interval = 30
        self.ambient_sound_index = 0
        
        if self.audio_available:
            self._load_ambient_sounds()
    
    def _load_button_sounds(self) -> None:
        """Load all available button sound files"""
        try:
            wav_files = list(self.button_sound_path.glob("*.wav"))
            for sound_file in wav_files:
                sound_name = sound_file.stem
                self.button_sounds[sound_name] = str(sound_file)
                print(f"Loaded button sound: {sound_name}")
        except Exception as e:
            print(f"Warning: Could not load button sounds: {e}")
    
    def _load_ambient_sounds(self) -> None:
        """Load all available ambient sound files"""
        if not self.ambient_sounds_dir.exists():
            print(f"Ambient sounds directory does not exist: {self.ambient_sounds_dir}")
            return
        
        try:
            wav_files = list(self.ambient_sounds_dir.glob("*.wav"))
            for sound_file in wav_files:
                sound_name = sound_file.stem
                self.ambient_sounds[sound_name] = str(sound_file)
                print(f"Loaded ambient sound: {sound_name}")
            
            # Start the first ambient sound
            if self.ambient_sounds:
                first_sound = list(self.ambient_sounds.keys())[0]
                self.start_ambient_sound(first_sound)
                
        except Exception as e:
            print(f"Warning: Could not load ambient sounds: {e}")
    
    def start_ambient_sound(self, sound_name: str = None, volume: float = None) -> bool:
        """Start playing ambient sound in background loop"""
        if not self.audio_available or not self.ambient_sounds:
            return False
        
        target_sound = sound_name
        if target_sound is None:
            target_sound = list(self.ambient_sounds.keys())[0]
        
        if target_sound not in self.ambient_sounds:
            print(f"Ambient sound '{target_sound}' not found")
            return False
        
        file_path = self.ambient_sounds[target_sound]
        
        # Set volume if specified
        if volume is not None:
            self.ambient_volume = max(0.0, min(1.0, volume))
        
        # Apply volume amplification
        final_volume = self.ambient_volume * self.volume_amplification
        
        success = self.audio_player.start_ambient_sound(file_path, final_volume)
        if success:
            self.current_ambient = target_sound
            sound_names = list(self.ambient_sounds.keys())
            self.ambient_sound_index = sound_names.index(target_sound)
            print(f"Started ambient sound: {target_sound}")
        
        return success
    
    def stop_ambient_sound(self) -> None:
        """Stop the currently playing ambient sound"""
        self.audio_player.stop_ambient_sound()
        self.current_ambient = None
    
    def set_ambient_volume(self, volume: float) -> None:
        """Set the volume of ambient sounds (0.0 to 1.0)"""
        if not self.audio_available:
            return
        
        self.ambient_volume = max(0.0, min(1.0, volume))
        # Restart current ambient sound with new volume
        if self.current_ambient:
            self.start_ambient_sound(self.current_ambient)
    
    def set_volume_amplification(self, amplification: float) -> None:
        """Set the volume amplification multiplier"""
        if not self.audio_available:
            return
        
        self.volume_amplification = max(1.0, min(5.0, amplification))
        # Restart current ambient sound with new amplification
        if self.current_ambient:
            self.start_ambient_sound(self.current_ambient)
    
    def get_volume_amplification(self) -> float:
        """Get the current volume amplification multiplier"""
        return self.volume_amplification
    
    def boost_volume(self) -> None:
        """Increase volume amplification by 0.5x"""
        current = self.volume_amplification
        self.set_volume_amplification(current + 0.5)
    
    def reduce_volume(self) -> None:
        """Decrease volume amplification by 0.5x"""
        current = self.volume_amplification
        self.set_volume_amplification(current - 0.5)
    
    def get_available_ambient_sounds(self) -> List[str]:
        """Get list of available ambient sound names"""
        return list(self.ambient_sounds.keys())
    
    def get_current_ambient_sound(self) -> str:
        """Get the name of the currently playing ambient sound"""
        return self.current_ambient
    
    def is_ambient_playing(self) -> bool:
        """Check if ambient sound is currently playing"""
        return self.audio_player.is_ambient_playing()
    
    def cycle_to_next_ambient_sound(self) -> bool:
        """Manually cycle to the next ambient sound"""
        if not self.audio_available or not self.ambient_sounds:
            return False
        
        sound_names = list(self.ambient_sounds.keys())
        if len(sound_names) > 1:
            self.ambient_sound_index = (self.ambient_sound_index + 1) % len(sound_names)
            next_sound = sound_names[self.ambient_sound_index]
            return self.start_ambient_sound(next_sound)
        return False
    
    def auto_cycle_ambient_sounds(self) -> None:
        """Automatically cycle to the next ambient sound"""
        if not self.audio_available or not self.ambient_sounds:
            return
        
        sound_names = list(self.ambient_sounds.keys())
        if len(sound_names) > 1:
            self.ambient_sound_index = (self.ambient_sound_index + 1) % len(sound_names)
            next_sound = sound_names[self.ambient_sound_index]
            self.start_ambient_sound(next_sound)
    
    def check_ambient_cycle_timer(self, delta_time: float = 1.0) -> None:
        """Check if it's time to cycle ambient sounds"""
        if not self.audio_available or not self.ambient_sounds:
            return
        
        self.ambient_cycle_timer += delta_time
        if self.ambient_cycle_timer >= self.ambient_cycle_interval:
            self.ambient_cycle_timer = 0
            self.auto_cycle_ambient_sounds()
    
    def get_ambient_sound_info(self) -> dict:
        """Get information about the current ambient sound sequence"""
        if not self.audio_available or not self.ambient_sounds:
            return {}
        
        sound_names = list(self.ambient_sounds.keys())
        return {
            'total_sounds': len(sound_names),
            'current_index': self.ambient_sound_index,
            'current_sound': self.current_ambient,
            'next_sound': sound_names[(self.ambient_sound_index + 1) % len(sound_names)] if len(sound_names) > 1 else None,
            'all_sounds': sound_names
        }
    
    def play_button_sound(self, sound_name: str = None, volume: float = 1.0) -> bool:
        """Play a button sound effect"""
        if not self.audio_available or not self.button_sounds:
            return False
        
        if sound_name is None:
            import random
            available_sounds = list(self.button_sounds.keys())
            if available_sounds:
                sound_name = random.choice(available_sounds)
            else:
                return False
        
        if sound_name not in self.button_sounds:
            return False
        
        file_path = self.button_sounds[sound_name]
        return self.audio_player.play_sound_file(file_path, volume)
    
    def get_available_button_sounds(self) -> List[str]:
        """Get list of available button sound names"""
        return list(self.button_sounds.keys())
    
    def set_button_sound_volume(self, volume: float) -> None:
        """Set the volume of button sounds (0.0 to 1.0)"""
        # Volume is set per-play for button sounds
        pass
    
    def play_random_button_sound(self, volume: float = 1.0) -> bool:
        """Play a random button sound effect"""
        return self.play_button_sound(None, volume)
    
    def ensure_ambient_playing(self) -> None:
        """Ensure ambient sound is playing, restart if it stopped"""
        if not self.audio_available or not self.ambient_sounds:
            return
        
        if not self.is_ambient_playing() and self.current_ambient:
            self.start_ambient_sound(self.current_ambient)
    
    def draw_detection_overlays(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw detection overlays on the frame"""
        # Update feedback overlay with current frame dimensions
        if hasattr(self, 'feedback_overlay') and self.feedback_overlay:
            self.feedback_overlay.set_frame_dimensions(frame.shape[1], frame.shape[0])
        
        for i, detection in enumerate(detections):
            color = ColorManager.get_state_color(self.display_config.color_state, detection.confidence)
            #print(f"Detection {i}: box={detection.box}, color={color}, color_state={self.display_config.color_state}")
            
            if self.display_config.solid_border:
                x1, y1, x2, y2 = detection.box
                corner_length = DEFAULT_CORNER_LENGTH
                thickness = 3
            else:
                x1, y1, x2, y2 = detection.box
                corner_length = DEFAULT_CORNER_LENGTH
                thickness = 2
                
                if self.display_config.blur_boxes:
                    self._apply_blur_effect(frame, detection.box)
            
            self._draw_box_corners(frame, (x1, y1, x2, y2), color, corner_length, thickness)
            self._draw_box_edges(frame, (x1, y1, x2, y2), color, thickness)
            self._apply_glow_effect(frame, detection.box, color)
            
            if self.display_config.show_enemy:
                self._draw_enemy_indicators(frame, detection.box, color)
        
        # Draw feedback overlays on top of detection overlays
        result_frame = self.feedback_overlay.draw_feedbacks(frame, debug_save=False)
        return result_frame
    
    def trigger_button_feedback(self, action_name: str, position: Tuple[int, int] = None):
        """Trigger visual feedback for a button press"""
        self.feedback_overlay.add_feedback(action_name=action_name, position=position)
    
    def trigger_key_feedback(self, key_code: int, position: Tuple[int, int] = None):
        """Trigger visual feedback for a key press"""
        self.feedback_overlay.add_feedback(key_code=key_code, position=position)
    
    def trigger_custom_feedback(self, color: Tuple[int, int, int], position: Tuple[int, int] = None, 
                               radius: int = None, duration: float = None):
        """Trigger custom visual feedback with specified parameters"""
        self.feedback_overlay.add_feedback(color=color, position=position, radius=radius, duration=duration)
    
    def clear_feedbacks(self):
        """Clear all active feedback overlays"""
        self.feedback_overlay.clear_all_feedbacks()
    
    def clear_feedbacks_immediately(self):
        """Clear all feedback overlays immediately without waiting for duration to expire"""
        self.feedback_overlay.clear_feedbacks_immediately()
    
    def disable_feedback(self):
        """Temporarily disable feedback (circles won't appear)"""
        self.feedback_overlay.disable_feedback()
    
    def enable_feedback(self):
        """Re-enable feedback after disabling"""
        self.feedback_overlay.enable_feedback()
    
    def is_feedback_enabled(self) -> bool:
        """Check if feedback is currently enabled"""
        return self.feedback_overlay.is_feedback_enabled()
    
    def get_feedback_info(self) -> List[Dict]:
        """Get detailed information about all active feedbacks for debugging"""
        return self.feedback_overlay.get_feedback_info()
    
    def get_feedback_count(self) -> int:
        """Get the number of active feedback overlays"""
        return self.feedback_overlay.get_feedback_count()
    
    def set_feedback_config(self, config: dict):
        """Update feedback configuration"""
        self.feedback_overlay.set_config(config)
    
    def set_feedback_action_colors(self, colors: dict):
        """Update action-specific feedback colors"""
        self.feedback_overlay.set_action_colors(colors)
    
    def set_feedback_key_colors(self, colors: dict):
        """Update key-specific feedback colors"""
        self.feedback_overlay.set_key_colors(colors)
    
    def update_frame_dimensions(self, width: int, height: int):
        """Update frame dimensions for feedback positioning"""
        if hasattr(self, 'feedback_overlay') and self.feedback_overlay:
            self.feedback_overlay.set_frame_dimensions(width, height)
    
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
        """Draw enemy indicators (coordinates text and cross)"""
        font = cv2.FONT_HERSHEY_DUPLEX
        
        # Extract coordinates from the box
        x1, y1, x2, y2 = box
        
        # Format coordinates text - show top-left and bottom-right coordinates
        coords_text = f"{x1},{y1} {x2},{y2}"
        
        # Draw coordinates text above the bounding box
        cv2.putText(frame, coords_text, (box[0], box[1] - 10), font, 0.6, color, 2)

        # cv2.putText(frame, "TARGET", (box[0], box[1] - 10), font, 0.8, color, 2)
        
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
        
        # Use frame index to animate the glitch from left to right
        glitch_width = min(w, (self.frame_idx % 60) * (w // 60))  # Animate over 60 frames
        
        if glitch_width > 0:
            # Fixed vertical position for consistent glitch line
            y = h // 3  # Fixed position at 1/3 of screen height
            
            points = []
            for x in range(0, glitch_width, 1):
                offset = np.random.randint(-3, 4)
                points.append((x, min(max(y + offset, 0), h - 1)))
            
            # Draw lines from left to right, connecting consecutive points
            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i + 1], (0, 0, 0), 1)
    
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
            
            # Play button sound when toggling display mode
            if self.audio_available:
                self.play_button_sound()
        elif key == ord('g') and ENABLE_GRAD_CAM_VIEW:  # 'g' key to toggle Grad-CAM
            if app_instance:
                app_instance.display_config.gradcam_enabled = not app_instance.display_config.gradcam_enabled
                print(f"Grad-CAM {'enabled' if app_instance.display_config.gradcam_enabled else 'disabled'}")
                # Play button sound when toggling Grad-CAM
                if self.audio_available:
                    self.play_button_sound()
        elif key == ord('w'):
            app_instance.display_config.enable_glitches = not app_instance.display_config.enable_glitches
            print(f"Glitches {'enabled' if app_instance.display_config.enable_glitches else 'disabled'}")
            # Play button sound when toggling glitches
            if self.audio_available:
                self.play_button_sound()
    
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
        try:
            if hasattr(self, 'audio_player') and self.audio_player:
                self.audio_player.cleanup()
                self.audio_player = None
        except Exception as e:
            print(f"Warning: Error during audio cleanup: {e}")
    
    def reinitialize_audio(self) -> bool:
        """Reinitialize sounddevice after cleanup"""
        try:
            # Clean up existing audio player
            if hasattr(self, 'audio_player') and self.audio_player:
                self.audio_player.cleanup()
                self.audio_player = None
            
            # Create new audio player
            self.audio_player = AudioPlayer()
            self.audio_available = self.audio_player.audio_available
            
            # Reload sounds if audio is now available
            if self.audio_available:
                if self.button_sound_path.exists():
                    self._load_button_sounds()
                if self.ambient_sounds_dir.exists():
                    self._load_ambient_sounds()
            
            return self.audio_available
            
        except Exception as e:
            print(f"Warning: Could not reinitialize sounddevice: {e}")
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
        try:
            if hasattr(self, 'audio_player') and self.audio_player:
                self.audio_player.cleanup()
                self.audio_player = None
        except Exception as e:
            print(f"Warning: Error during audio disable: {e}")
        
        self.audio_available = False
        self.current_ambient = None
        self.ambient_sounds = {}
        self.button_sounds = {}
    
    def is_audio_safe(self) -> bool:
        """Check if audio is safe to use"""
        if not self.audio_available:
            return False
        try:
            return (hasattr(self, 'audio_player') and 
                   self.audio_player and 
                   self.audio_player.audio_available)
        except Exception as e:
            print(f"Audio safety check failed: {e}")
            return False 