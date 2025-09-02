"""
Configuration file for the Object Detection application.
Contains all constants, default values, and configuration settings.
"""

# Application Constants
from numpy import False_


ENABLE_UNMASK = True
ENABLE_GRAD_CAM_VIEW = True
ENABLE_COUNTER_DISPLAY = True

ENABLE_AMBIENT_SOUNDS = False

# Multi-Application Configuration
# Each application can have its own configuration including screen, video folders, and model
APPLICATIONS = {
    'app_01': {
        'name': 'Application 01',
        'screen_id': 0,  # Primary monitor (0), secondary (1), etc.
        'video_folders': [
            '/home/theopsroom/Documents/NEW',
        ],
        'model_path': '/home/theopsroom/Human-OversAIght/object-detection/runs/train/weights/best.pt',
        'window_title': 'Object Detection - App 01',
        'enabled': True,
        'ndi': {
            'enabled': False,
            'source_name': 'Human-OversAIght-App01',
            'group_name': 'Detection-Apps',
            'video_format': 'BGRX',
            'frame_rate': 30,
            'video_width': 1920,
            'video_height': 1080
        }
    },
    'app_02': {
        'name': 'Application 02', 
        'screen_id': 0,  # Secondary monitor
        'video_folders': [
            '/home/theopsroom/Human-OversAIght/data/videos/vertical'
        ],
        'model_path': '/home/theopsroom/Human-OversAIght/object-detection/runs/train/weights/best.pt',
        'window_title': 'Object Detection - App 02',
        'enabled': True,
        'ndi': {
            'enabled': False,
            'source_name': 'Human-OversAIght-App02',
            'group_name': 'Detection-Apps',
            'video_format': 'BGRX',
            'frame_rate': 30,
            'video_width': 1920,
            'video_height': 1080
        }
    },
    'app_03': {
        'name': 'Application 03',
        'screen_id': 0,  # Same screen as app_01 but different position
        'video_folders': [
            '/home/theopsroom/Human-OversAIght/data/videos/square'
        ],
        'model_path': '/home/theopsroom/Human-OversAIght/object-detection/runs/train/weights/best.pt',
        'window_title': 'Object Detection - App 03',
        'enabled': True,  # Disabled by default
        'ndi': {
            'enabled': False,
            'source_name': 'Human-OversAIght-App03',
            'group_name': 'Detection-Apps',
            'video_format': 'BGRX',
            'frame_rate': 30,
            'video_width': 1920,
            'video_height': 1080
        }
    },
    'app_04': {
        'name': 'Application 04',
        'screen_id': 0,  # Same screen as app_01 but different position
        'video_folders': [
            '/home/theopsroom/Documents/VERTICAL-RED'
        ],
        'model_path': '/home/theopsroom/Human-OversAIght/object-detection/runs/train/weights/best.pt',
        'window_title': 'Object Detection - App 04',
        'enabled': True,  # Disabled by default
        'ndi': {
            'enabled': False,
            'source_name': 'Human-OversAIght-App04',
            'group_name': 'Detection-Apps',
            'video_format': 'BGRX',
            'frame_rate': 30,
            'video_width': 1920,
            'video_height': 1080
        }
    },
    'app_05': {
        'name': 'Application 05',
        'screen_id': 0,  # Same screen as app_01 but different position
        'video_folders': [
            '/home/theopsroom/Documents/NEW-R'
        ],
        'model_path': '/home/theopsroom/Human-OversAIght/object-detection/runs/train/weights/best.pt',
        'window_title': 'Object Detection - App 05',
        'enabled': True,  # Disabled by default
        'ndi': {
            'enabled': False,
            'source_name': 'Human-OversAIght-App04',
            'group_name': 'Detection-Apps',
            'video_format': 'BGRX',
            'frame_rate': 30,
            'video_width': 1920,
            'video_height': 1080
        }
    }
}

# Screen Configuration
# Maps screen IDs to display settings
SCREEN_CONFIG = {
    0: {  # Leftmost monitor
        'width': 1920,
        'height': 1080,
        'x_offset': 0,
        'y_offset': 0,
        'scale_mode': 'fit',
        'scale_multiplier': 1,
        'maintain_aspect_ratio': True,
        'center_video': True
    },
    1: {  # Middle monitor
        'width': 1920,
        'height': 1080,
        'x_offset': 1920,  # Starts where screen 0 ends
        'y_offset': 0,
        'scale_mode': 'fit',
        'scale_multiplier': 1,
        'maintain_aspect_ratio': True,
        'center_video': True
    }
}

# Default Thresholds
DEFAULT_BOX_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.2
DEFAULT_ANIMATION_ALPHA = 0.3

# Memory Management
DEFAULT_MEMORY_CHECK_INTERVAL = 60  # seconds
DEFAULT_MEMORY_CLEANUP_INTERVAL = 300  # seconds
DEFAULT_FRAME_SKIP_THRESHOLD = 50

# Display Settings
DEFAULT_CORNER_LENGTH = 20
DEFAULT_EDGE_MID_LENGTH = 20
DEFAULT_GLOW_THICKNESS = 8
DEFAULT_GLOW_ALPHA = 0.2
DEFAULT_CROSS_LENGTH = 20
DEFAULT_CROSS_THICKNESS = 2

# Blur Effect
DEFAULT_PIXEL_SIZE_DIVISOR = 8
MIN_PIXEL_SIZE = 8

# Grad-CAM Settings
DEFAULT_GRADCAM_CONF_THRESHOLD = 0.25
DEFAULT_GRADCAM_ALPHA = 1

# Video Settings
DEFAULT_FPS = 20

# NDI Settings
DEFAULT_NDI_ENABLED = False
DEFAULT_NDI_SOURCE_NAME = "Human-OversAIght-Detection"
DEFAULT_NDI_GROUP_NAME = ""
DEFAULT_NDI_VIDEO_FORMAT = "BGRX"  # BGRX, UYVY, RGBX, etc.
DEFAULT_NDI_FRAME_RATE = 30
DEFAULT_NDI_VIDEO_WIDTH = 1920
DEFAULT_NDI_VIDEO_HEIGHT = 1080

# Video Display Size Configuration
# These settings control the size of the video display window
# Set to None to use automatic screen size detection, or specify custom dimensions
DEFAULT_VIDEO_WIDTH = None  # None = auto-detect, or specify width in pixels
DEFAULT_VIDEO_HEIGHT = None  # None = auto-detect, or specify height in pixels

# Video scaling behavior
DEFAULT_VIDEO_SCALE_MODE = 'fit'  # 'fit', 'stretch', 'original'
DEFAULT_VIDEO_MAINTAIN_ASPECT_RATIO = True  # Whether to maintain video aspect ratio
DEFAULT_VIDEO_CENTER_ON_SCREEN = True  # Whether to center video on screen
DEFAULT_VIDEO_SCALE_MULTIPLIER = 1  # How much of screen to use (0.95 = 95% of available space)

# Window Settings
DEFAULT_WINDOW_WIDTH = 1920
DEFAULT_WINDOW_HEIGHT = 800
MIN_WINDOW_SIZE = 300

# Legend Display
LEGEND_WINDOW_WIDTH = 1000
LEGEND_WINDOW_HEIGHT = 500
LEGEND_FONT_SCALE = 1.2
LEGEND_LINE_HEIGHT = 40
LEGEND_BOX_PADDING = 30

# FPS Display
FPS_UPDATE_INTERVAL = 10  # frames
FPS_Y_OFFSET = 30
FPS_LINE_SPACING = 30

# File Paths
# DEFAULT_MODEL_PATH = "/home/theopsroom/App/src/cnn-detection/app/runs/train/weights/best.pt"
# DEFAULT_VIDEO_PATH = "/home/theopsroom/human-oversaight/client/apps/object-detection/videos/"
DEFAULT_MODEL_PATH = "/home/theopsroom/Human-OversAIght/object-detection/runs/train/weights/best.pt"
DEFAULT_VIDEO_PATH = "/home/theopsroom/Downloads/HUMAN OVERSAIGHT NEW VIDEOS"

# Button-to-Keyboard Mapping Configuration
# Maps the 48 buttons from Arduino Mega to keyboard actions in the detection application
# Extended to support multiple applications with app_id parameter
BUTTON_MAPPING = {
    0: {'key': 'space', 'action': 'toggle_unmask', 'app_id': 'all'},
    1: {'key': 'space', 'action': 'toggle_glitches', 'app_id': 'app_02,app_03'},
    2: {'key': 'space', 'action': 'toggle_gradcam', 'app_id': 'all'},
    3: {'key': 'space', 'action': 'toggle_unmask', 'app_id': 'all'},
    4: {'key': 'space', 'action': 'next_video', 'app_id': 'all'},
    5: {'key': 'space', 'action': 'toggle_gradcam', 'app_id': 'all'},
    6: {'key': 'space', 'action': 'toggle_unmask', 'app_id': 'all'},
    7: {'key': 'space', 'action': 'next_video', 'app_id': 'app_03'},
    8: {'key': 'space', 'action': 'toggle_gradcam', 'app_id': 'all'},
    9: {'key': 'space', 'action': 'toggle_unmask', 'app_id': 'all'},
    10: {'key': 'space', 'action': 'next_video', 'app_id': 'app_02'},
    11: {'key': 'space', 'action': 'toggle_gradcam', 'app_id': 'all'},
    12: {'key': 'space', 'action': 'toggle_unmask', 'app_id': 'all'},
    13: {'key': 'space', 'action': 'next_video', 'app_id': 'app_01'},
    14: {'key': 'space', 'action': 'toggle_gradcam', 'app_id': 'all'},
    15: {'key': 'space', 'action': 'toggle_unmask', 'app_id': 'all'},
    16: {'key': 'space', 'action': 'next_video', 'app_id': 'app_02'},
    17: {'key': 'space', 'action': 'toggle_gradcam', 'app_id': 'all'},
    18: {'key': 'space', 'action': 'toggle_unmask', 'app_id': 'all'},
    19: {'key': 'space', 'action': 'next_video', 'app_id': 'app_03'},
    20: {'key': 'space', 'action': 'toggle_gradcam', 'app_id': 'all'},
    21: {'key': 'space', 'action': 'toggle_unmask', 'app_id': 'all'},
    22: {'key': 'space', 'action': 'next_video', 'app_id': 'app_02'},
    23: {'key': 'space', 'action': 'toggle_gradcam', 'app_id': 'all'},
    24: {'key': 'space', 'action': 'next_video', 'app_id': 'app_01'},
    25: {'key': 'space', 'action': 'next_video', 'app_id': 'app_01'},
    26: {'key': 'space', 'action': 'toggle_gradcam', 'app_id': 'all'},
    27: {'key': 'space', 'action': 'next_video', 'app_id': 'app_02'},
    28: {'key': 'space', 'action': 'next_video', 'app_id': 'app_02'},
    29: {'key': 'space', 'action': 'toggle_gradcam', 'app_id': 'all'},
    30: {'key': 'space', 'action': 'next_video', 'app_id': 'app_03'},
    31: {'key': 'space', 'action': 'next_video', 'app_id': 'app_03'},
    32: {'key': 'space', 'action': 'toggle_gradcam', 'app_id': 'all'},
    33: {'key': 'space', 'action': 'next_video', 'app_id': 'app_02'},
    34: {'key': 'space', 'action': 'next_video', 'app_id': 'app_02'},
    35: {'key': 'space', 'action': 'toggle_gradcam', 'app_id': 'all'},
    36: {'key': 'space', 'action': 'next_video', 'app_id': 'app_03'},
    37: {'key': 'space', 'action': 'next_video', 'app_id': 'app_03'},
    38: {'key': 'space', 'action': 'toggle_gradcam', 'app_id': 'all'},
    39: {'key': 'space', 'action': 'next_video', 'app_id': 'app_01'},
    40: {'key': 'space', 'action': 'next_video', 'app_id': 'app_01'},
    41: {'key': 'space', 'action': 'toggle_gradcam', 'app_id': 'all'},
    42: {'key': 'space', 'action': 'next_video', 'app_id': 'app_02'},
    43: {'key': 'space', 'action': 'next_video', 'app_id': 'app_02'},
    44: {'key': 'space', 'action': 'toggle_gradcam', 'app_id': 'all'},
    45: {'key': 'space', 'action': 'next_video', 'app_id': 'app_03'},
    46: {'key': 'space', 'action': 'next_video', 'app_id': 'app_03'},
    47: {'key': 'space', 'action': 'toggle_gradcam', 'app_id': 'all'},
}

# Original button mappings (commented out for reference)
# BUTTON_MAPPING = {
#     # Navigation Controls (Buttons 0-7)
#     0: {'key': 'q', 'action': 'exit', 'description': 'Exit application'},
#     1: {'key': 'n', 'action': 'next_video', 'description': 'Next video'},
#     2: {'key': 'p', 'action': 'previous_video', 'description': 'Previous video'},
#     3: {'key': 'r', 'action': 'restart_video', 'description': 'Restart current video'},
#     4: {'key': 's', 'action': 'pause_resume', 'description': 'Pause/Resume video'},
#     5: {'key': 'f', 'action': 'fast_forward', 'description': 'Fast forward'},

#     7: {'key': 'h', 'action': 'home', 'description': 'Go to first video'},
#     
#     # Display Controls (Buttons 8-15)
#     8: {'key': 'l', 'action': 'toggle_legend', 'description': 'Toggle legend display'},
#     9: {'key': 'f', 'action': 'toggle_fps', 'description': 'Toggle FPS info'},
#     10: {'key': 'g', 'action': 'toggle_gradcam', 'description': 'Toggle Grad-CAM view'},
#     11: {'key': 'b', 'action': 'toggle_gradcam_box', 'description': 'Toggle Grad-CAM box mode'},
#     12: {'key': 'w', 'action': 'toggle_glitches', 'description': 'Toggle glitch effects'},
#     13: {'key': 'c', 'action': 'toggle_center_display', 'description': 'Toggle center display mode'},
#     14: {'key': 'v', 'action': 'toggle_visualization', 'description': 'Toggle visualization mode'},
#     15: {'key': 'd', 'action': 'toggle_debug', 'description': 'Toggle debug information'},
#     
#     # Detection Controls (Buttons 16-23)
#     16: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
#     17: {'key': '1', 'action': 'threshold_0.1', 'description': 'Set confidence threshold to 0.1'},
#     18: {'key': '2', 'action': 'threshold_0.2', 'description': 'Set confidence threshold to 0.2'},
#     19: {'key': '3', 'action': 'threshold_0.3', 'description': 'Set confidence threshold to 0.3'},
#     20: {'key': '4', 'action': 'threshold_0.4', 'description': 'Set confidence threshold to 0.4'},
#     21: {'key': '5', 'action': 'threshold_0.5', 'description': 'Set confidence threshold to 0.5'},
#     22: {'key': '6', 'action': 'threshold_0.6', 'description': 'Set confidence threshold to 0.6'},
#     23: {'key': '7', 'action': 'threshold_0.7', 'description': 'Set confidence threshold to 0.7'},
#     
#     # Advanced Controls (Buttons 24-31)
#     24: {'key': '8', 'action': 'threshold_0.8', 'description': 'Set confidence threshold to 0.8'},
#     25: {'key': '9', 'action': 'threshold_0.9', 'description': 'Set confidence threshold to 0.9'},
#     26: {'key': '0', 'action': 'threshold_1.0', 'description': 'Set confidence threshold to 1.0'},
#     27: {'key': 'i', 'action': 'increase_threshold', 'description': 'Increase confidence threshold'},
#     28: {'key': 'd', 'action': 'decrease_threshold', 'description': 'Decrease confidence threshold'},

#     
#     # Utility Controls (Buttons 32-39)
#     33: {'key': 'z', 'action': 'reset_display', 'description': 'Reset display settings'},
#     37: {'key': 'j', 'action': 'jump_frame_forward', 'description': 'Jump 10 frames forward'},
#     38: {'key': 'k', 'action': 'jump_frame_backward', 'description': 'Jump 10 frames backward'},
#     39: {'key': 'e', 'action': 'toggle_edge_detection', 'description': 'Toggle edge detection'},
#     
#     # Reserved/Experimental Controls (Buttons 40-47)
#     43: {'key': 'delete', 'action': 'clear_detections', 'description': 'Clear all detections'},
# }

# Button Configuration
BUTTON_CONFIG = {
    'serial_port': '/dev/ttyACM0',  # Default serial port for Arduino Mega
    'baud_rate': 115200,     # Default baud rate
    'polling_interval': 50,  # Milliseconds between button state updates
    'debounce_time': 100,    # Milliseconds to ignore button bounces
    'enable_sound_feedback': True,  # Enable audio feedback for button presses
    'enable_visual_feedback': True, # Enable visual feedback for button presses
    'button_hold_timeout': 1000,    # Milliseconds before button hold is detected
    'button_repeat_rate': 200,      # Milliseconds between repeated actions for held buttons
}

# LED Configuration
LED_CONFIG = {
    'enabled': True,                    # Enable LED control
    'serial_port': '/dev/ttyUSB0',     # Serial port for Arduino Nano (LED control)
    'baud_rate': 115200,               # Baud rate for LED communication
    'update_interval_ms': 20,          # LED update interval (20ms = 50 FPS)
    'default_brightness': 50,          # Default LED brightness (0-255)
    'button_press_brightness': 255,    # Brightness when button is pressed (max brightness)
    'feedback_type': 'brightness',     # Button press feedback type: 'brightness', 'pulse', 'fade'
    'fade_duration_ms': 100,          # Duration of fade effects in milliseconds
    'auto_dim': True,                  # Automatically dim LEDs after button press
    'dim_delay_ms': 100,              # Delay before dimming after button press
    'dim_to_brightness': 30,          # Brightness to dim to after button press
}

# LED-to-Button Mapping
# Maps button indices (0-47) to LED indices (0-47)
# If a button doesn't have a mapping, no LED will be controlled for that button
# Example: Button 0 controls LED 5, Button 1 controls LED 12, etc.

#42 and 12, 1 are wrongly mapped
# 47?
#missing led index: 0,9,17,20,26,31,47

LED_BUTTON_MAPPING = {
    # Button index: LED index (sorted by button index)
    0: 36, 
    1: 37,
    2: 38,  
    3: 39,
    4: 40,  
    5: 41, 
    6: 42,   
    7: 43,   
    8: 44,  
    9: 45,  
    10: 27, 
    11: 34, 
    12: 35,
    13: 34, 
    14: 33,   
    15: 32, 
    16: 28,  
    17: 35,  
    18: 42,  
    19: 8,  
    20: 15,  
    21: 4, 
    22: 29,
    23: 5,  
    24: 43,  
    25: 6,
    26: 16,  
    27: 7,  
    28: 30, 
    29: 37, 
    30: 44,  
    31: 6,   
    32: 2,  
    33: 24,  
    34: 3,  
    35: 1,   
    36: 23, 
    37: 15,
    38: 22,
    39: 14, 
    40: 21,
    41: 13,  
    42: 20,
    43: 12,  
    44: 19, 
    45: 11,  
    46: 18,  
    47: 9, 
}

# Button Action Definitions
BUTTON_ACTIONS = {
    'exit': {'type': 'immediate', 'requires_confirmation': False},
    'next_video': {'type': 'immediate', 'requires_confirmation': False},
    'next_folder': {'type': 'immediate', 'requires_confirmation': False},
    'previous_video': {'type': 'immediate', 'requires_confirmation': False},
    'restart_video': {'type': 'immediate', 'requires_confirmation': False},
    'pause_resume': {'type': 'toggle', 'requires_confirmation': False},
    'fast_forward': {'type': 'hold', 'requires_confirmation': False},
    'home': {'type': 'immediate', 'requires_confirmation': False},
    'toggle_legend': {'type': 'toggle', 'requires_confirmation': False},
    'toggle_fps': {'type': 'toggle', 'requires_confirmation': False},
    'toggle_gradcam': {'type': 'toggle', 'requires_confirmation': False},
    'toggle_gradcam_box': {'type': 'toggle', 'requires_confirmation': False},
    'toggle_glitches': {'type': 'toggle', 'requires_confirmation': False},
    'toggle_center_display': {'type': 'toggle', 'requires_confirmation': False},
    'toggle_visualization': {'type': 'toggle', 'requires_confirmation': False},
    'toggle_debug': {'type': 'toggle', 'requires_confirmation': False},
    'toggle_unmask': {'type': 'toggle', 'requires_confirmation': False},
    'threshold_0.1': {'type': 'immediate', 'requires_confirmation': False},
    'threshold_0.2': {'type': 'immediate', 'requires_confirmation': False},
    'threshold_0.3': {'type': 'immediate', 'requires_confirmation': False},
    'threshold_0.4': {'type': 'immediate', 'requires_confirmation': False},
    'threshold_0.5': {'type': 'immediate', 'requires_confirmation': False},
    'threshold_0.6': {'type': 'immediate', 'requires_confirmation': False},
    'threshold_0.7': {'type': 'immediate', 'requires_confirmation': False},
    'threshold_0.8': {'type': 'immediate', 'requires_confirmation': False},
    'threshold_0.9': {'type': 'immediate', 'requires_confirmation': False},
    'threshold_1.0': {'type': 'immediate', 'requires_confirmation': False},
    'increase_threshold': {'type': 'hold', 'requires_confirmation': False},
    'decrease_threshold': {'type': 'hold', 'requires_confirmation': False},
    'reset_display': {'type': 'immediate', 'requires_confirmation': True},
    'jump_frame_forward': {'type': 'immediate', 'requires_confirmation': False},
    'jump_frame_backward': {'type': 'immediate', 'requires_confirmation': False},
    'toggle_edge_detection': {'type': 'toggle', 'requires_confirmation': False},
    'clear_detections': {'type': 'immediate', 'requires_confirmation': True},
    'reset_counter': {'type': 'immediate', 'requires_confirmation': True}, 
}

# Counter Display Configuration
COUNTER_CONFIG = {
    'enabled': True,
    'position': 'top_left',  # 'top_left', 'top_right', 'bottom_left', 'bottom_right'
    'font_scale': 1.0,
    'font_thickness': 2,
    'text_color': (255, 255, 255),  # White
    'background_color': (0, 0, 0),  # Black
    'background_alpha': 0.7,
    'x_offset': 20,
    'y_offset': 50,
    'line_height': 30
}

# Visual Feedback Configuration
# Controls the appearance of feedback circles when buttons or keys are pressed
FEEDBACK_CONFIG = {
    'enabled': True,                    # Enable visual feedback
    'default_color': (0, 0, 255),      # Default color (BGR format) - Red
    'default_radius': 20,               # Default circle radius in pixels
    'fade_in_duration': 0.1,            # Fade in duration in seconds
    'fade_out_duration': 0.3,           # Fade out duration in seconds
    'total_duration': 0.8,              # Total feedback duration in seconds
    'offset_x': 40,                      # X offset from position (pixels)
    'offset_y': 40,                      # Y offset from position (pixels)
    'min_alpha': 0.0,                   # Minimum alpha (transparency)
    'max_alpha': 0.8,                   # Maximum alpha (opacity)
    'blend_mode': 'alpha',              # 'alpha' for transparency, 'add' for additive blending
}

# Action-specific feedback colors
# Maps action names to specific colors for visual feedback
ACTION_FEEDBACK_COLORS = {
    'exit': (0, 0, 255),               # Red
    'next_video': (0, 0, 0),         # Black
    'next_folder': (255, 255, 255),  # White
    'previous_video': (255, 255, 255),  # White
    'restart_video': (0, 0, 255),       # Red
    'pause_resume': (255, 255, 0),     # Cyan
    'fast_forward': (166, 166, 166),     # Grey
    'home': (255, 128, 0),             # Blue-Orange
    'toggle_legend': (255, 255, 255),  # White
    'toggle_fps': (255, 128, 128),     # Light Red
    'toggle_gradcam': (128, 128, 255), # Light Blue
    'toggle_gradcam_box': (200, 200, 200), # Light Yellow
    'toggle_glitches': (250, 227, 212), # Very Light Blue
    'toggle_center_display': (128, 255, 255), # Light Cyan
    'toggle_visualization': (255, 255, 255), # White
    'toggle_debug': (64, 64, 64),      # Dark Gray
    'toggle_unmask': (128, 128, 128),    # grey
    'threshold_0.1': (0, 25, 255),     # Dark Red
    'threshold_0.2': (0, 51, 255),     # Darker Red
    'threshold_0.3': (0, 76, 255),     # Medium Red
    'threshold_0.4': (0, 102, 255),    # Light Red
    'threshold_0.5': (0, 128, 255),    # Orange
    'threshold_0.6': (0, 153, 255),    # Light Orange
    'threshold_0.7': (250, 227, 212),    # Very Light Blue
    'threshold_0.8': (255, 255, 255),    # Light Yellow
    'threshold_0.9': (211, 211, 211),    # Very Light Grey
    'threshold_1.0': (255, 255, 255),    # Yellow
    'increase_threshold': (250, 227, 212), # Very Light Blue
    'decrease_threshold': (255, 0, 0), # Blue
    'reset_display': (255, 0, 0),      # Blue
    'jump_frame_forward': (255, 0, 0),      # Blue
    'jump_frame_backward': (0, 0, 255), # Red
    'toggle_edge_detection': (0, 0, 255), # Red
    'clear_detections': (0, 0, 0),   # Blue
    'reset_counter': (255, 255, 255),    # Dark Orange
}

# Key-specific feedback colors (for keyboard input)
KEY_FEEDBACK_COLORS = {
    ord('q'): (0, 0, 255),            # Red for quit
    ord('n'): (0, 255, 0),            # Green for next
    ord('f'): (0, 255, 255),          # Yellow (Cyan) for next folder
    ord('t'): (255, 128, 128),        # Light Red for FPS toggle
    ord('b'): (0, 255, 255),          # Yellow for toggle
    ord('l'): (128, 255, 128),        # Light Green for legend
    ord('d'): (128, 128, 255),        # Light Blue for debug/NDI
    27: (255, 0, 0),                  # Blue for ESC
    32: (0, 255, 128),                # Green-Blue for space
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO', #DEBUG
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S'
} 