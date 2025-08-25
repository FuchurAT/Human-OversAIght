"""
Configuration file for the Object Detection application.
Contains all constants, default values, and configuration settings.
"""

# Application Constants
ENABLE_UNMASK = True
ENABLE_GRAD_CAM_VIEW = True

# Multi-Application Configuration
# Each application can have its own configuration including screen, video folder, and model
APPLICATIONS = {
    'app_01': {
        'name': 'Application 01',
        'screen_id': 1,  # Primary monitor (0), secondary (1), etc.
        'video_folder': '/home/theopsroom/Human-OversAIght/data/videos/vertical',
        'model_path': '/home/theopsroom/Human-OversAIght/object-detection/runs/train/weights/best.pt',
        'window_title': 'Object Detection - App 01',
        'enabled': True
    },
    'app_02': {
        'name': 'Application 02', 
        'screen_id': 2,  # Secondary monitor
        'video_folder': '/home/theopsroom/Human-OversAIght/data/videos/horizontal',
        'model_path': '/home/theopsroom/Human-OversAIght/object-detection/runs/train/weights/best.pt',
        'window_title': 'Object Detection - App 02',
        'enabled': True
    },
    'app_03': {
        'name': 'Application 03',
        'screen_id': 3,
        'video_folder': '/home/theopsroom/Human-OversAIght/data/videos/square',
        'model_path': '/home/theopsroom/Human-OversAIght/object-detection/runs/train/weights/best.pt',
        'window_title': 'Object Detection - App 03',
        'enabled': True  
    }
}

# Screen Configuration
# Maps screen IDs to display settings
SCREEN_CONFIG = {
    1: {  # Primary monitor
        'width': 1080,
        'height': 1920,
        'x_offset': 1080,
        'y_offset': 0,
        'scale_mode': 'fit',
        'scale_multiplier': 1,
        'maintain_aspect_ratio': True,
        'center_video': True
    },
    2: {  # Secondary monitor
        'width': 1920,
        'height': 1080,
        'x_offset': 1920,  # Position to the right of primary
        'y_offset': 0,
        'scale_mode': 'fit',
        'scale_multiplier': 1,
        'maintain_aspect_ratio': True,
        'center_video': True
    },
    3: {  # Secondary monitor
        'width': 1080,
        'height': 1080,
        'x_offset': 3840,  # Position to the right of primary
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
    # All 48 buttons now act as space (toggle unmask/blur mode) for all applications
    0: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    1: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    2: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    3: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    4: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    5: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    6: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    7: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    8: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    9: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    10: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    11: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    12: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    13: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    14: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    15: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    16: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    17: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    18: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    19: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    20: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    21: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    22: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    23: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    24: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    25: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    26: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    27: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    28: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    29: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    30: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    31: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    32: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    33: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    34: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    35: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    36: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    37: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    38: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    39: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    40: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    41: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    42: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    43: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    44: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    45: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    46: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
    47: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode', 'app_id': 'all'},
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
#     6: {'key': 'b', 'action': 'rewind', 'description': 'Rewind'},
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
#     29: {'key': 'm', 'action': 'toggle_memory_cleanup', 'description': 'Toggle memory cleanup'},
#     30: {'key': 'a', 'action': 'toggle_audio', 'description': 'Toggle audio effects'},
#     31: {'key': 't', 'action': 'toggle_training_mode', 'description': 'Toggle training mode'},
#     
#     # Utility Controls (Buttons 32-39)
#     32: {'key': 'o', 'action': 'toggle_output', 'description': 'Toggle output recording'},
#     33: {'key': 'z', 'action': 'reset_display', 'description': 'Reset display settings'},
#     34: {'key': 'x', 'action': 'export_results', 'description': 'Export detection results'},
#     35: {'key': 'y', 'action': 'save_screenshot', 'description': 'Save current frame'},
#     36: {'key': 'u', 'action': 'toggle_ui', 'description': 'Toggle UI elements'},
#     37: {'key': 'j', 'action': 'jump_frame_forward', 'description': 'Jump 10 frames forward'},
#     38: {'key': 'k', 'action': 'jump_frame_backward', 'description': 'Jump 10 frames backward'},
#     39: {'key': 'e', 'action': 'toggle_edge_detection', 'description': 'Toggle edge detection'},
#     
#     # Reserved/Experimental Controls (Buttons 40-47)
#     40: {'key': 'tab', 'action': 'cycle_view_mode', 'description': 'Cycle through view modes'},
#     41: {'key': 'enter', 'action': 'confirm_action', 'description': 'Confirm current action'},
#     42: {'key': 'backspace', 'action': 'undo_action', 'description': 'Undo last action'},
#     43: {'key': 'delete', 'action': 'clear_detections', 'description': 'Clear all detections'},
#     44: {'key': 'insert', 'action': 'insert_marker', 'description': 'Insert frame marker'},
#     45: {'key': 'home', 'action': 'go_to_start', 'description': 'Go to video start'},
#     46: {'key': 'end', 'action': 'go_to_end', 'description': 'Go to video end'},
#     47: {'key': 'escape', 'action': 'emergency_stop', 'description': 'Emergency stop'}
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
    'dim_delay_ms': 300,              # Delay before dimming after button press
    'dim_to_brightness': 30,          # Brightness to dim to after button press
}

# LED-to-Button Mapping
# Maps button indices (0-47) to LED indices (0-47)
# If a button doesn't have a mapping, no LED will be controlled for that button
# Example: Button 0 controls LED 5, Button 1 controls LED 12, etc.
LED_BUTTON_MAPPING = {
    # Button index: LED index
    0: 5,  
    1: 12,
    2: 19,   
    3: 26,  
    4: 33,  
    5: 40,   
    6: 47,   
    7: 6,   
    8: 13,  
    9: 20,  
    10: 27, 
    11: 34, 
    12: 41, 
    13: 7,   
    14: 14, 
    15: 21,
    16: 28,  
    17: 35,  
    18: 42,  
    19: 8,  
    20: 15,  
    21: 22, 
    22: 29,
    23: 36, 
    24: 43,  
    25: 9,   
    26: 16,  
    27: 23, 
    28: 30, 
    29: 37, 
    30: 44,  
    31: 10,  
    32: 17,  
    33: 24,  
    34: 31,  
    35: 38,  
    36: 45,  
    37: 11,  
    38: 18,  
    39: 25, 
    40: 32, 
    41: 39,
    42: 46, 
    43: 0,  
    44: 1, 
    45: 2,
    46: 3, 
    47: 4, 
}

# Button Action Definitions
BUTTON_ACTIONS = {
    'exit': {'type': 'immediate', 'requires_confirmation': False},
    'next_video': {'type': 'immediate', 'requires_confirmation': False},
    'previous_video': {'type': 'immediate', 'requires_confirmation': False},
    'restart_video': {'type': 'immediate', 'requires_confirmation': False},
    'pause_resume': {'type': 'toggle', 'requires_confirmation': False},
    'fast_forward': {'type': 'hold', 'requires_confirmation': False},
    'rewind': {'type': 'hold', 'requires_confirmation': False},
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
    'toggle_memory_cleanup': {'type': 'toggle', 'requires_confirmation': False},
    'toggle_audio': {'type': 'toggle', 'requires_confirmation': False},
    'toggle_training_mode': {'type': 'toggle', 'requires_confirmation': False},
    'toggle_output': {'type': 'toggle', 'requires_confirmation': False},
    'reset_display': {'type': 'immediate', 'requires_confirmation': True},
    'export_results': {'type': 'immediate', 'requires_confirmation': False},
    'save_screenshot': {'type': 'immediate', 'requires_confirmation': False},
    'toggle_ui': {'type': 'toggle', 'requires_confirmation': False},
    'jump_frame_forward': {'type': 'immediate', 'requires_confirmation': False},
    'jump_frame_backward': {'type': 'immediate', 'requires_confirmation': False},
    'toggle_edge_detection': {'type': 'toggle', 'requires_confirmation': False},
    'cycle_view_mode': {'type': 'immediate', 'requires_confirmation': False},
    'confirm_action': {'type': 'immediate', 'requires_confirmation': False},
    'undo_action': {'type': 'immediate', 'requires_confirmation': False},
    'clear_detections': {'type': 'immediate', 'requires_confirmation': True},
    'insert_marker': {'type': 'immediate', 'requires_confirmation': False},
    'go_to_start': {'type': 'immediate', 'requires_confirmation': False},
    'go_to_end': {'type': 'immediate', 'requires_confirmation': False},
    'emergency_stop': {'type': 'immediate', 'requires_confirmation': False}
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'ERROR',
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S'
} 