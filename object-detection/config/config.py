"""
Configuration file for the Object Detection application.
Contains all constants, default values, and configuration settings.
"""

# Application Constants
ENABLE_UNMASK = True
ENABLE_GRAD_CAM_VIEW = True

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
DEFAULT_GRADCAM_ALPHA = 0.6

# Video Settings
DEFAULT_FPS = 25

# Window Settings
DEFAULT_WINDOW_WIDTH = 1920
DEFAULT_WINDOW_HEIGHT = 1080
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
BUTTON_MAPPING = {
    # All 48 buttons now act as space (toggle unmask/blur mode)
    0: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    1: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    2: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    3: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    4: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    5: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    6: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    7: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    8: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    9: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    10: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    11: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    12: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    13: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    14: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    15: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    16: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    17: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    18: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    19: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    20: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    21: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    22: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    23: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    24: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    25: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    26: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    27: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    28: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    29: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    30: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    31: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    32: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    33: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    34: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    35: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    36: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    37: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    38: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    39: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    40: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    41: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    42: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    43: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    44: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    45: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    46: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
    47: {'key': 'space', 'action': 'toggle_unmask', 'description': 'Toggle unmask/blur mode'},
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
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S'
} 