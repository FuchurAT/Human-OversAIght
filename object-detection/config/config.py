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
DEFAULT_FRAME_SKIP_THRESHOLD = 100

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
DEFAULT_MODEL_PATH = "E:/Projects/human-oversaight/object-detection/runs/detect/train/weights/best.pt"
DEFAULT_VIDEO_PATH = "E:/Projects/human-oversaight/data/videos/"

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S'
} 