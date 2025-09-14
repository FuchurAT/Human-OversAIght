# Object Detection Application

A modular object detection application built with YOLO, featuring real-time video processing, Grad-CAM visualization, and multi-monitor support.

**Part of the [Human Oversaight: The Ops Room](https://humanoversaight.cargo.site/) installation** - An interactive audio-visual installation that questions the role of human oversight of high-risk AI systems, established by the European Artificial Intelligence Act: Article 14.

## Project Context

This object detection application is the core technical component of **Human Oversaight: The Ops Room**, an interactive installation that explores human oversight of high-risk AI systems. The installation features:

- **Real-time computer vision** processing of curated video footage
- **Custom-trained models** for detecting high-risk scenarios (armed police, military operations, protests, weapons, environmental catastrophes)
- **Physical interface** with 48 buttons for human intervention
- **Multi-channel video display** across LED modules in semi-arc formation
- **Logging system** that tracks all human interactions with the AI system

The installation was presented at the **Ars Electronica Festival 2025** (September 3-7, 2025) in Linz, Austria, and represents a human-computer interface for AI oversight and stop-button functionality.

For more information about the broader project, visit: [https://humanoversaight.cargo.site/](https://humanoversaight.cargo.site/)

## Architecture Overview

### Core Classes

#### `Detection`
- Data class for storing detection information (bounding box, confidence, class ID)
- Immutable structure for better data integrity

#### `DisplayConfig`
- Configuration dataclass for all display-related settings
- Centralized configuration management

#### `MemoryManager`
- Handles memory monitoring and cleanup
- Automatic garbage collection and PyTorch cache management
- Configurable cleanup intervals

#### `ColorManager`
- Static utility class for color management
- Confidence-based color interpolation
- State-aware color selection

#### `BoxManager`
- Handles bounding box operations and tracking
- IoU calculations and overlap filtering
- Box interpolation for smooth animations

#### `GradCAMProcessor`
- Manages Grad-CAM visualization generation
- Frame buffering for performance optimization
- Configurable processing intervals

#### `DetectionVisualizer`
- Handles all visualization aspects
- Drawing detection overlays, corners, and effects
- FPS and legend display
- Glitch effects and enemy indicators

#### `VideoInferenceApp`
- Main application orchestrator
- Video processing pipeline
- User input handling
- Resource management

#### `ButtonHandler`
- Handles button inputs from Arduino Mega (48-button interface)
- Maps button presses to keyboard actions
- Supports multiple application instances
- Integrates with LED feedback system
- Serial communication management with debouncing

#### `LEDController`
- Controls 48 LED brightness levels via PWM over serial
- Arduino Nano integration for visual feedback
- Thread-safe LED value updates
- Fade, pulse, and brightness effects
- Button press visual feedback

#### `NDISender`
- Network Device Interface video streaming
- Sends processed frames to NDI-compatible receivers
- Multiple video format support (BGRX, RGBX, UYVY)
- Frame resizing and format conversion
- Graceful fallback when NDI library unavailable

#### `NDISenderDummy`
- Fallback implementation when NDI library not available
- Maintains API compatibility
- Prevents application crashes

#### `CountHandler`
- Tracks button and key press counts
- Displays counters on output frames
- Persistent count storage to text files
- Configurable display positioning and styling

#### `MultiAppManager`
- Manages multiple VideoInferenceApp instances
- Prevents OpenCV window conflicts
- Single-threaded execution for stability
- Coordinated video switching across applications
- Shared button handler integration

#### `ArmoredVehiclesTrainer`
- YOLO model training for armored vehicle detection
- Dataset splitting and validation
- Model configuration management
- Training progress monitoring
- Comprehensive logging and metrics

#### `AutoAnnotator`
- Automated dataset annotation using Grounding DINO
- Text-based object detection and labeling
- YOLO format output generation
- Batch processing capabilities
- Confidence threshold management

## Key Features

### Real-time Object Detection
- YOLO model integration with configurable confidence thresholds
- Efficient frame processing with memory management
- Support for multiple video formats

### Advanced Visualization
- Custom bounding box rendering with corner indicators
- Confidence-based color coding
- Glow effects and enemy indicators
- Configurable blur/pixelation effects

### Grad-CAM Integration
- Real-time attention visualization
- Performance-optimized processing
- Box-specific or full-frame overlays

### Multi-Monitor Support
- Fullscreen video playback on secondary monitors
- Legend and FPS display on separate screens
- Automatic monitor detection and positioning

### Hardware Integration
- Arduino Mega 48-button interface support
- Arduino Nano LED control system
- Serial communication with debouncing
- Visual and audio feedback systems
- Button-to-action mapping configuration

### Multi-Application Management
- Simultaneous multiple detection applications
- Single-threaded execution to prevent conflicts
- Coordinated video switching across apps
- Shared button handler for all instances
- Individual application state tracking

### OBS Studio Integration
- Automatic OBS Studio startup with scene collection management
- Preview projector positioning on designated monitors
- Real-time video streaming and recording capabilities
- Scene collection "theopsroom" for installation-specific setup
- Automatic OBS configuration and state management

### Autostart System
- Comprehensive autostart script for Ubuntu 24.04 compatibility
- Multi-monitor detection and setup
- Conda environment management
- Audio system configuration (PipeWire/PulseAudio)
- OBS Studio integration with automatic positioning
- System readiness checks and dependency validation

### User Controls

#### Keyboard Controls
- **Space**: Toggle border color and enemy indicators
- **G**: Toggle Grad-CAM visualization
- **W**: Toggle glitch effects
- **B**: Toggle Grad-CAM box mode
- **L**: Toggle legend display
- **F**: Toggle FPS information
- **N**: Next video
- **Q/ESC**: Quit application

#### Hardware Button Controls
- **48-button Arduino Mega interface**: Configurable button mapping
- **LED feedback**: Visual confirmation of button presses
- **Serial communication**: Real-time button state monitoring
- **Multi-app support**: Buttons can control specific or all applications
- **Threshold adjustment**: Direct confidence threshold control via buttons

## Configuration

All application settings are centralized in `config.py`:

- **Thresholds**: Detection confidence, IoU, memory cleanup intervals
- **Display**: Colors, dimensions, effects parameters
- **Performance**: Frame skip thresholds, buffer sizes
- **Paths**: Default model and video directories
- **Hardware**: Button mapping, LED configuration, serial ports
- **Multi-app**: Application instances, screen assignments, video folders
- **NDI**: Video streaming settings, format configuration
- **Counter**: Display positioning, styling, persistence settings
- **OBS**: Scene collection, preview positioning, streaming settings
- **Autostart**: System readiness, monitor detection, audio configuration

## Usage

### Basic Usage
```python
from inference import VideoInferenceApp

app = VideoInferenceApp(
    video_path="/path/to/videos/",
    model_path="/path/to/model.pt",
    box_threshold=0.25
)
app.run()
```

### Command Line
```bash
# Single application
python inference.py

# Multi-application system
python run_multi_apps.py

# Autostart with full system setup
./run_inference.sh
```

### OBS Studio Integration
```bash
# Manual OBS startup with scene collection
obs-studio --collection "theopsroom" --startpreview

# OBS with specific graphics backend
obs-studio --collection "theopsroom" --gl-backend nvidia

# OBS configuration check
./run_inference.sh --debug-obs
```

### Autostart System
```bash
# Full autostart with system checks
./run_inference.sh

# Autostart with specific arguments
./run_inference.sh --debug --verbose

# Check system readiness
./run_inference.sh --check-dependencies
```

### Custom Configuration
```python
from config import *
from inference import VideoInferenceApp

# Override defaults
DEFAULT_BOX_THRESHOLD = 0.3
DEFAULT_IOU_THRESHOLD = 0.15

app = VideoInferenceApp(video_path, model_path)
app.run()
```

## OBS Studio Setup

### Scene Collection Configuration
The installation uses a specific OBS scene collection called "theopsroom":

```bash
# Scene collection location
~/.config/obs-studio/basic/scenes/theopsroom/
# or for Flatpak installations:
~/.var/app/com.obsproject.Studio/config/obs-studio/basic/scenes/theopsroom/
```

### OBS Configuration
The autostart script automatically configures:
- Scene collection: "theopsroom"
- Preview projector positioning
- Graphics backend selection (NVIDIA/X11)
- Safe mode startup with shutdown check disabled

### Manual OBS Configuration
```bash
# Create scene collection
mkdir -p ~/.config/obs-studio/basic/scenes/theopsroom

# Configure global settings
echo "SceneCollection=theopsroom" >> ~/.config/obs-studio/global.ini
echo "FirstRun=false" >> ~/.config/obs-studio/global.ini
```

## Autostart System

### System Requirements
- **OS**: Ubuntu 24.04 (tested), compatible with other Linux distributions
- **Display**: Multi-monitor setup (minimum 3 screens recommended)
- **Audio**: PipeWire or PulseAudio
- **Graphics**: NVIDIA or Intel/AMD drivers
- **Python**: Conda environment with "cnn-detection" environment

### Autostart Features
The `run_inference.sh` script provides:

#### System Readiness Checks
- User session detection
- X server availability
- Conda environment validation
- Audio system setup
- Monitor detection and configuration

#### Multi-Monitor Setup
- Automatic monitor detection via `xrandr`
- Screen positioning and geometry calculation
- Monitor validation and connection status
- Fallback handling for missing monitors

#### Audio System Configuration
- **PipeWire**: Modern audio system with PulseAudio compatibility
- **PulseAudio**: Legacy audio system support
- Automatic audio device detection and configuration
- Runtime directory creation and permissions

### Autostart Logging
All autostart activities are logged to:
```bash
~/.human_oversaight_autostart.log
```

### Troubleshooting Autostart
```bash
# Check autostart log
tail -f ~/.human_oversaight_autostart.log

# Debug monitor detection
./run_inference.sh --debug-monitors

# Debug OBS startup
./run_inference.sh --debug-obs

# Check system dependencies
./run_inference.sh --check-deps
```

## Performance Optimizations

### Memory Management
- Automatic PyTorch cache cleanup
- Frame buffer reuse
- Configurable cleanup intervals
- Garbage collection optimization

### Processing Efficiency
- Frame skipping for Grad-CAM
- Buffered frame operations
- Efficient IoU calculations
- Optimized visualization rendering

### Display Optimization
- Aspect ratio preservation
- Fullscreen optimization
- Multi-monitor coordination
- Reduced frame allocations

## Dependencies

### Core Dependencies
- `opencv-python`: Video processing and display
- `torch`: PyTorch for model inference
- `ultralytics`: YOLO model loading
- `numpy`: Numerical operations
- `PIL`: Image processing

### Optional Dependencies
- `screeninfo`: Multi-monitor support
- `pyserial`: Arduino serial communication
- `NDI-Python`: Network Device Interface streaming
- `groundingdino`: Automated dataset annotation

### System Dependencies (for Autostart)
- `obs-studio`: OBS Studio for video streaming/recording
- `xrandr`: Multi-monitor management
- `xdotool`: Window positioning and automation
- `wmctrl`: Window management
- `pactl`: PulseAudio control
- `pw-cli`: PipeWire control
- `conda`: Python environment management

## File Structure

```
object-detection/
├── inference.py              # Main single application
├── run_multi_apps.py         # Multi-application runner
├── run_inference.sh          # Autostart script with system setup
├── train.py                  # YOLO training script
├── annotate.py               # Automated annotation tool
├── config/
│   ├── config.py            # Main configuration
│   └── classes.py           # YOLO class definitions
├── application/
│   ├── app.py               # Main application class
│   ├── button_handler.py    # Arduino button interface
│   ├── led_controller.py    # Arduino LED control
│   ├── ndi_sender.py        # NDI video streaming
│   ├── count_handler.py     # Button/key counting
│   ├── multi_app_manager.py # Multi-app coordination
│   ├── models.py            # Data models
│   ├── memory_manager.py    # Memory management
│   ├── color_manager.py     # Color utilities
│   ├── box_manager.py       # Bounding box operations
│   ├── gradcam_processor.py # Grad-CAM processing
│   ├── visualizer.py        # Visualization engine
│   └── utils.py             # Utility functions
├── data/videos/              # Input video directories
├── runs/                     # Training outputs
└── count.txt                 # Persistent counter storage
```

## Error Handling

The application includes comprehensive error handling:

- **Model Loading**: Graceful fallback for missing models
- **Video Processing**: Skip corrupted files, continue processing
- **Memory Issues**: Automatic cleanup and recovery
- **Display Errors**: Fallback to windowed mode
- **Signal Handling**: Graceful shutdown on system signals
- **Hardware Communication**: Serial port error recovery and reconnection
- **NDI Streaming**: Fallback to dummy sender when library unavailable
- **Multi-app Coordination**: Individual app failure isolation
- **Button Debouncing**: Prevents rapid-fire button events

## Contributing

When contributing to this codebase:

1. **Follow the modular architecture** - Keep classes focused and single-purpose
2. **Use type hints** - All functions should include proper type annotations
3. **Add documentation** - Include docstrings for all public methods
4. **Update configuration** - Add new constants to `config.py`
5. **Maintain separation of concerns** - Don't mix visualization logic with processing logic

## Troubleshooting

### Common Issues

**High Memory Usage**
- Adjust `DEFAULT_MEMORY_CLEANUP_INTERVAL` in config
- Reduce `DEFAULT_FRAME_SKIP_THRESHOLD`
- Check for memory leaks in custom visualizations

**Performance Issues**
- Increase frame skip thresholds
- Reduce visualization complexity
- Check GPU memory availability

**Display Problems**
- Verify monitor configuration
- Check OpenCV installation
- Ensure proper window manager support

**Hardware Issues**
- Check Arduino serial port connections (`/dev/ttyACM0`, `/dev/ttyUSB0`)
- Verify button mapping configuration in `config.py`
- Test LED controller with `led_test_sequence.py`
- Check serial port permissions and availability

**Multi-Application Issues**
- Check video folder paths and file availability
- Verify button handler initialization across all apps
- Monitor memory usage with multiple instances

**OBS Studio Issues**
- Check OBS installation (Flatpak, snap, or system package)
- Verify scene collection "theopsroom" exists
- Ensure graphics drivers are properly installed
- Check OBS configuration in `~/.config/obs-studio/`
- Verify preview projector positioning on screen 3

**Autostart Issues**
- Check autostart log: `tail -f ~/.human_oversaight_autostart.log`
- Verify conda environment "cnn-detection" exists
- Ensure multi-monitor setup is properly configured
- Check audio system (PipeWire/PulseAudio) status
- Verify system dependencies are installed

### Debug Mode
Enable debug logging by modifying `LOGGING_CONFIG['level']` in `config.py`:
```python
LOGGING_CONFIG = {
    'level': 'DEBUG',
    # ... other settings
}
```
