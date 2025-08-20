# Object Detection Application

A modular object detection application built with YOLO, featuring real-time video processing, Grad-CAM visualization, and multi-monitor support.

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

### User Controls
- **Space**: Toggle border color and enemy indicators
- **G**: Toggle Grad-CAM visualization
- **W**: Toggle glitch effects
- **B**: Toggle Grad-CAM box mode
- **L**: Toggle legend display
- **F**: Toggle FPS information
- **N**: Next video
- **Q/ESC**: Quit application

## Configuration

All application settings are centralized in `config.py`:

- **Thresholds**: Detection confidence, IoU, memory cleanup intervals
- **Display**: Colors, dimensions, effects parameters
- **Performance**: Frame skip thresholds, buffer sizes
- **Paths**: Default model and video directories

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
python inference.py
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
- `kornia`: Advanced computer vision operations
- `matplotlib`: Plotting and visualization

## File Structure

```
object-detection/
├── inference.py          # Main application
├── config.py            # Configuration constants
├── settings.py          # YOLO class definitions
├── README.md            # This documentation
├── videos/              # Input video directory
└── models/              # Model weights directory
```

## Error Handling

The application includes comprehensive error handling:

- **Model Loading**: Graceful fallback for missing models
- **Video Processing**: Skip corrupted files, continue processing
- **Memory Issues**: Automatic cleanup and recovery
- **Display Errors**: Fallback to windowed mode
- **Signal Handling**: Graceful shutdown on system signals

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

### Debug Mode
Enable debug logging by modifying `LOGGING_CONFIG['level']` in `config.py`:
```python
LOGGING_CONFIG = {
    'level': 'DEBUG',
    # ... other settings
}
```
