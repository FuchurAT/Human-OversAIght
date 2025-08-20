#!/usr/bin/env python3
"""
Main entry point for the Object Detection application.
This file now serves as a simple launcher for the modular inference package.
"""

import logging
import argparse
import signal
import sys
from pathlib import Path

# Import from the new modular package
from application import VideoInferenceApp
from config.config import LOGGING_CONFIG, DEFAULT_MODEL_PATH, DEFAULT_VIDEO_PATH

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG['level']),
    format=LOGGING_CONFIG['format']
)


def main() -> None:
    """Main entry point for the application"""
    # Global variable to store the app instance for signal handling
    global app_instance
    
    def signal_handler(signum: int, frame) -> None:
        """Handle system signals for graceful shutdown"""
        logging.info(f"Received signal {signum}, shutting down gracefully...")
        if app_instance:
            app_instance._cleanup_resources()
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(description="Run real-time inference on video files with YOLO.")
    
    # Add command line arguments
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to YOLO model (default: {DEFAULT_MODEL_PATH})"
    )
    
    parser.add_argument(
        '--video', '-v',
        type=str,
        default=DEFAULT_VIDEO_PATH,
        help=f"Path to video directory (default: {DEFAULT_VIDEO_PATH})"
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25)"
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default="",
        help="Output video path (optional)"
    )
    
    parser.add_argument(
        '--max-frames', '-f',
        type=int,
        default=0,
        help="Maximum frames to process (0 = unlimited, default: 0)"
    )
    
    parser.add_argument(
        '--video-width', '-W',
        type=int,
        default=None,
        help="Video display width in pixels (default: auto-detect from screen)"
    )
    
    parser.add_argument(
        '--video-height', '-H',
        type=int,
        default=None,
        help="Video display height in pixels (default: auto-detect from screen)"
    )
    
    parser.add_argument(
        '--scale-mode',
        type=str,
        choices=['fit', 'stretch', 'original'],
        default='fit',
        help="Video scaling mode: fit (full screen with aspect ratio, default), stretch (full screen without aspect ratio), original (no scaling)"
    )
    
    parser.add_argument(
        '--scale-multiplier',
        type=float,
        default=0.95,
        help="Scale multiplier for fit mode (0.1-1.0, default: 0.95 = 95% of screen)"
    )
    
    parser.add_argument(
        '--maintain-aspect-ratio',
        action='store_true',
        default=True,
        help="Maintain video aspect ratio (default: True)"
    )
    
    parser.add_argument(
        '--center-video',
        action='store_true',
        default=True,
        help="Center video on screen (default: True)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.scale_multiplier < 0.1 or args.scale_multiplier > 1.0:
        logging.error(f"Scale multiplier must be between 0.1 and 1.0, got: {args.scale_multiplier}")
        sys.exit(1)
    
    # Validate paths
    if not Path(args.model).exists():
        logging.error(f"Model not found: {args.model}")
        sys.exit(1)
    
    if not Path(args.video).exists():
        logging.error(f"Video directory not found: {args.video}")
        sys.exit(1)
    
    # Check if video directory contains any .mp4 files
    video_files = [f for f in Path(args.video).iterdir() if f.suffix.lower() == '.mp4']
    if not video_files:
        logging.error(f"No .mp4 files found in video directory: {args.video}")
        sys.exit(1)
    
    logging.info(f"Found {len(video_files)} video files: {[f.name for f in video_files]}")
    
    try:
        # Create and run the application
        app_instance = VideoInferenceApp(
            video_path=args.video,
            model_path=args.model,
            box_threshold=args.threshold
        )
        
        # Set additional options
        if args.output:
            app_instance.output_path = args.output
        if args.max_frames > 0:
            app_instance.max_frames = args.max_frames
        
        # Set video size configuration
        app_instance.set_video_size_config(
            width=args.video_width,
            height=args.video_height,
            scale_mode=args.scale_mode,
            maintain_aspect_ratio=args.maintain_aspect_ratio,
            center_video=args.center_video,
            scale_multiplier=args.scale_multiplier
        )
        
        logging.info(f"Starting Object Detection Application")
        logging.info(f"Model: {args.model}")
        logging.info(f"Video Directory: {args.video}")
        logging.info(f"Confidence Threshold: {args.threshold}")
        if args.output:
            logging.info(f"Output: {args.output}")
        if args.max_frames > 0:
            logging.info(f"Max Frames: {args.max_frames}")
        
        # Log video size configuration
        if args.video_width and args.video_height:
            logging.info(f"Video Size: {args.video_width}x{args.video_height}")
        else:
            logging.info("Video Size: Auto-detect from screen")
        logging.info(f"Scale Mode: {args.scale_mode}")
        logging.info(f"Scale Multiplier: {args.scale_multiplier}")
        logging.info(f"Maintain Aspect Ratio: {args.maintain_aspect_ratio}")
        logging.info(f"Center Video: {args.center_video}")
        
        app_instance.run()
        
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        if app_instance:
            app_instance._cleanup_resources()
    except Exception as e:
        logging.error(f"Error running inference: {e}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        if app_instance:
            app_instance._cleanup_resources()
        sys.exit(1)
    finally:
        # Ensure cleanup happens
        if app_instance:
            app_instance._cleanup_resources()
        logging.info("Application shutdown complete")


if __name__ == "__main__":
    main()