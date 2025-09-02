"""
Multi-Application Manager for Object Detection
Manages multiple VideoInferenceApp instances in a single thread to prevent OpenCV conflicts.
"""

import logging
import time
import signal
import sys
import threading
from pathlib import Path
from typing import Dict, Optional
from config.config import APPLICATIONS, SCREEN_CONFIG
from application.app import VideoInferenceApp
from application.button_handler import ButtonHandler
import cv2


class MultiAppManager:
    """
    Manages multiple object detection applications in a single thread.
    Prevents OpenCV window conflicts and GPU resource issues.
    """
    
    def __init__(self):
        self.apps: Dict[str, VideoInferenceApp] = {}
        self.app_states: Dict[str, dict] = {}
        self.button_handler: Optional[ButtonHandler] = None
        self.is_running = False
        self.shutdown_event = False
        self._state_lock = threading.Lock()  # Thread safety for app states
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logging.info("MultiAppManager initialized")
    
    def _signal_handler(self, signum: int, frame):
        """Handle system signals for graceful shutdown"""
        logging.info(f"Received signal {signum}, shutting down gracefully...")
        self.shutdown_event = True
        self.stop_all_apps()
        sys.exit(0)
    
    def initialize_applications(self) -> bool:
        """Initialize all enabled applications from configuration"""
        try:
            # Create shared button handler
            self.button_handler = ButtonHandler()
            
            # Initialize each enabled application
            for app_id, app_config in APPLICATIONS.items():
                if not app_config.get('enabled', False):
                    logging.info(f"Skipping disabled application: {app_id}")
                    continue
                
                if not self._initialize_single_app(app_id, app_config):
                    logging.error(f"Failed to initialize application: {app_id}")
                    continue
                
                logging.info(f"Successfully initialized application: {app_id}")
            
            if not self.apps:
                logging.error("No applications were initialized successfully")
                return False
            
            # Start button handler
            if not self.button_handler.start_serial_monitoring():
                logging.error("Failed to start button handler")
            else: 
                logging.info(f"Initialized {len(self.apps)} applications successfully")
                
            return True
        
        except Exception as e:
            logging.error(f"Error initializing applications: {e}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _initialize_single_app(self, app_id: str, app_config: dict) -> bool:
        """Initialize a single application instance"""
        try:
            # Validate configuration
            required_fields = ['video_folders', 'model_path', 'screen_id']
            for field in required_fields:
                if field not in app_config:
                    logging.error(f"Missing required field '{field}' for application {app_id}")
                    return False
            
            # Check if paths exist
            video_folders = app_config['video_folders']
            if not isinstance(video_folders, list) or len(video_folders) == 0:
                logging.error(f"video_folders must be a non-empty list for application {app_id}")
                return False
            
            # Check if first video folder exists
            first_video_path = Path(video_folders[0])
            model_path = Path(app_config['model_path'])
            
            if not first_video_path.exists():
                logging.error(f"First video folder not found: {first_video_path}")
                return False
            
            if not model_path.exists():
                logging.error(f"Model not found: {model_path}")
                return False
            
            # Check if first video folder contains .mp4 files
            video_files = [f for f in first_video_path.iterdir() if f.suffix.lower() == '.mp4']
            if not video_files:
                logging.error(f"No .mp4 files found in first video folder: {first_video_path}")
                return False
            
            # Create application instance with video_folders list
            app = VideoInferenceApp(
                video_path=video_folders,  # Pass the list of folders
                model_path=str(model_path),
                box_threshold=0.25,  # Default threshold
                app_id=app_id,
                screen_id=app_config['screen_id']
            )
            
            # Set button handler
            app.set_button_handler(self.button_handler)
            
            # Set multi-app manager reference for button actions
            app._multi_app_manager = self
            
            # Add to button handler
            self.button_handler.add_app_instance(app_id, app)
            
            # Store application and initialize state
            self.apps[app_id] = app
            self.app_states[app_id] = {
                'current_video_index': 0,
                'current_folder_index': 0,  # Track which folder we're in
                'video_files': video_files,
                'video_folders': video_folders,  # Store the list of folders
                'cap': None,
                'out_writer': None,
                'frame_count': 0,
                'first_frame': True,
                'fullscreen_size': None,
                'window_name': f'Object Detection - {app_id}',
                            'is_active': True,
            'next_video_requested': False,
            'next_video_request_count': 0,
            'last_next_video_time': 0
            }
            
            logging.info(f"Application {app_id} initialized: {len(video_files)} video files found")
            return True
            
        except Exception as e:
            logging.error(f"Error initializing application {app_id}: {e}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def run_applications_single_thread(self) -> bool:
        """Run all applications in a single thread to prevent OpenCV conflicts"""
        if not self.apps:
            logging.error("No applications to run")
            return False
        
        try:
            self.is_running = True
            logging.info("Starting applications in single-threaded mode...")
            
            # Initialize video captures for all apps
            for app_id, app in self.apps.items():
                self._initialize_app_video(app_id, app)
            
            # Main processing loop
            while self.is_running and not self.shutdown_event:
                try:
                    # Process one frame from each active application
                    active_apps = [app_id for app_id, state in self.app_states.items() if state['is_active']]
                    
                    if not active_apps:
                        logging.info("No active applications, stopping")
                        break
                    
                    # Process frame from each app
                    for app_id in active_apps:
                        if not self._process_app_frame(app_id):
                            # App finished or encountered error
                            self.app_states[app_id]['is_active'] = False
                    
                    # Handle global key input and pass to individual apps
                    # Use proper frame rate control based on video FPS
                    wait_ms = 33  # Default to ~30 FPS if no specific FPS available
                    if self.app_states:
                        # Use the first app's FPS as reference
                        first_app_state = next(iter(self.app_states.values()))
                        if 'fps' in first_app_state and first_app_state['fps'] > 0:
                            wait_ms = int(1000 / first_app_state['fps'])
                    
                    key = cv2.waitKey(wait_ms) & 0xFF
                    if key == 27:  # ESC
                        logging.info("ESC pressed, shutting down")
                        break
                    elif key == ord('q'):
                        logging.info("Q pressed, shutting down")
                        break
                    elif key != 0:  # Pass non-zero keys to active applications
                        # Pass key to all active applications for handling
                        for app_id in active_apps:
                            try:
                                app = self.apps[app_id]
                                app_state = self.app_states[app_id]
                                # Handle key input for each app
                                should_exit, should_next_video = app._handle_key_input(key, app_state['cap'], app_state['out_writer'])
                                if should_exit:
                                    logging.info(f"App {app_id} requested exit")
                                    self.app_states[app_id]['is_active'] = False
                                elif should_next_video:
                                    logging.info(f"App {app_id} requested next video")
                                    self._next_video_or_finish(app_id)
                            except Exception as e:
                                logging.warning(f"Error handling key input for app {app_id}: {e}")
                    
                    # Small delay to prevent busy waiting
                    time.sleep(0.01)
                    
                except KeyboardInterrupt:
                    logging.info("Keyboard interrupt received")
                    break
                except Exception as e:
                    logging.error(f"Error in main processing loop: {e}")
                    break
            
            logging.info("Main processing loop completed")
            return True
            
        except Exception as e:
            logging.error(f"Error running applications: {e}")
            return False
    
    def _initialize_app_video(self, app_id: str, app: VideoInferenceApp):
        """Initialize video capture for a specific application"""
        try:
            state = self.app_states[app_id]
            video_files = state['video_files']
            video_folders = state['video_folders']
            current_folder_index = state['current_folder_index']
            
            if state['current_video_index'] >= len(video_files):
                logging.info(f"App {app_id}: All videos processed")
                state['is_active'] = False
                return
            
            # Get the current folder path
            if current_folder_index >= len(video_folders):
                logging.error(f"App {app_id}: Invalid folder index {current_folder_index}")
                state['is_active'] = False
                return
            
            current_folder = Path(video_folders[current_folder_index])
            video_file = video_files[state['current_video_index']]
            video_path = current_folder / video_file
            
            logging.info(f"App {app_id}: Initializing video {video_file} from folder {current_folder}")
            
            # Open video capture
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logging.error(f"App {app_id}: Could not open video {video_path}")
                state['is_active'] = False
                return
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 25.0
            
            # Setup window
            window_name = state['window_name']
            app._setup_fullscreen_window(window_name)
            
            # Store capture
            state['cap'] = cap
            state['fps'] = fps
            state['wait_ms'] = int(1000 / fps)
            
            logging.info(f"App {app_id}: Video initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing video for app {app_id}: {e}")
            state['is_active'] = False
    
    def _process_app_frame(self, app_id: str) -> bool:
        """Process a single frame from a specific application"""
        try:
            state = self.app_states[app_id]
            app = self.apps[app_id]
            
            if not state['is_active'] or state['cap'] is None:
                return False
            
            cap = state['cap']
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                logging.info(f"App {app_id}: End of video reached")
                self._next_video_or_finish(app_id)
                return True
            
            # Check for button next video signal
            if state.get('next_video_requested', False):
                logging.info(f"App {app_id}: Button next video signal received")
                state['next_video_requested'] = False  # Reset the flag
                
                # Safety check: prevent too many rapid requests
                if state.get('next_video_request_count', 0) > 10:
                    logging.warning(f"App {app_id}: Too many rapid next video requests, ignoring")
                    state['next_video_request_count'] = 0
                    return True
                
                # Safety check: prevent hanging requests (timeout after 5 seconds)
                import time
                current_time = time.time()
                if current_time - state.get('last_next_video_time', 0) > 5.0:
                    logging.warning(f"App {app_id}: Next video request timed out, ignoring")
                    state['next_video_requested'] = False
                    state['next_video_request_count'] = 0
                    return True
                
                try:
                    self._next_video_or_finish(app_id)
                    logging.info(f"App {app_id}: Successfully moved to next video")
                except Exception as e:
                    logging.error(f"App {app_id}: Error moving to next video: {e}")
                    # Try to recover by reinitializing the video
                    try:
                        self._initialize_app_video(app_id, app)
                    except Exception as e2:
                        logging.error(f"App {app_id}: Failed to recover video: {e2}")
                        state['is_active'] = False
                return True
            
            # Process frame
            detections, inf_time = app._process_frame(frame)
            
            # Generate Grad-CAM if enabled
            gradcam_img = None
            if app.display_config.gradcam_enabled and app.gradcam_processor is not None:
                try:
                    gradcam_img = app.gradcam_processor.process_gradcam(
                        frame, detections, app.display_config.gradcam_in_box_only
                    )
                except Exception as e:
                    logging.warning(f"App {app_id}: GradCAM processing failed: {e}")
                    gradcam_img = frame.copy()
            else:
                gradcam_img = frame.copy()
            
            # Prepare display frame with proper GradCAM
            display_img = app._prepare_display_frame(frame, detections, gradcam_img)
            
            # Display frame
            window_name = state['window_name']
            first_frame = state['first_frame']
            fullscreen_size = state['fullscreen_size']
            
            try:
                first_frame, fullscreen_size = app._display_frame_fullscreen(
                    window_name, display_img, first_frame, fullscreen_size
                )
                state['first_frame'] = first_frame
                state['fullscreen_size'] = fullscreen_size
            except Exception as e:
                logging.warning(f"App {app_id}: Error displaying frame: {e}")
            
            # Update frame count
            state['frame_count'] += 1
            
            # Debug output every 30 frames
            if state['frame_count'] % 30 == 0:
                logging.debug(f"App {app_id}: Frame {state['frame_count']}, detections: {len(detections)}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error processing frame for app {app_id}: {e}")
            return False
    
    def _next_video_or_finish(self, app_id: str):
        """Move to next video or finish application"""
        try:
            state = self.app_states[app_id]
            app = self.apps[app_id]
            
            # Clean up current video
            if state['cap']:
                state['cap'].release()
                state['cap'] = None
            
            if state['out_writer']:
                state['out_writer'].release()
                state['out_writer'] = None
            
            # Move to next video
            state['current_video_index'] += 1
            state['frame_count'] = 0
            state['first_frame'] = True
            state['fullscreen_size'] = None
            state['next_video_requested'] = False  # Reset the flag
            
            # Check if we need to move to the next folder
            if state['current_video_index'] >= len(state['video_files']):
                # Move to next folder
                state['current_folder_index'] += 1
                state['current_video_index'] = 0  # Reset video index for new folder
                
                if state['current_folder_index'] < len(state['video_folders']):
                    # Initialize videos for the new folder
                    new_folder = Path(state['video_folders'][state['current_folder_index']])
                    logging.info(f"App {app_id}: Moving to next folder: {new_folder}")
                    
                    # Get video files from the new folder
                    new_video_files = [f for f in new_folder.iterdir() if f.suffix.lower() == '.mp4']
                    if new_video_files:
                        state['video_files'] = new_video_files
                        logging.info(f"App {app_id}: Found {len(new_video_files)} videos in new folder")
                        # Initialize the first video in the new folder
                        self._initialize_app_video(app_id, app)
                    else:
                        logging.warning(f"App {app_id}: No videos found in folder {new_folder}")
                        # Try the next folder
                        self._next_video_or_finish(app_id)
                else:
                    # All folders processed
                    logging.info(f"App {app_id}: All folders completed")
                    state['is_active'] = False
            else:
                # Initialize next video in current folder
                self._initialize_app_video(app_id, app)
                
        except Exception as e:
            logging.error(f"Error moving to next video for app {app_id}: {e}")
            state['is_active'] = False
    
    def signal_next_video(self, app_id: str) -> bool:
        """Signal that a specific application should move to the next video"""
        if app_id not in self.app_states:
            logging.warning(f"Cannot signal next video: app {app_id} not found")
            return False
        
        try:
            logging.info(f"Signaling next video for app {app_id}")
            # Use thread-safe approach with lock
            with self._state_lock:
                if app_id in self.app_states:
                    import time
                    current_time = time.time()
                    self.app_states[app_id]['next_video_requested'] = True
                    self.app_states[app_id]['next_video_request_count'] += 1
                    self.app_states[app_id]['last_next_video_time'] = current_time
                    logging.debug(f"Next video flag set for app {app_id} (count: {self.app_states[app_id]['next_video_request_count']})")
            return True
        except Exception as e:
            logging.error(f"Error signaling next video for app {app_id}: {e}")
            return False
    
    def signal_next_folder(self, app_id: str) -> bool:
        """Signal that a specific application should switch to the next folder"""
        if app_id not in self.app_states:
            logging.warning(f"Cannot signal next folder: app {app_id} not found")
            return False
        
        try:
            logging.info(f"Signaling next folder for app {app_id}")
            # Use thread-safe approach with lock
            with self._state_lock:
                if app_id in self.app_states:
                    state = self.app_states[app_id]
                    
                    # Clean up current video
                    if state['cap']:
                        state['cap'].release()
                        state['cap'] = None
                    
                    if state['out_writer']:
                        state['out_writer'].release()
                        state['out_writer'] = None
                    
                    # Move to next folder
                    state['current_folder_index'] += 1
                    state['current_video_index'] = 0  # Reset video index for new folder
                    state['frame_count'] = 0
                    state['first_frame'] = True
                    state['fullscreen_size'] = None
                    
                    if state['current_folder_index'] < len(state['video_folders']):
                        # Initialize videos for the new folder
                        new_folder = Path(state['video_folders'][state['current_folder_index']])
                        logging.info(f"App {app_id}: Switching to folder: {new_folder}")
                        
                        # Get video files from the new folder
                        new_video_files = [f for f in new_folder.iterdir() if f.suffix.lower() == '.mp4']
                        if new_video_files:
                            state['video_files'] = new_video_files
                            logging.info(f"App {app_id}: Found {len(new_video_files)} videos in new folder")
                            # Initialize the first video in the new folder
                            app = self.apps[app_id]
                            self._initialize_app_video(app_id, app)
                        else:
                            logging.warning(f"App {app_id}: No videos found in folder {new_folder}")
                            # Try the next folder recursively
                            return self.signal_next_folder(app_id)
                    else:
                        # All folders processed, loop back to first folder
                        state['current_folder_index'] = 0
                        first_folder = Path(state['video_folders'][0])
                        logging.info(f"App {app_id}: Looping back to first folder: {first_folder}")
                        
                        # Get video files from the first folder
                        first_video_files = [f for f in first_folder.iterdir() if f.suffix.lower() == '.mp4']
                        if first_video_files:
                            state['video_files'] = first_video_files
                            logging.info(f"App {app_id}: Found {len(first_video_files)} videos in first folder")
                            # Initialize the first video in the first folder
                            app = self.apps[app_id]
                            self._initialize_app_video(app_id, app)
                        else:
                            logging.error(f"App {app_id}: No videos found in first folder")
                            state['is_active'] = False
                    
                    logging.debug(f"Next folder signal processed for app {app_id}")
            return True
        except Exception as e:
            logging.error(f"Error signaling next folder for app {app_id}: {e}")
            return False
    
    def stop_all_apps(self) -> None:
        """Stop all running applications"""
        logging.info("Stopping all applications...")
        
        self.is_running = False
        
        # Clean up all video captures and writers
        for app_id, state in self.app_states.items():
            try:
                if state['cap']:
                    state['cap'].release()
                if state['out_writer']:
                    state['out_writer'].release()
            except Exception as e:
                logging.warning(f"Error cleaning up app {app_id}: {e}")
        
        # Stop button handler
        if self.button_handler:
            try:
                self.button_handler.stop()
                logging.info("Button handler stopped")
            except Exception as e:
                logging.warning(f"Error stopping button handler: {e}")
        
        # Clean up app resources
        for app_id, app in self.apps.items():
            try:
                app._cleanup_resources()
            except Exception as e:
                logging.warning(f"Error cleaning up app {app_id}: {e}")
        
        # Close all OpenCV windows
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            logging.warning(f"Error closing OpenCV windows: {e}")
        
        logging.info("All applications stopped")
    
    def get_app_status(self) -> Dict[str, dict]:
        """Get status of all applications"""
        status = {}
        
        for app_id, app in self.apps.items():
            state = self.app_states.get(app_id, {})
            
            # Get folder information if available
            folder_info = {}
            if hasattr(app, 'get_current_folder_info'):
                folder_info = app.get_current_folder_info()
            
            # Get additional folder statistics
            folder_stats = {}
            if hasattr(app, 'get_folder_video_counts'):
                folder_stats = app.get_folder_video_counts()
            
            total_videos = 0
            if hasattr(app, 'get_total_video_count'):
                total_videos = app.get_total_video_count()
            
            status[app_id] = {
                'running': state.get('is_active', False),
                'current_video': state.get('current_video_index', 0),
                'total_videos': total_videos,
                'frame_count': state.get('frame_count', 0),
                'app_id': app.app_id,
                'screen_id': app.screen_id,
                'video_path': app.video_path,
                'model_path': app.model_path,
                'folder_info': folder_info,
                'folder_stats': folder_stats
            }
        
        return status
    
    def get_button_handler_status(self) -> dict:
        """Get status of the button handler"""
        if self.button_handler:
            return self.button_handler.get_connection_health()
        return {'error': 'Button handler not initialized'}
    
    def run(self) -> None:
        """Main run loop for the multi-application manager"""
        try:
            # Initialize applications
            if not self.initialize_applications():
                logging.error("Failed to initialize applications")
                return
            
            logging.info("Multi-application system initialized successfully")
            logging.info("Press Ctrl+C or ESC to stop all applications gracefully")
            
            # Run applications in single-threaded mode
            self.run_applications_single_thread()
            
        except KeyboardInterrupt:
            logging.info("Keyboard interrupt received - shutting down...")
        except Exception as e:
            logging.error(f"Error in multi-application manager: {e}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
        finally:
            logging.info("Shutting down multi-application system...")
            self.stop_all_apps()
            logging.info("Multi-application system shutdown complete")


def main():
    """Main entry point for multi-application system"""
    manager = MultiAppManager()
    manager.run()


if __name__ == "__main__":
    main()
