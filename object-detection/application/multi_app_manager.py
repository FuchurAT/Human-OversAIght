"""
Multi-Application Manager for Object Detection
Manages multiple VideoInferenceApp instances running in separate threads.
"""

import threading
import logging
import time
import signal
import sys
from pathlib import Path
from typing import Dict, List, Optional
from config.config import APPLICATIONS, SCREEN_CONFIG
from application.app import VideoInferenceApp
from application.button_handler import ButtonHandler


class MultiAppManager:
    """
    Manages multiple object detection applications running in separate threads.
    Provides centralized button handling and application lifecycle management.
    """
    
    def __init__(self):
        self.apps: Dict[str, VideoInferenceApp] = {}
        self.app_threads: Dict[str, threading.Thread] = {}
        self.app_stop_events: Dict[str, threading.Event] = {}
        self.button_handler: Optional[ButtonHandler] = None
        self.is_running = False
        self.shutdown_event = threading.Event()
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logging.info("MultiAppManager initialized")
    
    def _signal_handler(self, signum: int, frame):
        """Handle system signals for graceful shutdown"""
        logging.info(f"Received signal {signum}, shutting down gracefully...")
        self.shutdown_event.set()
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
                return False
            
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
            required_fields = ['video_folder', 'model_path', 'screen_id']
            for field in required_fields:
                if field not in app_config:
                    logging.error(f"Missing required field '{field}' for application {app_id}")
                    return False
            
            # Check if paths exist
            video_path = Path(app_config['video_folder'])
            model_path = Path(app_config['model_path'])
            
            if not video_path.exists():
                logging.error(f"Video folder not found: {video_path}")
                return False
            
            if not model_path.exists():
                logging.error(f"Model not found: {model_path}")
                return False
            
            # Check if video folder contains .mp4 files
            video_files = [f for f in video_path.iterdir() if f.suffix.lower() == '.mp4']
            if not video_files:
                logging.error(f"No .mp4 files found in video folder: {video_path}")
                return False
            
            # Create application instance
            app = VideoInferenceApp(
                video_path=str(video_path),
                model_path=str(model_path),
                box_threshold=0.25,  # Default threshold
                app_id=app_id,
                screen_id=app_config['screen_id']
            )
            
            # Set button handler
            app.set_button_handler(self.button_handler)
            
            # Add to button handler
            self.button_handler.add_app_instance(app_id, app)
            
            # Store application
            self.apps[app_id] = app
            self.app_stop_events[app_id] = threading.Event()
            
            logging.info(f"Application {app_id} initialized: {len(video_files)} video files found")
            return True
            
        except Exception as e:
            logging.error(f"Error initializing application {app_id}: {e}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def start_all_apps(self) -> bool:
        """Start all initialized applications in separate threads"""
        if not self.apps:
            logging.error("No applications to start")
            return False
        
        try:
            self.is_running = True
            
            for app_id, app in self.apps.items():
                if self._start_single_app(app_id, app):
                    logging.info(f"Started application: {app_id}")
                else:
                    logging.error(f"Failed to start application: {app_id}")
            
            # Wait for all threads to start
            time.sleep(1.0)
            
            # Check if all threads are running
            running_count = sum(1 for thread in self.app_threads.values() if thread.is_alive())
            logging.info(f"Started {running_count}/{len(self.apps)} applications")
            
            return running_count > 0
            
        except Exception as e:
            logging.error(f"Error starting applications: {e}")
            return False
    
    def _start_single_app(self, app_id: str, app: VideoInferenceApp) -> bool:
        """Start a single application in a separate thread"""
        try:
            # Create stop event
            stop_event = self.app_stop_events[app_id]
            
            # Create and start thread
            thread = threading.Thread(
                target=self._app_worker,
                args=(app_id, app, stop_event),
                daemon=True,
                name=f"AppThread-{app_id}"
            )
            
            self.app_threads[app_id] = thread
            thread.start()
            
            return True
            
        except Exception as e:
            logging.error(f"Error starting application {app_id}: {e}")
            return False
    
    def _app_worker(self, app_id: str, app: VideoInferenceApp, stop_event: threading.Event):
        """Worker thread for running an application"""
        try:
            logging.info(f"Application {app_id} worker thread started")
            
            # Run the application
            app.run()
            
            logging.info(f"Application {app_id} worker thread completed")
            
        except Exception as e:
            logging.error(f"Error in application {app_id} worker thread: {e}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
        finally:
            logging.info(f"Application {app_id} worker thread stopped")
    
    def stop_all_apps(self) -> None:
        """Stop all running applications"""
        logging.info("Stopping all applications...")
        
        self.is_running = False
        
        # Signal all threads to stop
        for app_id, stop_event in self.app_stop_events.items():
            stop_event.set()
            logging.info(f"Stop signal sent to application: {app_id}")
        
        # Wait for threads to finish
        for app_id, thread in self.app_threads.items():
            if thread.is_alive():
                logging.info(f"Waiting for application {app_id} to stop...")
                thread.join(timeout=5.0)
                if thread.is_alive():
                    logging.warning(f"Application {app_id} did not stop gracefully")
                else:
                    logging.info(f"Application {app_id} stopped")
        
        # Stop button handler
        if self.button_handler:
            self.button_handler.stop()
            logging.info("Button handler stopped")
        
        # Clean up resources
        for app_id, app in self.apps.items():
            try:
                app._cleanup_resources()
            except Exception as e:
                logging.warning(f"Error cleaning up app {app_id}: {e}")
        
        logging.info("All applications stopped")
    
    def stop_app(self, app_id: str) -> bool:
        """Stop a specific application"""
        if app_id not in self.apps:
            logging.warning(f"Application {app_id} not found")
            return False
        
        try:
            logging.info(f"Stopping application: {app_id}")
            
            # Signal thread to stop
            self.app_stop_events[app_id].set()
            
            # Wait for thread to finish
            thread = self.app_threads.get(app_id)
            if thread and thread.is_alive():
                thread.join(timeout=5.0)
                if thread.is_alive():
                    logging.warning(f"Application {app_id} did not stop gracefully")
                    return False
            
            # Clean up
            app = self.apps[app_id]
            app._cleanup_resources()
            
            # Remove from button handler
            if self.button_handler:
                self.button_handler.remove_app_instance(app_id)
            
            # Remove from tracking
            del self.apps[app_id]
            del self.app_threads[app_id]
            del self.app_stop_events[app_id]
            
            logging.info(f"Application {app_id} stopped successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error stopping application {app_id}: {e}")
            return False
    
    def restart_app(self, app_id: str) -> bool:
        """Restart a specific application"""
        if app_id not in APPLICATIONS:
            logging.error(f"Application {app_id} not found in configuration")
            return False
        
        try:
            logging.info(f"Restarting application: {app_id}")
            
            # Stop the app
            if app_id in self.apps:
                self.stop_app(app_id)
            
            # Reinitialize and start
            app_config = APPLICATIONS[app_id]
            if self._initialize_single_app(app_id, app_config):
                return self._start_single_app(app_id, self.apps[app_id])
            else:
                return False
                
        except Exception as e:
            logging.error(f"Error restarting application {app_id}: {e}")
            return False
    
    def get_app_status(self) -> Dict[str, dict]:
        """Get status of all applications"""
        status = {}
        
        for app_id, app in self.apps.items():
            thread = self.app_threads.get(app_id)
            stop_event = self.app_stop_events.get(app_id)
            
            status[app_id] = {
                'running': thread.is_alive() if thread else False,
                'thread_name': thread.name if thread else None,
                'stop_requested': stop_event.is_set() if stop_event else False,
                'app_id': app.app_id,
                'screen_id': app.screen_id,
                'video_path': app.video_path,
                'model_path': app.model_path
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
            
            # Start all applications
            if not self.start_all_apps():
                logging.error("Failed to start applications")
                return
            
            logging.info("Multi-application system started successfully")
            
            # Main loop - wait for shutdown signal
            while not self.shutdown_event.is_set():
                try:
                    # Check if any apps have stopped unexpectedly
                    for app_id, thread in self.app_threads.items():
                        if not thread.is_alive() and self.is_running:
                            logging.warning(f"Application {app_id} stopped unexpectedly")
                            # Could implement auto-restart here if desired
                    
                    # Sleep briefly
                    time.sleep(1.0)
                    
                except KeyboardInterrupt:
                    logging.info("Keyboard interrupt received")
                    break
                except Exception as e:
                    logging.error(f"Error in main loop: {e}")
                    break
            
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
