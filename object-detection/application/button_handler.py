"""
Button Handler Module for Arduino Mega Integration
Handles button inputs from the 48-button Arduino Mega and maps them to keyboard actions
in the detection application. Now supports multiple applications.
"""

import threading
import time
import logging
import serial
from typing import Optional
from config.config import BUTTON_MAPPING, BUTTON_CONFIG, BUTTON_ACTIONS, APPLICATIONS, LED_CONFIG, LED_BUTTON_MAPPING
from .led_controller import LEDController


class ButtonHandler:
    """
    Handles button inputs from Arduino Mega and maps them to application actions.
    Integrates with the existing keyboard handling system and supports multiple applications.
    """
    
    def __init__(self, app_instances=None):
        # Support both single app instance (backward compatibility) and multiple instances
        if app_instances is None:
            self.app_instances = {}
        elif isinstance(app_instances, dict):
            self.app_instances = app_instances
        else:
            # Single app instance for backward compatibility
            self.app_instances = {'default': app_instances}
        
        self.button_states = [False] * 48
        
        # Serial connection
        self.serial_connection = None
        self.serial_thread = None
        self.is_running = False
        
        # Action callbacks
        self.action_callbacks = {}
        self._setup_action_callbacks()
        
        # Configuration
        self.config = BUTTON_CONFIG.copy()
        
        # Button debouncing
        self.last_button_press_time = [0.0] * 48
        self.debounce_delay = self.config.get('debounce_time', 100) / 1000.0  # Convert ms to seconds
        
        # Connection monitoring
        self.last_heartbeat = time.time()
        self.heartbeat_interval = 5.0  # Check connection every 5 seconds
        self.connection_errors = 0
        self.max_connection_errors = 3
        
        # LED Controller
        self.led_controller = None
        if LED_CONFIG.get('enabled', False):
            try:
                self.led_controller = LEDController(
                    serial_port=LED_CONFIG.get('serial_port', '/dev/ttyUSB0'),
                    baud_rate=LED_CONFIG.get('baud_rate', 115200)
                )
                # Set default brightness for all LEDs
                default_brightness = LED_CONFIG.get('default_brightness', 50)
                self.led_controller.set_all_leds(default_brightness)
                logging.info("LED Controller initialized successfully")
            except Exception as e:
                logging.warning(f"Failed to initialize LED Controller: {e}")
                self.led_controller = None
        
        logging.info("ButtonHandler initialized with multi-application support")
    
    def _setup_action_callbacks(self):
        """Setup action callback functions for each button action"""
        self.action_callbacks = {
            'exit': self._action_exit,
            'next_video': self._action_next_video,
            'previous_video': self._action_previous_video,
            'restart_video': self._action_restart_video,
            'pause_resume': self._action_pause_resume,
            'fast_forward': self._action_fast_forward,
            'rewind': self._action_rewind,
            'home': self._action_home,
            'toggle_legend': self._action_toggle_legend,
            'toggle_fps': self._action_toggle_fps,
            'toggle_gradcam': self._action_toggle_gradcam,
            'toggle_gradcam_box': self._action_toggle_gradcam_box,
            'toggle_glitches': self._action_toggle_glitches,
            'toggle_center_display': self._action_toggle_center_display,
            'toggle_visualization': self._action_toggle_visualization,
            'toggle_debug': self._action_toggle_debug,
            'toggle_unmask': self._action_toggle_unmask,
            'threshold_0.1': lambda: self._action_set_threshold(0.1),
            'threshold_0.2': lambda: self._action_set_threshold(0.2),
            'threshold_0.3': lambda: self._action_set_threshold(0.3),
            'threshold_0.4': lambda: self._action_set_threshold(0.4),
            'threshold_0.5': lambda: self._action_set_threshold(0.5),
            'threshold_0.6': lambda: self._action_set_threshold(0.6),
            'threshold_0.7': lambda: self._action_set_threshold(0.7),
            'threshold_0.8': lambda: self._action_set_threshold(0.8),
            'threshold_0.9': lambda: self._action_set_threshold(0.9),
            'threshold_1.0': lambda: self._action_set_threshold(1.0),
            'increase_threshold': self._action_increase_threshold,
            'decrease_threshold': self._action_decrease_threshold,
            'toggle_memory_cleanup': self._action_toggle_memory_cleanup,
            'toggle_audio': self._action_toggle_audio,
            'toggle_training_mode': self._action_toggle_training_mode,
            'toggle_output': self._action_toggle_output,
            'reset_display': self._action_reset_display,
            'export_results': self._action_export_results,
            'save_screenshot': self._action_save_screenshot,
            'toggle_ui': self._action_toggle_ui,
            'jump_frame_forward': self._action_jump_frame_forward,
            'jump_frame_backward': self._action_jump_frame_backward,
            'toggle_edge_detection': self._action_toggle_edge_detection,
            'cycle_view_mode': self._action_cycle_view_mode,
            'confirm_action': self._action_confirm_action,
            'undo_action': self._action_undo_action,
            'clear_detections': self._action_clear_detections,
            'insert_marker': self._action_insert_marker,
            'go_to_start': self._action_go_to_start,
            'go_to_end': self._action_go_to_end,
            'emergency_stop': self._action_emergency_stop,
        }
    
    def add_app_instance(self, app_id: str, app_instance) -> None:
        """Add an application instance to the button handler"""
        self.app_instances[app_id] = app_instance
        logging.info(f"Added app instance '{app_id}' to ButtonHandler")
    
    def remove_app_instance(self, app_id: str) -> None:
        """Remove an application instance from the button handler"""
        if app_id in self.app_instances:
            del self.app_instances[app_id]
            logging.info(f"Removed app instance '{app_id}' from ButtonHandler")
    
    def get_app_instance(self, app_id: str = None):
        """Get an application instance. If app_id is None, return the first available instance for backward compatibility"""
        if app_id is None:
            # Backward compatibility: return first available instance
            return next(iter(self.app_instances.values())) if self.app_instances else None
        return self.app_instances.get(app_id)
    
    def get_all_app_instances(self) -> dict:
        """Get all application instances"""
        return self.app_instances.copy()
    
    def _get_target_apps(self, button_id: int) -> list:
        """Get target applications for a button press based on app_id in button mapping"""
        if button_id not in BUTTON_MAPPING:
            return []
        
        action_info = BUTTON_MAPPING[button_id]
        app_id = action_info.get('app_id', 'all')
        
        if app_id == 'all':
            # Return all enabled applications
            return list(self.app_instances.values())
        elif app_id in self.app_instances:
            # Return specific application
            return [self.app_instances[app_id]]
        else:
            # App not found, return empty list
            logging.warning(f"Application '{app_id}' not found for button {button_id}")
            return []
    
    def check_serial_port_availability(self) -> bool:
        """Check if the configured serial port is available"""
        try:
            import os
            port_path = self.config.get('serial_port', '/dev/ttyACM0')
            
            if not os.path.exists(port_path):
                logging.error(f"Serial port {port_path} does not exist")
                return False
            
            # Try to open the port briefly to test access
            test_ser = serial.Serial(
                port_path,
                self.config.get('baud_rate', 115200),
                timeout=0.1
            )
            test_ser.close()
            logging.info(f"Serial port {port_path} is available and accessible")
            return True
            
        except (serial.SerialException, OSError) as e:
            logging.error(f"Serial port {port_path} is not accessible: {e}")
            return False
        except Exception as e:
            logging.error(f"Error checking serial port availability: {e}")
            return False
    
    def start_serial_monitoring(self):
        """Start monitoring serial port for button inputs"""
        try:
            # Check if already running
            if self.is_running:
                logging.debug("Button monitoring already running")
                return True
            
            # Check port availability first
            if not self.check_serial_port_availability():
                logging.error("Cannot start button monitoring - serial port unavailable")
                return False
            
            self.is_running = True
            self.serial_thread = threading.Thread(
                target=self._serial_monitor_loop,
                daemon=True
            )
            self.serial_thread.start()
            
            # Start LED controller if available
            if self.led_controller:
                self.led_controller.start_updates()
                logging.info("LED controller started")
            
            logging.info(f"Started button monitoring on {self.config['serial_port']}")
            return True
            
        except ImportError:
            logging.error("pyserial not installed. Please install it: pip install pyserial")
            return False
        except Exception as e:
            logging.error(f"Failed to start button monitoring: {e}")
            return False
    
    def _serial_monitor_loop(self):
        """Main loop for monitoring serial port - simplified like mega_48_buttons_read.py"""
        
        while self.is_running:
            try:
                # Create persistent serial connection
                ser = serial.Serial(
                    self.config['serial_port'], 
                    self.config['baud_rate'], 
                    timeout=0.1
                )
                logging.info(f"Serial connection established on {self.config['serial_port']}")
                self.connection_errors = 0  # Reset error counter on successful connection
                
                # Monitor serial data with persistent connection
                while self.is_running and ser.is_open:
                    try:
                        # Check heartbeat
                        current_time = time.time()
                        if current_time - self.last_heartbeat >= self.heartbeat_interval:
                            self._check_connection_health(ser)
                            self.last_heartbeat = current_time
                        
                        # Use non-blocking read with timeout
                        if ser.in_waiting > 0:
                            line = ser.readline().decode('ascii', errors='replace').strip()
                            if line:
                                self._process_serial_line(line)
                        else:
                            # Small sleep to prevent busy waiting
                            time.sleep(0.01)
                            
                    except (serial.SerialException, OSError) as e:
                        logging.error(f"Serial read error: {e}")
                        self.connection_errors += 1
                        break
                    except Exception as e:
                        logging.error(f"Unexpected error in serial read: {e}")
                        self.connection_errors += 1
                        break
                
                # Close connection before retrying
                if ser.is_open:
                    ser.close()
                    
            except (serial.SerialException, OSError) as e:
                logging.error(f"Serial connection error: {e}")
                self.connection_errors += 1
                time.sleep(1.0)  # Wait before retrying
            except Exception as e:
                logging.error(f"Unexpected error in serial monitoring: {e}")
                self.connection_errors += 1
                time.sleep(1.0)  # Wait before retrying
            
            # Check if we should force a restart after too many errors
            if self.connection_errors >= self.max_connection_errors:
                logging.warning(f"Too many connection errors ({self.connection_errors}), forcing restart...")
                time.sleep(2.0)  # Wait longer before restart
                self.connection_errors = 0  # Reset counter
    
    def _process_serial_line(self, line: str):
        """Process incoming serial data and update button states - simplified like mega_48_buttons_read.py"""
        parts = line.split(',')
        if len(parts) == 48:
            # Update all button states at once like the working version
            for i, part in enumerate(parts):
                button_pressed = part == '1'
                # Only handle button press if state changed from False to True and debounce time has passed
                if button_pressed and not self.button_states[i]:
                    current_time = time.time()
                    if current_time - self.last_button_press_time[i] >= self.debounce_delay:
                        self._handle_button_press(i)
                        self.last_button_press_time[i] = current_time
                self.button_states[i] = button_pressed
    
    def _handle_button_press(self, button_id: int):
        """Handle button press event with debouncing and multi-application support"""
        if button_id in BUTTON_MAPPING:
            action_info = BUTTON_MAPPING[button_id]
            action_name = action_info['action']
            key = action_info['key']
            app_id = action_info.get('app_id', 'all')
            
            logging.info(f"Button {button_id} pressed: {action_name} (key: {key}, app_id: {app_id})")
            
            # Get target applications for this button
            target_apps = self._get_target_apps(button_id)
            
            if not target_apps:
                logging.warning(f"No target applications found for button {button_id}")
                return
            
            # Execute action for all target applications
            for app_instance in target_apps:
                try:
                    if action_name in self.action_callbacks:
                        # Temporarily set the app instance for the callback
                        old_app_instance = getattr(self, '_current_app_instance', None)
                        self._current_app_instance = app_instance
                        
                        self.action_callbacks[action_name]()
                        
                        # Restore previous app instance
                        self._current_app_instance = old_app_instance
                        
                        if self.config['enable_sound_feedback']:
                            self._play_button_sound(button_id)
                        if self.config['enable_visual_feedback']:
                            self._show_button_feedback(button_id)
                        
                        print(f"button_id: {button_id}")
                        # LED feedback for button press
                        if self.led_controller:
                            self._handle_led_feedback(button_id)
                    else:
                        logging.warning(f"Button action '{action_name}' has no callback implementation")
                except Exception as e:
                    logging.error(f"Error executing button action {action_name} for app instance: {e}")
                    # Don't let button action errors stop the serial monitoring
                    import traceback
                    logging.debug(f"Button action traceback: {traceback.format_exc()}")
        else:
            logging.debug(f"Button {button_id} pressed but not mapped to any action")
    
    def _play_button_sound(self, button_id: int):
        """Play sound feedback for button press"""
        # TODO: Implement sound feedback
        pass
    
    def _show_button_feedback(self, button_id: int):
        """Show visual feedback for button press"""
        # TODO: Implement visual feedback
        pass
    
    def _handle_led_feedback(self, button_id: int):
        """Handle LED feedback when a button is pressed"""
        if not self.led_controller:
            return
        
        # Check if this button has an LED mapping
        if button_id not in LED_BUTTON_MAPPING:
            logging.debug(f"No LED mapping for button {button_id}")
            return
        
        # Get the LED index for this button
        led_index = LED_BUTTON_MAPPING[button_id]
        
        print(f"button_id: {button_id}, led_index: {led_index}")

        try:
            feedback_type = LED_CONFIG.get('feedback_type', 'brightness')
            button_press_brightness = LED_CONFIG.get('button_press_brightness', 255)
            fade_duration = LED_CONFIG.get('fade_duration_ms', 100)
            auto_dim = LED_CONFIG.get('auto_dim', True)
            dim_delay = LED_CONFIG.get('dim_delay_ms', 300)
            dim_to_brightness = LED_CONFIG.get('dim_to_brightness', 30)
            
            # Get current LED brightness
            #current_brightness = self.led_controller.led_values[led_index]
            
            if feedback_type == 'brightness':
                # Set LED to maximum brightness for button press
                self.led_controller.set_led_brightness(led_index, button_press_brightness)
                
                # Auto-dim after delay if enabled
                if auto_dim:
                    def restore_brightness():
                        time.sleep(dim_delay / 1000.0)
                        # Fade back to dim brightness
                        self.led_controller.fade_led(led_index, button_press_brightness, dim_to_brightness, fade_duration)
                    
                    threading.Thread(target=restore_brightness, daemon=True).start()
                    
            elif feedback_type == 'pulse':
                # Quick pulse effect
                self.led_controller.button_press_feedback(led_index, 'pulse')
                
            elif feedback_type == 'fade':
                # Smooth fade effect
                self.led_controller.button_press_feedback(led_index, 'fade')
            
            logging.debug(f"LED feedback applied for button {button_id} -> LED {led_index}: {feedback_type}")
            
        except Exception as e:
            logging.warning(f"LED feedback failed for button {button_id} -> LED {led_index}: {e}")
    
    # Action callback implementations - now support multiple applications
    def _action_exit(self):
        """Exit the application"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance:
            logging.info("Exit requested via button")
            # Signal exit through app instance
    
    def _action_next_video(self):
        """Move to next video"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance:
            logging.info("Next video requested via button")
            # Signal next video through app instance
    
    def _action_previous_video(self):
        """Move to previous video"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance:
            logging.info("Previous video requested via button")
            # Signal previous video through app instance
    
    def _action_restart_video(self):
        """Restart current video"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance:
            logging.info("Restart video requested via button")
            # Signal restart through app instance
    
    def _action_pause_resume(self):
        """Toggle pause/resume"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance:
            logging.info("Pause/Resume toggled via button")
            # Signal pause/resume through app instance
    
    def _action_fast_forward(self):
        """Fast forward video"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance:
            logging.info("Fast forward requested via button")
            # Signal fast forward through app instance
    
    def _action_rewind(self):
        """Rewind video"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance:
            logging.info("Rewind requested via button")
            # Signal rewind through app instance
    
    def _action_home(self):
        """Go to first video"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance:
            logging.info("Home requested via button")
            # Signal home through app instance
    
    def _action_toggle_legend(self):
        """Toggle legend display"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance:
            app_instance.display_config.show_legend = not app_instance.display_config.show_legend
            logging.info(f"Legend display toggled via button: {app_instance.display_config.show_legend}")
    
    def _action_toggle_fps(self):
        """Toggle FPS display"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance:
            app_instance.display_config.show_fps_info = not app_instance.display_config.show_fps_info
            logging.info(f"FPS display toggled via button: {app_instance.display_config.show_fps_info}")
    
    def _action_toggle_gradcam(self):
        """Toggle Grad-CAM view"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance:
            app_instance.display_config.gradcam_enabled = not app_instance.display_config.gradcam_enabled
            logging.info(f"Grad-CAM toggled via button: {app_instance.display_config.gradcam_enabled}")
    
    def _action_toggle_gradcam_box(self):
        """Toggle Grad-CAM box mode"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance:
            app_instance.display_config.gradcam_in_box_only = not app_instance.display_config.gradcam_in_box_only
            logging.info(f"Grad-CAM box mode toggled via button: {app_instance.display_config.gradcam_in_box_only}")
    
    def _action_toggle_glitches(self):
        """Toggle glitch effects"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance:
            app_instance.display_config.enable_glitches = not app_instance.display_config.enable_glitches
            logging.info(f"Glitches toggled via button: {app_instance.display_config.enable_glitches}")
    
    def _action_toggle_center_display(self):
        """Toggle center display mode"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance:
            # TODO: Implement center display toggle
            logging.info("Center display toggle requested via button")
    
    def _action_toggle_visualization(self):
        """Toggle visualization mode"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance:
            # TODO: Implement visualization toggle
            logging.info("Visualization toggle requested via button")
    
    def _action_toggle_debug(self):
        """Toggle debug information"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance:
            # TODO: Implement debug toggle
            logging.info("Debug toggle requested via button")
    
    def _action_toggle_unmask(self):
        """Toggle unmask/blur mode"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance and hasattr(app_instance, 'visualizer'):
            app_instance.visualizer.handle_key_press(32, app_instance)  # Space key
            logging.info("Unmask/blur toggled via button")
    
    def _action_set_threshold(self, threshold: float):
        """Set confidence threshold"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance:
            app_instance.box_threshold = threshold
            logging.info(f"Confidence threshold set to {threshold} via button")
    
    def _action_increase_threshold(self):
        """Increase confidence threshold"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance:
            new_threshold = min(1.0, app_instance.box_threshold + 0.1)
            app_instance.box_threshold = new_threshold
            logging.info(f"Confidence threshold increased to {new_threshold} via button")
    
    def _action_decrease_threshold(self):
        """Decrease confidence threshold"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance:
            new_threshold = max(0.1, app_instance.box_threshold - 0.1)
            app_instance.box_threshold = new_threshold
            logging.info(f"Confidence threshold decreased to {new_threshold} via button")
    
    def _action_toggle_memory_cleanup(self):
        """Toggle memory cleanup"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance and hasattr(app_instance, 'memory_manager'):
            # TODO: Implement memory cleanup toggle
            logging.info("Memory cleanup toggle requested via button")
    
    def _action_toggle_audio(self):
        """Toggle audio effects"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance and hasattr(app_instance, 'visualizer'):
            # TODO: Implement audio toggle
            logging.info("Audio toggle requested via button")
    
    def _action_toggle_training_mode(self):
        """Toggle training mode"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance:
            # TODO: Implement training mode toggle
            logging.info("Training mode toggle requested via button")
    
    def _action_toggle_output(self):
        """Toggle output recording"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance:
            # TODO: Implement output toggle
            logging.info("Output toggle requested via button")
    
    def _action_reset_display(self):
        """Reset display settings"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance:
            # TODO: Implement display reset
            logging.info("Display reset requested via button")
    
    def _action_export_results(self):
        """Export detection results"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance:
            # TODO: Implement results export
            logging.info("Results export requested via button")
    
    def _action_save_screenshot(self):
        """Save current frame"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance:
            # TODO: Implement screenshot save
            logging.info("Screenshot save requested via button")
    
    def _action_toggle_ui(self):
        """Toggle UI elements"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance:
            # TODO: Implement UI toggle
            logging.info("UI toggle requested via button")
    
    def _action_jump_frame_forward(self):
        """Jump 10 frames forward"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance:
            # TODO: Implement frame jump forward
            logging.info("Frame jump forward requested via button")
    
    def _action_jump_frame_backward(self):
        """Jump 10 frames backward"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance:
            # TODO: Implement frame jump backward
            logging.info("Frame jump backward requested via button")
    
    def _action_toggle_edge_detection(self):
        """Toggle edge detection"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance:
            # TODO: Implement edge detection toggle
            logging.info("Edge detection toggle requested via button")
    
    def _action_cycle_view_mode(self):
        """Cycle through view modes"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance:
            # TODO: Implement view mode cycling
            logging.info("View mode cycle requested via button")
    
    def _action_confirm_action(self):
        """Confirm current action"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance:
            # TODO: Implement action confirmation
            logging.info("Action confirmation requested via button")
    
    def _action_undo_action(self):
        """Undo last action"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance:
            # TODO: Implement action undo
            logging.info("Action undo requested via button")
    
    def _action_clear_detections(self):
        """Clear all detections"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance:
            # TODO: Implement detections clear
            logging.info("Detections clear requested via button")
    
    def _action_insert_marker(self):
        """Insert frame marker"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance:
            # TODO: Implement frame marker insertion
            logging.info("Frame marker insertion requested via button")
    
    def _action_go_to_start(self):
        """Go to video start"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance:
            # TODO: Implement go to start
            logging.info("Go to start requested via button")
    
    def _action_go_to_end(self):
        """Go to video end"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance:
            # TODO: Implement go to end
            logging.info("Go to end requested via button")
    
    def _action_emergency_stop(self):
        """Emergency stop"""
        app_instance = getattr(self, '_current_app_instance', None)
        if app_instance:
            logging.warning("Emergency stop requested via button")
            # Signal emergency stop through app instance
    
    def stop(self):
        """Stop the button handler and clean up resources"""
        logging.info("Stopping button handler...")
        self.is_running = False
        
        # Stop LED controller if available
        if self.led_controller:
            self.led_controller.stop_updates()
            logging.info("LED controller stopped")
        
        # Wait for serial thread to finish
        if self.serial_thread and self.serial_thread.is_alive():
            self.serial_thread.join(timeout=2.0)
            if self.serial_thread.is_alive():
                logging.warning("Serial thread did not stop gracefully")
        
        # Close serial connection if open
        if hasattr(self, 'serial_connection') and self.serial_connection:
            try:
                if self.serial_connection.is_open:
                    self.serial_connection.close()
            except Exception as e:
                logging.warning(f"Error closing serial connection: {e}")
        
        logging.info("Button handler stopped")
    
    def restart(self):
        """Restart button monitoring"""
        logging.info("Restarting button monitoring...")
        self.stop()
        time.sleep(0.1)  # Brief pause to ensure clean shutdown
        return self.start_serial_monitoring()
    
    def is_connected(self) -> bool:
        """Check if serial connection is active"""
        return self.is_running and self.serial_thread and self.serial_thread.is_alive()
    
    def get_connection_status(self) -> dict:
        """Get detailed connection status"""
        status = {
            'is_running': self.is_running,
            'thread_alive': self.serial_thread.is_alive() if self.serial_thread else False,
            'serial_port': self.config.get('serial_port', 'Unknown'),
            'baud_rate': self.config.get('baud_rate', 'Unknown'),
            'app_instances': list(self.app_instances.keys())
        }
        
        # Add LED controller status if available
        if self.led_controller:
            status['led_controller'] = self.led_controller.get_connection_status()
            status['led_mapping'] = {
                'enabled': True,
                'mapped_buttons': len(LED_BUTTON_MAPPING),
                'total_buttons': 48,
                'mapping': LED_BUTTON_MAPPING
            }
        else:
            status['led_controller'] = {'enabled': False}
            status['led_mapping'] = {'enabled': False}
        
        return status
    
    def get_button_state(self, button_id: int) -> bool:
        """Get current state of a button"""
        if 0 <= button_id < 48:
            return self.button_states[button_id]
        return False
    
    def get_all_button_states(self) -> list:
        """Get states of all buttons"""
        return self.button_states.copy()
    
    def get_button_info(self, button_id: int) -> dict:
        """Get detailed information about a button including last press time"""
        if 0 <= button_id < 48:
            return {
                'state': self.button_states[button_id],
                'last_press_time': self.last_button_press_time[button_id],
                'mapped_action': BUTTON_MAPPING.get(button_id, 'No mapping'),
                'debounce_delay': self.debounce_delay
            }
        return {}
    
    def reset_button_states(self):
        """Reset all button states and press times (useful for debugging)"""
        self.button_states = [False] * 48
        self.last_button_press_time = [0.0] * 48
        logging.info("Button states and press times reset")
    
    def set_app_instance(self, app_instance):
        """Set the application instance for action callbacks (backward compatibility)"""
        # For backward compatibility, add as 'default' instance
        self.app_instances['default'] = app_instance
        logging.info("App instance set for ButtonHandler (backward compatibility mode)")
    
    def set_debounce_delay(self, delay: float):
        """Set the debounce delay for button presses (in seconds)"""
        self.debounce_delay = max(0.01, delay)  # Minimum 10ms delay
        logging.info(f"Button debounce delay set to {self.debounce_delay}s")
    
    def get_debounce_delay(self) -> float:
        """Get the current debounce delay"""
        return self.debounce_delay
    
    def update_config(self, new_config: dict):
        """Update configuration and recalculate debounce delay"""
        self.config.update(new_config)
        self.debounce_delay = self.config.get('debounce_time', 100) / 1000.0
        logging.info(f"ButtonHandler config updated, debounce delay: {self.debounce_delay}s")
    
    def get_led_mapping(self, button_id: int) -> Optional[int]:
        """Get the LED index mapped to a button, or None if no mapping exists"""
        return LED_BUTTON_MAPPING.get(button_id)
    
    def get_button_mapping(self, led_id: int) -> Optional[int]:
        """Get the button index mapped to an LED, or None if no mapping exists"""
        for button_id, led_index in LED_BUTTON_MAPPING.items():
            if led_index == led_id:
                return button_id
        return None
    
    def set_led_brightness_for_button(self, button_id: int, brightness: int):
        """Set LED brightness for a specific button using the mapping"""
        if not self.led_controller:
            return False
        
        led_index = self.get_led_mapping(button_id)
        if led_index is None:
            logging.debug(f"No LED mapping for button {button_id}")
            return False
        
        try:
            self.led_controller.set_led_brightness(led_index, brightness)
            return True
        except Exception as e:
            logging.warning(f"Failed to set LED brightness for button {button_id}: {e}")
            return False
    
    def test_led_mapping(self, button_id: int):
        """Test LED mapping for a specific button by briefly lighting it up"""
        if not self.led_controller:
            return False
        
        led_index = self.get_led_mapping(button_id)
        if led_index is None:
            logging.debug(f"No LED mapping for button {button_id}")
            return False
        
        try:
            # Store current brightness
            current_brightness = self.led_controller.led_values[led_index]
            
            # Flash the LED
            self.led_controller.set_led_brightness(led_index, 255)
            
            # Restore after delay
            def restore():
                time.sleep(0.5)
                self.led_controller.set_led_brightness(led_index, current_brightness)
            
            threading.Thread(target=restore, daemon=True).start()
            logging.info(f"LED mapping test: Button {button_id} -> LED {led_index}")
            return True
            
        except Exception as e:
            logging.warning(f"LED mapping test failed for button {button_id}: {e}")
            return False

    def restart_connection(self):
        """Restart the serial connection"""
        logging.info("Restarting serial connection...")
        self.stop()
        time.sleep(0.5)  # Brief pause before restarting
        self.start_serial_monitoring()
        logging.info("Serial connection restart completed")
    
    def force_reconnect(self):
        """Force a reconnection by stopping and starting the monitoring"""
        if self.is_running:
            logging.info("Forcing serial reconnection...")
            self.restart_connection()
        else:
            logging.warning("Cannot force reconnect - monitoring is not running")

    def _check_connection_health(self, ser):
        """Check if the serial connection is still healthy"""
        try:
            if not ser.is_open:
                logging.warning("Serial connection is closed")
                return False
            
            # Try to read a small amount of data to test connection
            if ser.in_waiting > 0:
                # Data is available, connection is healthy
                return True
            else:
                # No data, but connection might still be healthy
                return True
                
        except Exception as e:
            logging.error(f"Connection health check failed: {e}")
            return False
    
    def get_running_status(self) -> bool:
        """Check if the button handler is running"""
        return self.is_running
    
    def get_connection_health(self) -> dict:
        """Get connection health status"""
        current_time = time.time()
        time_since_heartbeat = current_time - self.last_heartbeat
        
        return {
            'is_running': self.is_running,
            'serial_thread_alive': self.serial_thread.is_alive() if self.serial_thread else False,
            'time_since_heartbeat': time_since_heartbeat,
            'connection_errors': self.connection_errors,
            'max_connection_errors': self.max_connection_errors,
            'heartbeat_interval': self.heartbeat_interval
        }

    def test_button_functionality(self, button_id: int = 0):
        """Test button functionality by simulating a button press"""
        if 0 <= button_id < 48:
            logging.info(f"Testing button {button_id} functionality...")
            self._handle_button_press(button_id)
            return True
        else:
            logging.error(f"Invalid button ID: {button_id}")
            return False
    
    def debug_button_states(self):
        """Print debug information about all button states"""
        logging.info("=== Button Handler Debug Info ===")
        logging.info(f"Running: {self.is_running}")
        logging.info(f"Thread alive: {self.serial_thread.is_alive() if self.serial_thread else False}")
        logging.info(f"Connection errors: {self.connection_errors}")
        logging.info(f"Last heartbeat: {time.time() - self.last_heartbeat:.1f}s ago")
        logging.info(f"App instances: {list(self.app_instances.keys())}")
        
        # Show first few button states
        active_buttons = [i for i, state in enumerate(self.button_states) if state]
        if active_buttons:
            logging.info(f"Active buttons: {active_buttons}")
        else:
            logging.info("No active buttons")
        
        # Show button mappings
        logging.info(f"Button mappings configured: {len(BUTTON_MAPPING)}")
        for btn_id, mapping in list(BUTTON_MAPPING.items())[:5]:  # Show first 5
            logging.info(f"  Button {btn_id}: {mapping}")
        if len(BUTTON_MAPPING) > 5:
            logging.info(f"  ... and {len(BUTTON_MAPPING) - 5} more")
        
        logging.info("=== End Debug Info ===")
