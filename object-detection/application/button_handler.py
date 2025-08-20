"""
Button Handler Module for Arduino Mega Integration
Handles button inputs from the 48-button Arduino Mega and maps them to keyboard actions
in the detection application.
"""

import threading
import time
import logging
import serial
from config.config import BUTTON_MAPPING, BUTTON_CONFIG, BUTTON_ACTIONS


class ButtonHandler:
    """
    Handles button inputs from Arduino Mega and maps them to application actions.
    Integrates with the existing keyboard handling system.
    """
    
    def __init__(self, app_instance=None):
        self.app_instance = app_instance
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
        
        logging.info("ButtonHandler initialized")
    
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
    
    def start_serial_monitoring(self):
        """Start monitoring serial port for button inputs"""
        try:
            self.is_running = True
            self.serial_thread = threading.Thread(
                target=self._serial_monitor_loop,
                daemon=True
            )
            self.serial_thread.start()
            logging.info(f"Started button monitoring on {self.config['serial_port']}")
        except ImportError:
            logging.error("pyserial not installed. Please install it: pip install pyserial")
        except Exception as e:
            logging.error(f"Failed to start button monitoring: {e}")
    
    def _serial_monitor_loop(self):
        """Main loop for monitoring serial port - simplified like mega_48_buttons_read.py"""
        
        while self.is_running:
            try:
                with serial.Serial(self.config['serial_port'], self.config['baud_rate'], timeout=0.1) as ser:
                    while self.is_running:
                        line = ser.readline().decode('ascii', errors='replace').strip()
                        if line:
                            self._process_serial_line(line)
            except Exception as e:
                logging.error(f"Serial connection error: {e}")
                time.sleep(1.0)  # Wait before retrying
    
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
        """Handle button press event with debouncing"""
        if button_id in BUTTON_MAPPING:
            action_info = BUTTON_MAPPING[button_id]
            action_name = action_info['action']
            key = action_info['key']
            
            logging.info(f"Button {button_id} pressed: {action_name} (key: {key})")
            
            # Execute action if callback exists
            if action_name in self.action_callbacks:
                try:
                    self.action_callbacks[action_name]()
                    if self.config['enable_sound_feedback']:
                        self._play_button_sound(button_id)
                    if self.config['enable_visual_feedback']:
                        self._show_button_feedback(button_id)
                except Exception as e:
                    logging.error(f"Error executing button action {action_name}: {e}")
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
    
    # Action callback implementations
    def _action_exit(self):
        """Exit the application"""
        if self.app_instance:
            logging.info("Exit requested via button")
            # Signal exit through app instance
    
    def _action_next_video(self):
        """Move to next video"""
        if self.app_instance:
            logging.info("Next video requested via button")
            # Signal next video through app instance
    
    def _action_previous_video(self):
        """Move to previous video"""
        if self.app_instance:
            logging.info("Previous video requested via button")
            # Signal previous video through app instance
    
    def _action_restart_video(self):
        """Restart current video"""
        if self.app_instance:
            logging.info("Restart video requested via button")
            # Signal restart through app instance
    
    def _action_pause_resume(self):
        """Toggle pause/resume"""
        if self.app_instance:
            logging.info("Pause/Resume toggled via button")
            # Signal pause/resume through app instance
    
    def _action_fast_forward(self):
        """Fast forward video"""
        if self.app_instance:
            logging.info("Fast forward requested via button")
            # Signal fast forward through app instance
    
    def _action_rewind(self):
        """Rewind video"""
        if self.app_instance:
            logging.info("Rewind requested via button")
            # Signal rewind through app instance
    
    def _action_home(self):
        """Go to first video"""
        if self.app_instance:
            logging.info("Home requested via button")
            # Signal home through app instance
    
    def _action_toggle_legend(self):
        """Toggle legend display"""
        if self.app_instance:
            self.app_instance.display_config.show_legend = not self.app_instance.display_config.show_legend
            logging.info(f"Legend display toggled via button: {self.app_instance.display_config.show_legend}")
    
    def _action_toggle_fps(self):
        """Toggle FPS display"""
        if self.app_instance:
            self.app_instance.display_config.show_fps_info = not self.app_instance.display_config.show_fps_info
            logging.info(f"FPS display toggled via button: {self.app_instance.display_config.show_fps_info}")
    
    def _action_toggle_gradcam(self):
        """Toggle Grad-CAM view"""
        if self.app_instance:
            self.app_instance.display_config.gradcam_enabled = not self.app_instance.display_config.gradcam_enabled
            logging.info(f"Grad-CAM toggled via button: {self.app_instance.display_config.gradcam_enabled}")
    
    def _action_toggle_gradcam_box(self):
        """Toggle Grad-CAM box mode"""
        if self.app_instance:
            self.app_instance.display_config.gradcam_in_box_only = not self.app_instance.display_config.gradcam_in_box_only
            logging.info(f"Grad-CAM box mode toggled via button: {self.app_instance.display_config.gradcam_in_box_only}")
    
    def _action_toggle_glitches(self):
        """Toggle glitch effects"""
        if self.app_instance:
            self.app_instance.display_config.enable_glitches = not self.app_instance.display_config.enable_glitches
            logging.info(f"Glitches toggled via button: {self.app_instance.display_config.enable_glitches}")
    
    def _action_toggle_center_display(self):
        """Toggle center display mode"""
        if self.app_instance:
            # TODO: Implement center display toggle
            logging.info("Center display toggle requested via button")
    
    def _action_toggle_visualization(self):
        """Toggle visualization mode"""
        if self.app_instance:
            # TODO: Implement visualization toggle
            logging.info("Visualization toggle requested via button")
    
    def _action_toggle_debug(self):
        """Toggle debug information"""
        if self.app_instance:
            # TODO: Implement debug toggle
            logging.info("Debug toggle requested via button")
    
    def _action_toggle_unmask(self):
        """Toggle unmask/blur mode"""
        if self.app_instance and hasattr(self.app_instance, 'visualizer'):
            self.app_instance.visualizer.handle_key_press(32, self.app_instance)  # Space key
            logging.info("Unmask/blur toggled via button")
    
    def _action_set_threshold(self, threshold: float):
        """Set confidence threshold"""
        if self.app_instance:
            self.app_instance.box_threshold = threshold
            logging.info(f"Confidence threshold set to {threshold} via button")
    
    def _action_increase_threshold(self):
        """Increase confidence threshold"""
        if self.app_instance:
            new_threshold = min(1.0, self.app_instance.box_threshold + 0.1)
            self.app_instance.box_threshold = new_threshold
            logging.info(f"Confidence threshold increased to {new_threshold} via button")
    
    def _action_decrease_threshold(self):
        """Decrease confidence threshold"""
        if self.app_instance:
            new_threshold = max(0.1, self.app_instance.box_threshold - 0.1)
            self.app_instance.box_threshold = new_threshold
            logging.info(f"Confidence threshold decreased to {new_threshold} via button")
    
    def _action_toggle_memory_cleanup(self):
        """Toggle memory cleanup"""
        if self.app_instance and hasattr(self.app_instance, 'memory_manager'):
            # TODO: Implement memory cleanup toggle
            logging.info("Memory cleanup toggle requested via button")
    
    def _action_toggle_audio(self):
        """Toggle audio effects"""
        if self.app_instance and hasattr(self.app_instance, 'visualizer'):
            # TODO: Implement audio toggle
            logging.info("Audio toggle requested via button")
    
    def _action_toggle_training_mode(self):
        """Toggle training mode"""
        if self.app_instance:
            # TODO: Implement training mode toggle
            logging.info("Training mode toggle requested via button")
    
    def _action_toggle_output(self):
        """Toggle output recording"""
        if self.app_instance:
            # TODO: Implement output toggle
            logging.info("Output toggle requested via button")
    
    def _action_reset_display(self):
        """Reset display settings"""
        if self.app_instance:
            # TODO: Implement display reset
            logging.info("Display reset requested via button")
    
    def _action_export_results(self):
        """Export detection results"""
        if self.app_instance:
            # TODO: Implement results export
            logging.info("Results export requested via button")
    
    def _action_save_screenshot(self):
        """Save current frame"""
        if self.app_instance:
            # TODO: Implement screenshot save
            logging.info("Screenshot save requested via button")
    
    def _action_toggle_ui(self):
        """Toggle UI elements"""
        if self.app_instance:
            # TODO: Implement UI toggle
            logging.info("UI toggle requested via button")
    
    def _action_jump_frame_forward(self):
        """Jump 10 frames forward"""
        if self.app_instance:
            # TODO: Implement frame jump forward
            logging.info("Frame jump forward requested via button")
    
    def _action_jump_frame_backward(self):
        """Jump 10 frames backward"""
        if self.app_instance:
            # TODO: Implement frame jump backward
            logging.info("Frame jump backward requested via button")
    
    def _action_toggle_edge_detection(self):
        """Toggle edge detection"""
        if self.app_instance:
            # TODO: Implement edge detection toggle
            logging.info("Edge detection toggle requested via button")
    
    def _action_cycle_view_mode(self):
        """Cycle through view modes"""
        if self.app_instance:
            # TODO: Implement view mode cycling
            logging.info("View mode cycle requested via button")
    
    def _action_confirm_action(self):
        """Confirm current action"""
        if self.app_instance:
            # TODO: Implement action confirmation
            logging.info("Action confirmation requested via button")
    
    def _action_undo_action(self):
        """Undo last action"""
        if self.app_instance:
            # TODO: Implement action undo
            logging.info("Action undo requested via button")
    
    def _action_clear_detections(self):
        """Clear all detections"""
        if self.app_instance:
            # TODO: Implement detections clear
            logging.info("Detections clear requested via button")
    
    def _action_insert_marker(self):
        """Insert frame marker"""
        if self.app_instance:
            # TODO: Implement frame marker insertion
            logging.info("Frame marker insertion requested via button")
    
    def _action_go_to_start(self):
        """Go to video start"""
        if self.app_instance:
            # TODO: Implement go to start
            logging.info("Go to start requested via button")
    
    def _action_go_to_end(self):
        """Go to video end"""
        if self.app_instance:
            # TODO: Implement go to end
            logging.info("Go to end requested via button")
    
    def _action_emergency_stop(self):
        """Emergency stop"""
        if self.app_instance:
            logging.warning("Emergency stop requested via button")
            # Signal emergency stop through app instance
    
    def stop(self):
        """Stop button monitoring"""
        self.is_running = False
        if self.serial_thread and self.serial_thread.is_alive():
            self.serial_thread.join(timeout=1.0)
        logging.info("ButtonHandler stopped")
    
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
        """Set the application instance for action callbacks"""
        self.app_instance = app_instance
        logging.info("App instance set for ButtonHandler")
    
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
