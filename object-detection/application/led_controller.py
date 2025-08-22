"""
LED Controller Module for Arduino Nano Integration
Controls 48 LED brightness levels via PWM over serial communication.
Based on the nano_48_led_PWM.py implementation.
"""

import serial
import time
import logging
import threading
from typing import List, Optional


class LEDController:
    """
    Controls 48 LED brightness levels via PWM over serial communication.
    Sends CSV format: "value1,value2,...,value48\n" where values are 0-255.
    """
    
    def __init__(self, serial_port: str = '/dev/ttyUSB0', baud_rate: int = 115200):
        """
        Initialize LED controller
        
        Args:
            serial_port: Serial port for Arduino Nano (default: /dev/ttyUSB0)
            baud_rate: Baud rate for serial communication (default: 115200)
        """
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.serial_connection: Optional[serial.Serial] = None
        self.is_connected = False
        
        # LED brightness values (0-255 for each of 48 LEDs)
        self.led_values = [0] * 48
        
        # Update interval for sending LED values (20ms = 50 FPS)
        self.update_interval_ms = 20
        
        # Thread for continuous LED updates
        self.update_thread: Optional[threading.Thread] = None
        self.is_running = False
        
        # Lock for thread-safe LED value updates
        self.led_lock = threading.Lock()
        
        logging.info(f"LED Controller initialized for port {serial_port}")
    
    def connect(self) -> bool:
        """
        Establish serial connection to Arduino Nano
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.serial_connection = serial.Serial(
                self.serial_port, 
                self.baud_rate, 
                timeout=0.01
            )
            time.sleep(2)  # Wait for Arduino to reset
            self.serial_connection.reset_input_buffer()
            self.is_connected = True
            logging.info(f"LED Controller connected to {self.serial_port}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to connect LED Controller to {self.serial_port}: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """Close serial connection"""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
        self.is_connected = False
        logging.info("LED Controller disconnected")
    
    def set_led_brightness(self, led_index: int, brightness: int):
        """
        Set brightness for a specific LED
        
        Args:
            led_index: LED index (0-47)
            brightness: Brightness value (0-255)
        """
        if not 0 <= led_index < 48:
            logging.warning(f"Invalid LED index: {led_index}, must be 0-47")
            return
        
        if not 0 <= brightness <= 255:
            logging.warning(f"Invalid brightness value: {brightness}, must be 0-255")
            brightness = max(0, min(255, brightness))
        
        with self.led_lock:
            self.led_values[led_index] = brightness
        
        logging.debug(f"LED {led_index} brightness set to {brightness}")
    
    def set_all_leds(self, brightness: int):
        """
        Set all LEDs to the same brightness
        
        Args:
            brightness: Brightness value (0-255)
        """
        if not 0 <= brightness <= 255:
            logging.warning(f"Invalid brightness value: {brightness}, must be 0-255")
            brightness = max(0, min(255, brightness))
        
        with self.led_lock:
            self.led_values = [brightness] * 48
        
        logging.debug(f"All LEDs set to brightness {brightness}")
    
    def fade_led(self, led_index: int, start_brightness: int, end_brightness: int, duration_ms: int):
        """
        Fade an LED from start to end brightness over specified duration
        
        Args:
            led_index: LED index (0-47)
            start_brightness: Starting brightness (0-255)
            end_brightness: Ending brightness (0-255)
            duration_ms: Fade duration in milliseconds
        """
        if not 0 <= led_index < 48:
            logging.warning(f"Invalid LED index: {led_index}, must be 0-47")
            return
        
        def fade_thread():
            steps = max(1, duration_ms // self.update_interval_ms)
            step_size = (end_brightness - start_brightness) / steps
            
            for step in range(steps + 1):
                brightness = int(start_brightness + (step * step_size))
                self.set_led_brightness(led_index, brightness)
                time.sleep(duration_ms / 1000 / steps)
        
        threading.Thread(target=fade_thread, daemon=True).start()
    
    def pulse_led(self, led_index: int, min_brightness: int = 0, max_brightness: int = 255, 
                  pulse_duration_ms: int = 1000):
        """
        Create a pulsing effect on an LED
        
        Args:
            led_index: LED index (0-47)
            min_brightness: Minimum brightness (0-255)
            max_brightness: Maximum brightness (0-255)
            pulse_duration_ms: Duration of one pulse cycle in milliseconds
        """
        if not 0 <= led_index < 48:
            logging.warning(f"Invalid LED index: {led_index}, must be 0-47")
            return
        
        def pulse_thread():
            while self.is_running:
                # Fade up
                self.fade_led(led_index, min_brightness, max_brightness, pulse_duration_ms // 2)
                time.sleep(pulse_duration_ms / 1000 / 2)
                
                # Fade down
                self.fade_led(led_index, max_brightness, min_brightness, pulse_duration_ms // 2)
                time.sleep(pulse_duration_ms / 1000 / 2)
        
        threading.Thread(target=pulse_thread, daemon=True).start()
    
    def button_press_feedback(self, button_index: int, feedback_type: str = 'brightness'):
        """
        Provide visual feedback when a button is pressed
        
        Args:
            button_index: Button index (0-47) - maps to LED index
            feedback_type: Type of feedback ('brightness', 'pulse', 'fade')
        """
        if not 0 <= button_index < 48:
            logging.warning(f"Invalid button index: {button_index}, must be 0-47")
            return
        
        if feedback_type == 'brightness':
            # Increase brightness temporarily
            current_brightness = self.led_values[button_index]
            target_brightness = min(255, current_brightness + 50)
            self.fade_led(button_index, current_brightness, target_brightness, 100)
            
            # Return to original brightness after delay
            def restore_brightness():
                time.sleep(0.2)
                self.fade_led(button_index, target_brightness, current_brightness, 100)
            
            threading.Thread(target=restore_brightness, daemon=True).start()
            
        elif feedback_type == 'pulse':
            # Quick pulse effect
            current_brightness = self.led_values[button_index]
            self.fade_led(button_index, current_brightness, 255, 50)
            
            def restore_brightness():
                time.sleep(0.1)
                self.fade_led(button_index, 255, current_brightness, 50)
            
            threading.Thread(target=restore_brightness, daemon=True).start()
            
        elif feedback_type == 'fade':
            # Smooth fade effect
            current_brightness = self.led_values[button_index]
            self.fade_led(button_index, current_brightness, 255, 200)
            
            def restore_brightness():
                time.sleep(0.3)
                self.fade_led(button_index, 255, current_brightness, 200)
            
            threading.Thread(target=restore_brightness, daemon=True).start()
    
    def start_updates(self):
        """Start continuous LED updates thread"""
        if self.is_running:
            logging.warning("LED updates already running")
            return
        
        if not self.is_connected:
            if not self.connect():
                logging.error("Cannot start LED updates: not connected")
                return
        
        self.is_running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        logging.info("LED updates started")
    
    def stop_updates(self):
        """Stop continuous LED updates thread"""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=1.0)
        logging.info("LED updates stopped")
    
    def _update_loop(self):
        """Main loop for sending LED values to Arduino"""
        while self.is_running:
            try:
                if self.is_connected and self.serial_connection and self.serial_connection.is_open:
                    # Get current LED values (thread-safe)
                    with self.led_lock:
                        led_values_copy = self.led_values.copy()
                    
                    # Create CSV line: "value1,value2,...,value48\n"
                    line = ",".join(str(v) for v in led_values_copy) + "\n"
                    
                    # Send to Arduino
                    self.serial_connection.write(line.encode('ascii'))
                    
                time.sleep(self.update_interval_ms / 1000)
                
            except Exception as e:
                logging.error(f"LED update error: {e}")
                self.is_connected = False
                time.sleep(1.0)  # Wait before retry
    
    def get_connection_status(self) -> dict:
        """Get detailed connection status"""
        return {
            'connected': self.is_connected,
            'port': self.serial_port,
            'baud_rate': self.baud_rate,
            'running': self.is_running,
            'led_values': self.led_values.copy()
        }
    
    def __del__(self):
        """Cleanup on deletion"""
        self.stop_updates()
        self.disconnect() 