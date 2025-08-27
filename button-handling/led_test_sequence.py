import serial
import time

# === Configuration ===
SERIAL_PORT        = '/dev/ttyUSB0'    # your Mega's port found in device manager on Windows or Linux equivalent
BAUD_RATE          = 115200
NUM_SLIDERS        = 48
LED_ON_TIME_MS     = 1000       # time each LED stays on (1 second)

def main():
    # --- Serial setup ---
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
        print(f"Connected to {SERIAL_PORT}")
    except Exception as e:
        print(f"Serial Error: Cannot open {SERIAL_PORT}: {e}")
        return
    
    time.sleep(2)
    ser.reset_input_buffer()
    
    try:
        print("Starting LED test sequence...")
        print("Press Ctrl+C to stop")
        print("-" * 30)
        
        while True:
            for led_index in range(NUM_SLIDERS):
                print(f"LED {led_index} ON (brightness: 255)")
                
                # Turn on current LED
                set_led_brightness(ser, led_index, 255)
                
                # Wait for LED on time
                time.sleep(LED_ON_TIME_MS / 1000.0)
                
                # Turn off current LED
                set_led_brightness(ser, led_index, 0)
                print(f"LED {led_index} OFF (brightness: 0)")
                print("-" * 30)
            
            print("Sequence complete, restarting...")
            print("=" * 50)
            time.sleep(1)  # Brief pause before restarting
            
    except KeyboardInterrupt:
        print("\nStopping test sequence...")
    finally:
        # Reset all LEDs to 0
        reset_all_leds(ser)
        ser.close()
        print("All LEDs reset to 0")
        print("Serial connection closed")

def set_led_brightness(ser, led_index, brightness):
    """Set a specific LED to a specific brightness"""
    # Create array of all zeros
    values = [0] * NUM_SLIDERS
    # Set the target LED to specified brightness
    values[led_index] = brightness
    
    # Send to Arduino
    line = ",".join(map(str, values)) + "\n"
    try:
        ser.write(line.encode('ascii'))
    except Exception as e:
        print(f"Serial write error: {e}")

def reset_all_leds(ser):
    """Reset all LEDs to 0"""
    values = [0] * NUM_SLIDERS
    line = ",".join(map(str, values)) + "\n"
    try:
        ser.write(line.encode('ascii'))
    except Exception as e:
        print(f"Serial write error: {e}")

if __name__ == "__main__":
    # requires: pip install pyserial
    main() 