# basic_controller.py
import threading
import serial
import time

# — Configuration — 
WRITE_PORT   = '/dev/ttyUSB0'    # Port to send PWM values (Nano)s
READ_PORT    = '/dev/ttyACM0'     # Port to read button states (Mega)
BAUD_RATE    = 115200
NUM_CHANNELS = 48
FPS          = 50
INTERVAL     = 1.0 / FPS   # seconds

# — Public arrays —
led_values     = [0] * NUM_CHANNELS   # 0–255 values to send
button_states  = [0] * NUM_CHANNELS   # 0/1 states read back

def send_loop():
    """Continuously sends out_values as CSV over WRITE_PORT."""
    try:
        ser = serial.Serial(WRITE_PORT, BAUD_RATE, timeout=0)
        time.sleep(2)  # allow Arduino reset
        ser.reset_input_buffer()
    except Exception as e:
        print("❌ Write-port error:", e)
        return

    while True:
        line = ",".join(str(v) for v in led_values) + "\n"
        ser.write(line.encode('ascii'))
        print("▶️ Sent:   ", line.strip())
        time.sleep(INTERVAL)

def recv_loop():
    """Continuously reads button_states CSV from READ_PORT."""
    try:
        ser = serial.Serial(READ_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(2)
        ser.reset_input_buffer()
    except Exception as e:
        print("❌ Read-port error:", e)
        return

    while True:
        raw = ser.readline().decode('ascii', errors='replace').strip()
        parts = raw.split(',')
        if len(parts) == NUM_CHANNELS:
            for i,p in enumerate(parts):
                button_states[i] = 1 if p == '1' else 0
            print("📥 Recv:   ", ",".join(str(s) for s in button_states))

def start():
    """Start background threads for send/receive."""
    threading.Thread(target=send_loop,  daemon=True).start()
    threading.Thread(target=recv_loop,  daemon=True).start()

if __name__ == "__main__":
    start()
    # Keep main alive
    while True:
        time.sleep(1)
