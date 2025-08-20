import threading
import tkinter as tk
import serial

# === Configuration ===
SERIAL_PORT = '/dev/ttyACM0' # your Mega’s port found in device manager on Windows or Linux equivalent, is usually detected once and stays the same.
BAUD_RATE   = 115200

# Shared array for button states
button_states = [0] * 48

def serial_reader():
    """Background thread: read lines and update button_states."""
    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1) as ser:
            while True:
                line = ser.readline().decode('ascii', errors='replace').strip()
                parts = line.split(',')
                if len(parts) == 48:
                    for i, p in enumerate(parts):
                        button_states[i] = 1 if p == '1' else 0
    except Exception as e:
        print("Serial error:", e)

class ButtonMonitor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("48‑Button Monitor")
        self.configure(padx=10, pady=10)

        # Frame for 48 quads in 8×6 grid
        self.quads = []
        grid = tk.Frame(self)
        grid.pack()
        for i in range(48):
            row = i // 8
            col = i % 8
            cell = tk.Frame(grid, padx=5, pady=5)
            cell.grid(row=row*2, column=col, sticky='n')

            # Canvas square
            c = tk.Canvas(cell, width=30, height=30, bd=1, relief='solid')
            rect = c.create_rectangle(2,2,28,28, fill='red')
            c.pack()
            # Label with ID
            lbl = tk.Label(cell, text=str(i))
            lbl.pack()
            self.quads.append((c, rect))

        # Periodically refresh display
        self.after(50, self.update_quads)

    def update_quads(self):
        for i, (canvas, rect) in enumerate(self.quads):
            color = 'green' if button_states[i] else 'red'
            canvas.itemconfig(rect, fill=color)
        self.after(50, self.update_quads)

if __name__ == "__main__":
    # Ensure pyserial is installed: pip install pyserial
    # Start serial reader thread
    t = threading.Thread(target=serial_reader, daemon=True)
    t.start()

    # Launch GUI
    app = ButtonMonitor()
    app.mainloop()
