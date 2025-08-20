import tkinter as tk
from tkinter import ttk, messagebox
import serial
import time

# === Configuration ===
SERIAL_PORT        = '/dev/ttyUSB0'    # your Mega’s port found in device manager on Windows or Linux equivalent, is usually detected once and stays the same.
BAUD_RATE          = 115200
UPDATE_INTERVAL_MS = 20         # send every 20 ms (~50 FPS)
NUM_SLIDERS        = 48
COLUMNS            = 12         # horizontal layout: 12 columns × 4 rows

class PWM48SliderApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("48‑Channel PWM Controller")
        self.configure(padx=10, pady=10)
        
        # --- Serial setup ---
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
        except Exception as e:
            messagebox.showerror("Serial Error", f"Cannot open {SERIAL_PORT}:\n{e}")
            self.destroy()
            return
        time.sleep(2)
        self.ser.reset_input_buffer()
        
        # --- Style sliders ---
        style = ttk.Style(self)
        style.theme_use('default')
        style.configure('TScale', troughcolor='#e0e0e0', background='#4a90e2')
        
        # --- Create sliders in horizontal grid ---
        self.slider_vars = []
        main_frame = ttk.Frame(self)
        main_frame.pack(fill='both', expand=True)
        
        for i in range(NUM_SLIDERS):
            row = i // COLUMNS
            col = i % COLUMNS
            cell = ttk.Frame(main_frame, padding=3)
            cell.grid(row=row*2, column=col, padx=2, pady=2, sticky='n')
            
            var = tk.IntVar(value=0)
            lbl = ttk.Label(cell, text=f"{i}: 0", width=6, anchor='center')
            lbl.pack()
            
            s = ttk.Scale(
                cell,
                from_=0, to=255,
                orient='vertical',
                length=120,
                variable=var,
                command=lambda val, idx=i, l=lbl, v=var: self.on_slide(idx, val, l, v)
            )
            s.pack()
            self.slider_vars.append(var)
        
        # --- Start periodic send ---
        self.after(UPDATE_INTERVAL_MS, self.send_values)
    
    def on_slide(self, idx, val, label, var):
        v = int(float(val))
        label.config(text=f"{idx}: {v}")
        var.set(v)
    
    def send_values(self):
        """Pack and send CSV: 48 slider ints only."""
        parts = [str(v.get()) for v in self.slider_vars]
        line = ",".join(parts) + "\n"
        try:
            self.ser.write(line.encode('ascii'))
        except Exception as e:
            print("Serial write error:", e)
        self.after(UPDATE_INTERVAL_MS, self.send_values)

if __name__ == "__main__":
    # requires: pip install pyserial
    app = PWM48SliderApp()
    app.mainloop()
