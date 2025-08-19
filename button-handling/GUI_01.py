# gui_app.py
import tkinter as tk
from tkinter import ttk
import controller_01 as controller

NUM_CHANNELS   = 48
SLIDER_COLS    = 12
SLIDER_ROWS    = NUM_CHANNELS // SLIDER_COLS
QUAD_COLS      = 8
QUAD_ROWS      = NUM_CHANNELS // QUAD_COLS  
UPDATE_MS      = 20

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("48‑Slider PWM & Button Monitor")
        self.configure(padx=10, pady=10)

        # Slider panel
        slider_frame = ttk.LabelFrame(self, text="PWM Sliders (0–255)")
        slider_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.slider_vars = []
        for i in range(NUM_CHANNELS):
            row = i // SLIDER_COLS
            col = i % SLIDER_COLS
            cell = ttk.Frame(slider_frame, padding=2)
            cell.grid(row=row*2, column=col, sticky="n")
            var = tk.IntVar(value=0)
            lbl = ttk.Label(cell, text=f"{i}: 0")
            lbl.pack()
            s = ttk.Scale(
                cell, from_=0, to=255, orient="vertical",
                variable=var,
                command=lambda v, idx=i, l=lbl, var=var: self.on_slide(idx, v, l, var)
            )
            s.pack()
            self.slider_vars.append(var)

        # Button‑state quad panel
        quad_frame = ttk.LabelFrame(self, text="Button States")
        quad_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.quads = []
        for i in range(NUM_CHANNELS):
            row = i // QUAD_COLS
            col = i % QUAD_COLS
            cell = ttk.Frame(quad_frame, padding=2)
            cell.grid(row=row*2, column=col, sticky="n")
            c = tk.Canvas(cell, width=30, height=30, bd=1, relief="solid")
            rect = c.create_rectangle(2,2,28,28, fill="red")
            c.pack()
            lbl = ttk.Label(cell, text=str(i))
            lbl.pack()
            self.quads.append((c, rect))

        # Kick off periodic updates
        self.after(UPDATE_MS, self.update_loop)

    def on_slide(self, idx, val, label, var):
        v = int(float(val))
        var.set(v)
        label.config(text=f"{idx}: {v}")
        # Update the shared array
        controller.led_values[idx] = v

    def update_loop(self):
        # Reflect incoming button_states
        for i, (canvas, rect) in enumerate(self.quads):
            color = "green" if controller.button_states[i] else "red"
            canvas.itemconfig(rect, fill=color)
        self.after(UPDATE_MS, self.update_loop)

if __name__ == "__main__":
    # 1) Start the serial I/O controller
    controller.start()
    # 2) Run the GUI
    app = App()
    app.mainloop()
