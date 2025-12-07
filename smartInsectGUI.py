# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 22:57:34 2025

@author: Kim Bjerge
"""

import tkinter as tk
from tkinter import ttk
import subprocess
import threading
import sys


class ParallelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart AI Insect Program Launcher")

        # Track processes
        self.processes = {}

        # Build the GUI
        self.create_widgets()

    # -----------------------------------------------------------
    # GUI Elements
    # -----------------------------------------------------------
    def create_widgets(self):
        frame = ttk.Frame(self.root, padding=20)
        frame.grid(row=0, column=0)

        ttk.Button(frame, text="Insect Detector and Classifier",
                   command=lambda: self.start_program("pipeDetectAndClassifyInsectsGUI.py")).grid(row=0, column=0, padx=10, pady=10)

        ttk.Button(frame, text="Insect Tracker",
                   command=lambda: self.start_program("pipeTrackInsectsGUI.py")).grid(row=0, column=1, padx=10, pady=10)

        ttk.Button(frame, text="Insect Crop Generator",
                   command=lambda: self.start_program("createCropsGUI.py")).grid(row=1, column=0, padx=10, pady=10)

        ttk.Button(frame, text="Insect Track Plotter",
                   command=lambda: self.start_program("createTrackCropsGUI.py")).grid(row=1, column=1, padx=10, pady=10)

    # -----------------------------------------------------------
    # Start a separate Python script
    # -----------------------------------------------------------
    def start_program(self, script_name):
        # Prevent duplicate starts
        if script_name in self.processes and self.processes[script_name] is not None:
            print(f"{script_name} is already running.")
            return

        cmd = [sys.executable, script_name]

        # Start the process
        process = subprocess.Popen(cmd)
        self.processes[script_name] = process

        # Monitor process in background
        threading.Thread(target=self.monitor_process, args=(script_name, process), daemon=True).start()

    # -----------------------------------------------------------
    # Detect when a script finishes
    # -----------------------------------------------------------
    def monitor_process(self, script_name, process):
        process.wait()  # Wait until it finishes
        print(f"{script_name} finished.")
        self.processes[script_name] = None


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = ParallelApp(root)
    root.mainloop()