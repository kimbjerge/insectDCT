# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 09:46:45 2025

@author: Kim Bjerge
"""

import tkinter as tk
from tkinter import ttk
import subprocess
import threading
import sys
import webbrowser
import os


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

        # Existing buttons
        ttk.Button(frame, text="Insect Detector and Classifier",
                   command=lambda: self.start_program("pipeDetectAndClassifyInsectsGUI.py")).grid(row=0, column=0, padx=10, pady=10)

        ttk.Button(frame, text="Insect Tracker",
                   command=lambda: self.start_program("pipeTrackInsectsGUI.py")).grid(row=0, column=1, padx=10, pady=10)

        ttk.Button(frame, text="Insect Crop Generator",
                   command=lambda: self.start_program("createCropsGUI.py")).grid(row=1, column=0, padx=10, pady=10)

        ttk.Button(frame, text="Insect Track Generator",
                   command=lambda: self.start_program("createTrackCropsGUI.py")).grid(row=1, column=1, padx=10, pady=10)

        # -------------------------------------------------------
        # NEW BUTTONS → Open HTML viewers in browser
        # -------------------------------------------------------

        ttk.Button(frame, text="Open Insect Viewer in Browser",
                   command=lambda: self.open_html("insect_viewer.html")).grid(row=2, column=0, padx=10, pady=20)
        
        ttk.Button(frame, text="Open Insect Crops Viewer in Browser",
                   command=lambda: self.open_html("insect_crops_viewer.html")).grid(row=3, column=0, padx=10, pady=20)
        
        ttk.Button(frame, text="Open Track Viewer in Browser",
                   command=lambda: self.open_html("track_viewer.html")).grid(row=2, column=1, padx=10, pady=20)

        ttk.Button(frame, text="Open Track Crops Viewer in Browser",
                   command=lambda: self.open_html("track_crops_viewer.html")).grid(row=3, column=1, padx=10, pady=20)
        
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
    # Open an HTML page in the default browser
    # -----------------------------------------------------------
    def open_html(self, filename):
        # Ensure file exists in the same directory as this script
        full_path = os.path.join(os.getcwd(), filename)

        if not os.path.exists(full_path):
            print(f"HTML file not found: {full_path}")
            return

        webbrowser.open(f"file://{full_path}")

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
