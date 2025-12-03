# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 21:51:21 2025

@author: Kim Bjerge
"""
import tkinter as tk
from tkinter import ttk, filedialog
import subprocess
import threading
import sys

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Detection and Classification of Insects")

        # Make the window fully resizable
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

        self.process = None

        # -------------------------------
        # Argument variables with defaults
        # -------------------------------
        self.modelType = tk.StringVar(value="ConvNextBase")
        self.dataset = tk.StringVar(value="V6")
        self.useExifTime = tk.BooleanVar(value=False)
        self.video = tk.StringVar(value="")
        self.images = tk.StringVar(value="./images/pi1_2025_02_21/")
        self.confidence = tk.DoubleVar(value=0.401)
        self.thresholdStd = tk.DoubleVar(value=0.0)
        self.device = tk.StringVar(value="cpu")
        self.videoMIE = tk.BooleanVar(value=False)
        self.moviePredict = tk.StringVar(value="movie")
        self.CSVformat = tk.StringVar(value="tracking")
        self.resultsDir = tk.StringVar(value="./detections")

        self.create_widgets()

    # -----------------------------------------------------------
    # GUI ELEMENTS
    # -----------------------------------------------------------
    def create_widgets(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.grid(row=0, column=0, sticky="ew")
        frame.columnconfigure(1, weight=1)

        # -------------------------------
        # Helper: Create entry + label
        # -------------------------------
        def add_entry(label, variable, row, browse=False, folder=False):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w")
            entry = ttk.Entry(frame, textvariable=variable)
            entry.grid(row=row, column=1, sticky="ew", padx=5)

            if browse:
                if folder:
                    btn = ttk.Button(frame, text="Browse", command=lambda: self.browse_folder(variable))
                else:
                    btn = ttk.Button(frame, text="Browse", command=lambda: self.browse_file(variable))
                btn.grid(row=row, column=2, padx=5)

        # -------------------------------
        # Dropdowns
        # -------------------------------
        ttk.Label(frame, text="Model Type").grid(row=0, column=0, sticky="w")
        ttk.Combobox(frame, textvariable=self.modelType, 
                     values=["ResNet50", "ConvNextBase", "EfficientNetV2S"]).grid(row=0, column=1, sticky="ew")

        ttk.Label(frame, text="Dataset").grid(row=1, column=0, sticky="w")
        ttk.Combobox(frame, textvariable=self.dataset,
                     values=["V3", "V4", "V5", "V6"]).grid(row=1, column=1, sticky="ew")

        # -------------------------------
        # Video and images (with browse)
        # -------------------------------
        add_entry("Video", self.video, 2, browse=True, folder=False)
        add_entry("Images Dir", self.images, 3, browse=True, folder=True)

        ttk.Checkbutton(frame, text="Use Image Exif Time", variable=self.useExifTime).grid(row=4, column=1, sticky="w")

        # -------------------------------
        # Confidence
        # -------------------------------
        add_entry("Confidence", self.confidence, 5)

        # -------------------------------
        # Confidence
        # -------------------------------
        add_entry("Threshold", self.thresholdStd, 6)

        # -------------------------------
        # Device dropdown
        # -------------------------------
        ttk.Label(frame, text="Device").grid(row=7, column=0, sticky="w")
        ttk.Combobox(frame, textvariable=self.device, 
                     values=["cuda:0", "cpu"]).grid(row=7, column=1, sticky="ew")

        # -------------------------------
        # Movie Predict dropdown
        # -------------------------------
        ttk.Label(frame, text="Predictions").grid(row=8, column=0, sticky="w")
        ttk.Combobox(frame, textvariable=self.moviePredict, 
                     values=["movie", "CSV only"]).grid(row=8, column=1, sticky="ew")

        ttk.Checkbutton(frame, text="Movie MIE format", variable=self.videoMIE).grid(row=9, column=1, sticky="w")

        # -------------------------------
        # CSV format dropdown
        # -------------------------------
        ttk.Label(frame, text="CSV Format").grid(row=10, column=0, sticky="w")
        ttk.Combobox(frame, textvariable=self.CSVformat,
                     values=["tracking", "detection"]).grid(row=10, column=1, sticky="ew")

        # -------------------------------
        # Results dir (browse folder)
        # -------------------------------
        add_entry("Results Dir", self.resultsDir, 11, browse=True, folder=True)

        # -------------------------------
        # Start / Stop buttons
        # -------------------------------
        self.start_btn = ttk.Button(frame, text="Start", command=self.start_process)
        self.start_btn.grid(row=12, column=0, pady=10)

        self.stop_btn = ttk.Button(frame, text="Stop", command=self.stop_process, state=tk.DISABLED)
        self.stop_btn.grid(row=12, column=1, pady=10)

        # -------------------------------
        # Console Output Frame (resizeable)
        # -------------------------------
        console_frame = ttk.LabelFrame(self.root, text="Console Output")
        console_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        console_frame.rowconfigure(0, weight=1)
        console_frame.columnconfigure(0, weight=1)

        self.console = tk.Text(console_frame, wrap="word", bg="black", fg="white")
        self.console.grid(row=0, column=0, sticky="nsew")

        # Scrollbar for console
        scrollbar = ttk.Scrollbar(console_frame, orient="vertical", command=self.console.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.console["yscrollcommand"] = scrollbar.set

    # -----------------------------------------------------------
    # Browse functions
    # -----------------------------------------------------------
    def browse_folder(self, variable):
        path = filedialog.askdirectory()
        if path:
            variable.set(path)

    def browse_file(self, variable):
        path = filedialog.askopenfilename()
        if path:
            variable.set(path)

    # -----------------------------------------------------------
    # Start subprocess
    # -----------------------------------------------------------
    def start_process(self):
        if self.process is not None:
            return

        cmd = [
            sys.executable, "pipeDetectAndClassifyInsectsTaxon.py",
            "--modelType", self.modelType.get(),
            "--dataset", self.dataset.get(),
            "--useExifTime", "True" if self.useExifTime.get() else "",
            "--thresholdStd", str(self.thresholdStd.get()),
            "--video", self.video.get(),
            "--images", self.images.get(),
            "--confidence", str(self.confidence.get()),
            "--device", self.device.get(),
            "--videoMIE", "True" if self.videoMIE.get() else "",
            "--moviePredict", self.moviePredict.get() if self.moviePredict.get() != "CSV only" else "",
            "--CSVformat", self.CSVformat.get() if self.CSVformat.get() != "detection" else "",
            "--resultsDir", self.resultsDir.get()
        ]

        self.console.insert(tk.END, "Starting process...\n")
        self.console.see(tk.END)

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        threading.Thread(target=self.read_output, daemon=True).start()

    # -----------------------------------------------------------
    # Read stdout
    # -----------------------------------------------------------
    def read_output(self):
        for line in self.process.stdout:
            self.console.insert(tk.END, line)
            self.console.see(tk.END)

        self.process = None
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    # -----------------------------------------------------------
    # Stop process
    # -----------------------------------------------------------
    def stop_process(self):
        if self.process:
            self.process.kill()
            self.console.insert(tk.END, "\nProcess stopped by user.\n")
            self.console.see(tk.END)
            self.process = None

        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
root = tk.Tk()
app = App(root)
root.mainloop()

