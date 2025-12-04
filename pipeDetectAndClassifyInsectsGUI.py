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
        self.yoloWeights = tk.StringVar(value="./runs/detect/insects5Motion/weights/best.pt")
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

        rowIdx = 0
        # -------------------------------
        # YOLO weights (with browse)
        # -------------------------------
        add_entry("YOLO Model", self.yoloWeights, rowIdx, browse=True, folder=False)
  
        # -------------------------------
        # Dropdowns
        # -------------------------------
        rowIdx += 1
        ttk.Label(frame, text="Taxa Model").grid(row=rowIdx, column=0, sticky="w")
        ttk.Combobox(frame, textvariable=self.modelType, 
                     values=["ResNet50", "ConvNextBase", "EfficientNetV2S"]).grid(row=rowIdx, column=1, sticky="ew")
        rowIdx += 1
        ttk.Label(frame, text="Dataset").grid(row=rowIdx, column=0, sticky="w")
        ttk.Combobox(frame, textvariable=self.dataset,
                     values=["V3", "V4", "V5", "V6"]).grid(row=rowIdx, column=1, sticky="ew")

        # -------------------------------
        # Video and images (with browse)
        # -------------------------------
        rowIdx += 1
        add_entry("Video", self.video, rowIdx, browse=True, folder=False)
        rowIdx += 1
        add_entry("Images Dir", self.images, rowIdx, browse=True, folder=True)

        rowIdx += 1
        ttk.Checkbutton(frame, text="Use Image Exif Time", variable=self.useExifTime).grid(row=rowIdx, column=1, sticky="w")

        # -------------------------------
        # Confidence
        # -------------------------------
        rowIdx += 1
        add_entry("Confidence", self.confidence, rowIdx)

        # -------------------------------
        # Confidence
        # -------------------------------
        rowIdx += 1
        add_entry("Threshold", self.thresholdStd, rowIdx)

        # -------------------------------
        # Device dropdown
        # -------------------------------
        rowIdx += 1
        ttk.Label(frame, text="Device").grid(row=rowIdx, column=0, sticky="w")
        ttk.Combobox(frame, textvariable=self.device, 
                     values=["cuda:0", "cpu"]).grid(row=rowIdx, column=1, sticky="ew")

        # -------------------------------
        # Movie Predict dropdown
        # -------------------------------
        rowIdx += 1
        ttk.Label(frame, text="Predictions").grid(row=rowIdx, column=0, sticky="w")
        ttk.Combobox(frame, textvariable=self.moviePredict, 
                     values=["movie", "CSV only"]).grid(row=rowIdx, column=1, sticky="ew")

        rowIdx += 1
        ttk.Checkbutton(frame, text="Movie MIE format", variable=self.videoMIE).grid(row=rowIdx, column=1, sticky="w")

        # -------------------------------
        # CSV format dropdown
        # -------------------------------
        rowIdx += 1
        ttk.Label(frame, text="CSV Format").grid(row=rowIdx, column=0, sticky="w")
        ttk.Combobox(frame, textvariable=self.CSVformat,
                     values=["tracking", "detection"]).grid(row=rowIdx, column=1, sticky="ew")

        # -------------------------------
        # Results dir (browse folder)
        # -------------------------------
        rowIdx += 1
        add_entry("Results Dir", self.resultsDir, rowIdx, browse=True, folder=True)

        # -------------------------------
        # Start / Stop buttons
        # -------------------------------
        rowIdx += 1
        self.start_btn = ttk.Button(frame, text="Start", command=self.start_process)
        self.start_btn.grid(row=rowIdx, column=0, pady=10)

        self.stop_btn = ttk.Button(frame, text="Stop", command=self.stop_process, state=tk.DISABLED)
        self.stop_btn.grid(row=rowIdx, column=1, pady=10)

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
            "--yoloWeights", self.yoloWeights.get(),
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

