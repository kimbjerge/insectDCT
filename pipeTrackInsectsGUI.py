# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 08:22:34 2025

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
        self.root.title("Insect Tracker Ver. 1.1")

        # Make GUI resizeable
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

        self.process = None

        # ---------------------------------------
        # Parameter variables (with defaults)
        # ---------------------------------------
        self.images = tk.StringVar(value="./images/")
        self.video = tk.StringVar(value="")
        self.detections = tk.StringVar(value="./detections/")
        self.tracks = tk.StringVar(value="./tracks/")
        self.dataset = tk.StringVar(value="V6")

        self.checkTaxa = tk.BooleanVar(value=False)
        self.trapFilePath = tk.BooleanVar(value=False)
        self.ignoreVegetation = tk.BooleanVar(value=True)

        self.movieTrack = tk.StringVar(value="movie")

        self.create_widgets()

    # -----------------------------------------------------------
    # Create GUI elements
    # -----------------------------------------------------------
    def create_widgets(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.grid(row=0, column=0, sticky="ew")
        frame.columnconfigure(1, weight=1)

        # Helper: label + entry + optional browse button
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
                
            #if browse:
            #    btn = ttk.Button(frame, text="Browse", command=lambda: self.browse_folder(variable))
            #    btn.grid(row=row, column=2, padx=5)

        # -----------------------
        # Paths with browse
        # -----------------------
        rowIdx = 0
        add_entry("Images Dir", self.images, rowIdx, browse=True, folder=True)
        rowIdx += 1
        add_entry("Video Dir", self.video, rowIdx, browse=True, folder=True)
        rowIdx += 1
        add_entry("Detections Dir", self.detections, rowIdx, browse=True, folder=True)
        rowIdx += 1
        add_entry("Tracks Dir", self.tracks, rowIdx, browse=True, folder=True)

        # -----------------------
        # Dataset dropdown
        # -----------------------
        rowIdx += 1
        ttk.Label(frame, text="Dataset").grid(row=rowIdx, column=0, sticky="w")
        ttk.Combobox(frame, textvariable=self.dataset,
                     values=["V3", "V5", "V6"]).grid(row=rowIdx, column=1, sticky="ew")

        # -----------------------
        # MovieTrack dropdown
        # -----------------------
        rowIdx += 1
        ttk.Label(frame, text="Tracking").grid(row=rowIdx, column=0, sticky="w")
        ttk.Combobox(frame, textvariable=self.movieTrack,
                     values=["movie", "CSV only"]).grid(row=rowIdx, column=1, sticky="ew")

        # -----------------------
        # Checkboxes
        # -----------------------
        rowIdx += 1
        ttk.Checkbutton(frame, text="Check Taxa", variable=self.checkTaxa)\
            .grid(row=rowIdx, column=1, sticky="w")

        #rowIdx += 1
        #ttk.Checkbutton(frame, text="Trap File Path", variable=self.trapFilePath)\
        #    .grid(row=rowIdx, column=1, sticky="w")

        rowIdx += 1
        ttk.Checkbutton(frame, text="Ignore Vegetation", variable=self.ignoreVegetation)\
            .grid(row=rowIdx, column=1, sticky="w")

        # -----------------------
        # Start/Stop buttons
        # -----------------------
        rowIdx += 1
        self.start_btn = ttk.Button(frame, text="Start", command=self.start_process)
        self.start_btn.grid(row=rowIdx, column=0, pady=10)

        self.stop_btn = ttk.Button(frame, text="Stop", command=self.stop_process, state=tk.DISABLED)
        self.stop_btn.grid(row=rowIdx, column=1, pady=10)

        # -----------------------
        # Console window
        # -----------------------
        console_frame = ttk.LabelFrame(self.root, text="Console Output")
        console_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        console_frame.rowconfigure(0, weight=1)
        console_frame.columnconfigure(0, weight=1)

        self.console = tk.Text(console_frame, bg="black", fg="white", wrap="word")
        self.console.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(console_frame, orient="vertical", command=self.console.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.console["yscrollcommand"] = scrollbar.set

    # -----------------------------------------------------------
    # File/folder browsing
    # -----------------------------------------------------------
    def browse_folder(self, variable):
        path = filedialog.askdirectory()
        if path:
            variable.set(path + "/")
            
    # -----------------------------------------------------------
    # Browse functions
    # -----------------------------------------------------------
    #def browse_folder(self, variable):
    #    path = filedialog.askdirectory()
    #    if path:
    #        variable.set(path)

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
            sys.executable, "pipeTrackInsectsTaxon.py",
            "--images", self.images.get(),
            "--video", self.video.get(),
            "--detections", self.detections.get(),
            "--tracks", self.tracks.get(),
            "--dataset", self.dataset.get(),
            "--checkTaxa", "True" if self.checkTaxa.get() else "",
            #"--trapFilePath", str(self.trapFilePath.get()),
            "--ignoreVegetation",  "True" if self.ignoreVegetation.get() else "",
            "--movieTrack",  self.movieTrack.get() if self.movieTrack.get() != "CSV only" else "",
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
    # Read subprocess output
    # -----------------------------------------------------------
    def read_output(self):
        for line in self.process.stdout:
            self.console.insert(tk.END, line)
            self.console.see(tk.END)

        self.process = None
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    # -----------------------------------------------------------
    # Stop subprocess
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
