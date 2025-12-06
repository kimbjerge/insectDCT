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
        self.root.title("Smart Insect Crop Generator")

        # Resizable GUI
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

        self.process = None

        # --------------------------
        # Parameter Variables
        # --------------------------
        self.CSVfiles = tk.StringVar(value="./detections/")
        self.imagesPath = tk.StringVar(value="./images/")
        self.videoPath = tk.StringVar(value="")
        self.cropsPath = tk.StringVar(value="./crops/")
        self.dataset = tk.StringVar(value="V6")

        self.create_widgets()

    # -----------------------------------------------------------
    # Create GUI elements
    # -----------------------------------------------------------
    def create_widgets(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.grid(row=0, column=0, sticky="ew")
        frame.columnconfigure(1, weight=1)

        # Helper: label + entry + browse (folder)
        def add_entry(label, variable, row):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w")
            ttk.Entry(frame, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=5)
            ttk.Button(frame, text="Browse", command=lambda: self.browse_folder(variable)).grid(row=row, column=2, padx=5)

        rowIdx = 0
        add_entry("CSV Files Dir", self.CSVfiles, rowIdx); rowIdx += 1
        add_entry("Images Dir", self.imagesPath, rowIdx); rowIdx += 1
        add_entry("Video Dir", self.videoPath, rowIdx); rowIdx += 1
        add_entry("Crops Dir", self.cropsPath, rowIdx); rowIdx += 1

        # -----------------------
        # Dataset dropdown
        # -----------------------
        ttk.Label(frame, text="Dataset").grid(row=rowIdx, column=0, sticky="w")
        ttk.Combobox(frame, textvariable=self.dataset,
                     values=["V3", "V4", "V5", "V6"]).grid(row=rowIdx, column=1, sticky="ew")
        rowIdx += 1

        # -----------------------
        # Start & Stop buttons
        # -----------------------
        self.start_btn = ttk.Button(frame, text="Start", command=self.start_process)
        self.start_btn.grid(row=rowIdx, column=0, pady=10)

        self.stop_btn = ttk.Button(frame, text="Stop", command=self.stop_process, state=tk.DISABLED)
        self.stop_btn.grid(row=rowIdx, column=1, pady=10)

        # -----------------------
        # Console output
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
    # Folder browsing
    # -----------------------------------------------------------
    def browse_folder(self, variable):
        path = filedialog.askdirectory()
        if path:
            variable.set(path + "/")

    # -----------------------------------------------------------
    # Start subprocess
    # -----------------------------------------------------------
    def start_process(self):
        if self.process is not None:
            return

        cmd = [
            sys.executable, "createCrops.py",   # TODO: replace script name
            "--CSVfiles", self.CSVfiles.get(),
            "--imagesPath", self.imagesPath.get(),
            "--videoPath", self.videoPath.get(),
            "--cropsPath", self.cropsPath.get(),
            "--dataset", self.dataset.get()
        ]

        self.console.insert(tk.END, "Starting process...\n")
        self.console.insert(tk.END, " ".join(cmd) + "\n\n")
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
