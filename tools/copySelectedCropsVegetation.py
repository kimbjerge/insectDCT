# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 08:23:15 2025

@author: Kim Bjerge
"""
import os
import numpy as np
import shutil

# %% Insect plots
if __name__ == '__main__':
    
    
    cropPath = "E:/insectsDCT_datasets/classifier/NI2_MAMBOV2/vegetation/"
    pathDest = "E:/insectsDCT_datasets/classifier/MAMBO_NI2/vegetation20p/"
    
    selectedVegetation = 5
    
    count = 0
    cntFiles = 0
    for fileName in sorted(os.listdir(cropPath)):
        if fileName != '.gitignore':
            count += 1
            if count%selectedVegetation == 0:
                cntFiles += 1
                fileNameSrc = cropPath + fileName    
                shutil.copy(fileNameSrc, pathDest+fileName)
                print(fileNameSrc)
                
    print(f"Seleted {cntFiles} out of {count} vegetation files")