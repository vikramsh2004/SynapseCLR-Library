# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 13:54:15 2025

@author: Chen-Lab
"""

import numpy as np

file_path = r'C:\Users\jckgb\SynapseCLR\data\processed\processed_1000004__2_256_256_52.npy'

# Load the 3D data
data = np.load(file_path)

# Print information about the array
print("Array shape:", data.shape)
print("Data type:", data.dtype)
print(data)