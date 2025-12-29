# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 21:14:32 2025

@author: crazy
"""

import sys
from types import ModuleType
!pip install cloud-volume

# 1. Create a dummy module for synapse_augmenter
m = ModuleType("synapse_augmenter")

# 2. Create a dummy module for consts
consts_mod = ModuleType("consts")

# 3. Define the constants the script expects
# IMPORTANT: If your mask integers are different (e.g., 255 for cleft), change them here.
consts_mod.MASK_PRE_SYNAPTIC_NEURON = 1
consts_mod.MASK_SYNAPTIC_CLEFT = 2
consts_mod.MASK_POST_SYNAPTIC_NEURON = 3

# 4. Attach consts to synapse_augmenter
m.consts = consts_mod

# 5. Inject it into sys.modules so Python thinks it's installed
sys.modules["synapse_augmenter"] = m

print("Successfully mocked synapse_augmenter. You can now run the import cell.")

import os
import sys
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import logging

from typing import Tuple
import bisect
from collections import Counter
from operator import itemgetter

from synapse_augmenter import consts

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.style.use('dark_background')

logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_info = print

# Configuration

# Use the specific paths you provided
raw_em_data_path = r'C:/Users/crazy/Downloads/SynapseCLR_Work/Raw' #change this to file path on computer
raw_mask_data_path = r'C:/Users/crazy/Downloads/SynapseCLR_Work/Raw_Mask' #change this to file path on computer
proc_data_path = r'C:\Users\crazy\Downloads\SynapseCLR_Work\SynapseCLR' #change this to SynapseCLR folder on computer

# hyperparameters

# promote sections with at least this much fraction of cutout pixels to fully masked
cutout_threshold = 0.01

# z spacing vs. xy spacing
axial_to_sagittal_spacing = 5

seg_mask_map = {
    'MASK_PRE_SYNAPTIC_NEURON': consts.MASK_PRE_SYNAPTIC_NEURON,
    'MASK_SYNAPTIC_CLEFT': consts.MASK_SYNAPTIC_CLEFT,
    'MASK_POST_SYNAPTIC_NEURON': consts.MASK_POST_SYNAPTIC_NEURON
}

# which segmentation region(s) to cut the EM data with?
cut_intensity_with_seg_masks = False
output_channel_desc = [
    (consts.MASK_PRE_SYNAPTIC_NEURON,),
    (consts.MASK_SYNAPTIC_CLEFT,),
    (consts.MASK_POST_SYNAPTIC_NEURON,)
]

import os

from cloudvolume import CloudVolume

# --- NEW CONFIGURATION ---
# 1. Path to your CSV with columns: 'id', 'x', 'y', 'z', 'pre_id', 'post_id'
csv_path = r"PUT_CSV_TABLE_HERE"

# 2. CloudVolume Links
em_layer_path = "PUT_EM_LAYER_LINK_HERE" 
seg_layer_path = "PUT_SEG_LAYER_LINK_HERE"

# 3. Settings
mip_level = 0 
cutout_size = [256, 256, 32] 
resolution = [15, 15, 50] # CHANGE THIS to match your dataset resolution (nm)

# Initialize
vol_em = CloudVolume(em_layer_path, mip=mip_level, use_https=True, fill_missing=True)
vol_seg = CloudVolume(seg_layer_path, mip=mip_level, use_https=True, fill_missing=True)
df_synapses = pd.read_csv(csv_path)

def get_voxel_bounds(center_nm, size_voxels, resolution):
    center_vox = np.array(center_nm) / np.array(resolution)
    center_vox = center_vox.astype(int)
    start = center_vox - (np.array(size_voxels) // 2)
    end = start + np.array(size_voxels)
    return np.s_[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

def download_cloud_data(row, vol_em, vol_seg, cutout_size):
    center_pos = [row['x'], row['y'], row['z']] # ENSURE these match your CSV headers
    slices = get_voxel_bounds(center_pos, cutout_size, resolution)
    
    img = vol_em[slices].squeeze()
    seg = vol_seg[slices].squeeze()
    
    mask = np.zeros_like(seg, dtype=np.uint8)
    mask[seg == row['pre_id']] = 1  # 1 = Pre-synaptic
    mask[seg == row['post_id']] = 2 # 2 = Post-synaptic
    # (Optional: Add cleft logic here if you have cleft IDs)

    return np.stack([img, mask], axis=-1)

def get_cutout_fraction_z(img_xyz: np.ndarray, cutout_pixel_value: int = 0) -> float:
    section_area = img_xyz.shape[0] * img_xyz.shape[1]
    cutout_fraction_z = np.sum(img_xyz == cutout_pixel_value, axis=(0, 1)) / section_area
    return cutout_fraction_z

import os

# 1. Check if files were found at all
print(f"Total EM files found: {len(raw_em_data_file_path_list)}")
print(f"Total Mask files found: {len(raw_mask_data_file_path_list)}")

# 2. Inspect the first file to see how the ID extraction works
if len(raw_em_data_file_path_list) > 0:
    sample_em = raw_em_data_file_path_list[0]
    em_filename = os.path.basename(sample_em)
    em_id = get_id_from_em_data_path(sample_em)
    print(f"\n--- EM SAMPLE ---")
    print(f"File: {em_filename}")
    print(f"Extracted ID: '{em_id}'")

if len(raw_mask_data_file_path_list) > 0:
    sample_mask = raw_mask_data_file_path_list[0]
    mask_filename = os.path.basename(sample_mask)
    mask_id = get_id_from_mask_data_path(sample_mask)
    print(f"\n--- MASK SAMPLE ---")
    print(f"File: {mask_filename}")
    print(f"Extracted ID: '{mask_id}'")
    
# DIAGNOSTIC CELL - Run this to see why the match is failing

print("--- DIAGNOSTICS ---")
print(f"Number of EM files loaded: {len(raw_em_data_file_path_list)}")
print(f"Number of Mask files loaded: {len(raw_mask_data_file_path_list)}")

if len(raw_em_data_file_path_list) > 0:
    print("\nExample EM file path:", raw_em_data_file_path_list[0])
    print("Example EM ID extracted:", get_id_from_em_data_path(raw_em_data_file_path_list[0]))
else:
    print("\n❌ EM file list is empty! Check your 'raw_em_data_path' variable.")

if len(raw_mask_data_file_path_list) > 0:
    print("\nExample Mask file path:", raw_mask_data_file_path_list[0])
    print("Example Mask ID extracted:", get_id_from_mask_data_path(raw_mask_data_file_path_list[0]))
else:
    print("\n❌ Mask file list is empty! Check your 'raw_mask_data_path' variable.")
    
# make a dataframe
import os

def get_id_from_em_data_path(em_data_path: str) -> str:
    filename = os.path.basename(em_data_path)
    # Example: data_..._raw_1000379.npy
    # 1. Remove .npy extension -> data_..._raw_1000379
    # 2. Split by underscore and take the last part -> 1000379
    return filename.replace('.npy', '').split('_')[-1]

def get_id_from_mask_data_path(mask_data_path: str) -> str:
    filename = os.path.basename(mask_data_path)
    # Example: data_..._masks_1000348_mask.npy
    # 1. Remove '_mask.npy' suffix -> data_..._masks_1000348
    # 2. Split by underscore and take the last part -> 1000348
    return filename.replace('_mask.npy', '').split('_')[-1]

synapse_id_to_em_data_path_map = {
    get_id_from_em_data_path(em_data_path): em_data_path
    for em_data_path in raw_em_data_file_path_list
}

synapse_id_to_mask_data_path_map = {
    get_id_from_mask_data_path(mask_data_path): mask_data_path
    for mask_data_path in raw_mask_data_file_path_list
}

# synapse IDs with both EM data and mask data
complete_synapse_id_set = set(synapse_id_to_em_data_path_map.keys()).intersection(set(synapse_id_to_mask_data_path_map.keys()))
complete_synapse_id_list = sorted(list(complete_synapse_id_set))
complete_em_data_path_list = list(map(synapse_id_to_em_data_path_map.get, complete_synapse_id_list))
complete_mask_data_path_list = list(map(synapse_id_to_mask_data_path_map.get, complete_synapse_id_list))

print(f"Found {len(complete_synapse_id_list)} matching pairs.")

def center_crop_3d_np(layer: np.ndarray, target_shape: Tuple[int]) -> np.ndarray:
    layer_depth, layer_height, layer_width = layer.shape
    target_depth, target_height, target_width = target_shape
    assert layer_depth >= target_depth
    assert layer_height >= target_height
    assert layer_width >= target_width
    diff_x = (layer_width - target_width) // 2
    diff_y = (layer_height - target_height) // 2
    diff_z = (layer_depth - target_depth) // 2
    return layer[
        diff_z:(diff_z + target_depth),
        diff_y:(diff_y + target_height),
        diff_x:(diff_x + target_width)]

os.makedirs(proc_data_path, exist_ok=True)

log_frequency = 500

out_filename_list = []
n_cutout_sections_list = []
pre_synaptic_neuron_volume_list = []
post_synaptic_neuron_volume_list = []
synaptic_cleft_volume_list = []

print(f"Starting processing of {len(df_synapses)} synapses...")

for index, row in df_synapses.iterrows():
    synapse_id = str(row['id'])
    
    try:
        # 1. Download
        img_cxyz = download_cloud_data(row, vol_em, vol_seg, cutout_size)
        
        # 2. Extract (recreating the variables the rest of the logic needs)
        em_xyz = img_cxyz[..., 0] 
        mask_xyz = img_cxyz[..., 1]

        # 3. Simple Cutout Check (using the helper function you kept)
        if get_cutout_fraction_z(em_xyz) > cutout_threshold:
            print(f"Skipping {synapse_id}: Too much empty space.")
            continue

        # 4. Save
        out_filename = f"cloud_{synapse_id}.npy"
        save_path = os.path.join(proc_data_path, out_filename)
        np.save(save_path, img_cxyz)
        
        if index % 10 == 0:
            print(f"Saved {synapse_id} ({index}/{len(df_synapses)})")
            
    except Exception as e:
        print(f"Failed on {synapse_id}: {e}")