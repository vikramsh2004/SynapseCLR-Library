# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 17:30:56 2025

@author: Chen-Lab
"""

import sys
from types import ModuleType
import os
import numpy as np
import pandas as pd
import time
from cloudvolume import CloudVolume
import cloudfiles.monitoring
from caveclient import CAVEclient
from scipy.ndimage import zoom  # <--- NEW IMPORT for resampling

# 1. Initialize Client
client = CAVEclient("jchen_mouse_cortex")

# ==========================================
# 1. SETUP & WINDOWS FIX
# ==========================================
m = ModuleType("synapse_augmenter")
consts_mod = ModuleType("consts")
consts_mod.MASK_BACKGROUND = 0            
consts_mod.MASK_PRE_SYNAPTIC_NEURON = 1   
consts_mod.MASK_SYNAPTIC_CLEFT = 2        
consts_mod.MASK_POST_SYNAPTIC_NEURON = 3  
m.consts = consts_mod
sys.modules["synapse_augmenter"] = m

def patched_end_io(self, flight_id, num_bytes):
    if flight_id not in self._in_flight: return
    start_us = int(self._in_flight.pop(flight_id) * 1e6)
    end_us = int(time.time() * 1e6)
    if end_us <= start_us: end_us = start_us + 1
    self._in_flight_bytes -= num_bytes
    self._intervaltree.addi(start_us, end_us, [flight_id, num_bytes]) 
    self._total_bytes_landed += num_bytes

cloudfiles.monitoring.TransmissionMonitor.end_io = patched_end_io

# ==========================================
# 2. CONFIGURATION
# ==========================================
proc_data_path = r'D:\SynapseCLR\training_data' 
os.makedirs(proc_data_path, exist_ok=True)

csv_path = r"C:/Users/jckgb/SynapseCLR/combined_synapses_ourdata.csv"

# --- CLOUD LAYERS ---
em_layer_path = "precomputed://gs://zetta_jchen_mouse_cortex_001_alignment/img" 
seg_layer_path = "graphene://middleauth+https://cave.fanc-fly.com/segmentation/table/jchen_mouse_cortex/"
cleft_layer_path = "precomputed://gs://zetta_jchen_mouse_cortex_001_synapse/250728_assignment/seg"

# --- TARGET CONFIGURATION (What the Model Expects) ---
# The model was trained on 4nm x 4nm x 40nm resolution
target_res = np.array([4, 4, 40])       
target_shape = np.array([256, 256, 52]) 

# Calculate the physical Field of View (FOV) we need to grab
target_phys_size = target_shape * target_res  # [1024, 1024, 2080] nm
print(f"Target Physical FOV: {target_phys_size} nm")

# --- SOURCE RESOLUTIONS (What we are downloading) ---
res_csv = np.array([7.5, 7.5, 50])  
res_em = np.array([7.5, 7.5, 50])
res_seg = np.array([15, 15, 50])
res_cleft = np.array([15, 15, 50])

print("Initializing CloudVolumes...")
vol_em = CloudVolume(em_layer_path, mip=0, use_https=True, fill_missing=True, parallel=1)
vol_seg = CloudVolume(seg_layer_path, mip=0, use_https=True, fill_missing=True, parallel=1)
vol_cleft = CloudVolume(cleft_layer_path, mip=0, use_https=True, fill_missing=True, parallel=1)

df_synapses = pd.read_csv(csv_path).head(100)
print(f"Test Mode: Processing {len(df_synapses)} synapse.")

# ==========================================
# 3. COORDINATE LOGIC
# ==========================================

def get_voxel_bounds(center_vox, size_voxels):
    center_vox = np.array(center_vox).astype(int)
    start = center_vox - (np.array(size_voxels) // 2)
    end = start + np.array(size_voxels)
    return np.s_[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

def upsample_layer(layer_data, target_shape):
    """Upsamples a smaller 3D array to match the target shape (via repeat)."""
    if layer_data.shape == target_shape:
        return layer_data
    rep_x = int(target_shape[0] / layer_data.shape[0])
    rep_y = int(target_shape[1] / layer_data.shape[1])
    rep_z = int(target_shape[2] / layer_data.shape[2])
    return layer_data.repeat(rep_x, axis=0).repeat(rep_y, axis=1).repeat(rep_z, axis=2)

def download_cloud_data(row, vol_em, vol_seg, vol_cleft, client=client):
    # -----------------------------------------------------------
    # 1. Calculate Dynamic Download Size (CORRECTED)
    # -----------------------------------------------------------
    # We calculate how many voxels cover the target physical size
    raw_voxels = target_phys_size / res_em
    download_shape_em = np.round(raw_voxels).astype(int)
    
    # --- FIX: Enforce Even Numbers for X and Y ---
    # Because Seg (15nm) is 2x coarser than EM (7.5nm), we need an even number 
    # of EM pixels to avoid rounding errors during upsampling.
    # (e.g., 137 px EM -> 68 px Seg -> 136 px Upsampled Mask -> ERROR)
    if download_shape_em[0] % 2 != 0: download_shape_em[0] += 1
    if download_shape_em[1] % 2 != 0: download_shape_em[1] += 1
    
    # -----------------------------------------------------------
    # 2. Coordinate Math
    # -----------------------------------------------------------
    coord_str = str(row['ctr_pt_position']).replace('[', '').replace(']', '')
    if ',' in coord_str: raw_csv_vox = np.array([float(x) for x in coord_str.split(',')])
    else: raw_csv_vox = np.array([float(x) for x in coord_str.split()])

    phys_center = raw_csv_vox * res_csv
    
    # EM Bounds (7.5 nm)
    center_em_vox = phys_center / res_em
    slices_em = get_voxel_bounds(center_em_vox, download_shape_em)
    
    # Seg/Cleft Bounds (15 nm)
    center_seg_vox = phys_center / res_seg
    scale_factor = res_em / res_seg 
    
    # Calculate Seg shape based on the NOW EVEN download_shape_em
    seg_download_shape = (np.array(download_shape_em) * scale_factor).astype(int)
    slices_seg = get_voxel_bounds(center_seg_vox, seg_download_shape)

    # -----------------------------------------------------------
    # 3. Download Data (EM, Neurons, Clefts)
    # -----------------------------------------------------------
    img = vol_em[slices_em].squeeze()
    seg_sv = vol_seg[slices_seg].squeeze()
    seg_cleft_raw = vol_cleft[slices_seg].squeeze()
    
    # -----------------------------------------------------------
    # 4. Process Neurons (Update Roots + Upsample)
    # -----------------------------------------------------------
    unique_sv_ids, inverse_indices = np.unique(seg_sv, return_inverse=True)
    root_ids_list = client.chunkedgraph.get_roots(unique_sv_ids)
    seg_neuron_15nm = np.array(root_ids_list)[inverse_indices].reshape(seg_sv.shape)

    pre_id = int(row['pre_pt_root_id'])
    post_id = int(row['post_pt_root_id'])
    try:
        updated_roots = client.chunkedgraph.get_latest_roots([pre_id, post_id])
        pre_id_curr = updated_roots[0]
        post_id_curr = updated_roots[1]
    except Exception as e:
        print(f"Warning: Root update failed. {e}")
        pre_id_curr = pre_id
        post_id_curr = post_id

    mask_pre_15nm = np.zeros_like(seg_neuron_15nm, dtype=np.uint8)
    mask_post_15nm = np.zeros_like(seg_neuron_15nm, dtype=np.uint8)
    mask_pre_15nm[seg_neuron_15nm == pre_id_curr] = 1
    mask_post_15nm[seg_neuron_15nm == post_id_curr] = 1

    # -----------------------------------------------------------
    # 5. Process Cleft
    # -----------------------------------------------------------
    mask_cleft_15nm = np.zeros_like(seg_cleft_raw, dtype=np.uint8)
    cleft_pxs = np.argwhere(seg_cleft_raw > 0)
    if len(cleft_pxs) > 0:
        chunk_center = np.array(seg_cleft_raw.shape) / 2
        deltas = np.linalg.norm(cleft_pxs - chunk_center, axis=1)
        closest_idx = np.argmin(deltas)
        target_cleft_id = seg_cleft_raw[tuple(cleft_pxs[closest_idx])]
        mask_cleft_15nm[seg_cleft_raw == target_cleft_id] = 1

    # -----------------------------------------------------------
    # 6. Combine at Source Resolution (7.5nm)
    # -----------------------------------------------------------
    # Upsample masks to match the EM image shape
    mask_pre_up = upsample_layer(mask_pre_15nm, img.shape)
    mask_post_up = upsample_layer(mask_post_15nm, img.shape)
    mask_cleft_up = upsample_layer(mask_cleft_15nm, img.shape)

    final_mask = np.zeros_like(img, dtype=np.uint8)
    final_mask[:] = consts_mod.MASK_BACKGROUND
    final_mask[mask_post_up == 1] = consts_mod.MASK_POST_SYNAPTIC_NEURON
    final_mask[mask_pre_up == 1] = consts_mod.MASK_PRE_SYNAPTIC_NEURON
    final_mask[mask_cleft_up == 1] = consts_mod.MASK_SYNAPTIC_CLEFT 

    # -----------------------------------------------------------
    # 7. RESAMPLE TO TARGET (4nm / 256px)
    # -----------------------------------------------------------
    # Calculate scale factors
    zoom_factors = target_shape / np.array(img.shape)
    
    # Resize Image (order=1 for smoothness)
    img_resampled = zoom(img, zoom_factors, order=1)
    
    # Resize Mask (order=0 to preserve integers)
    mask_resampled = zoom(final_mask, zoom_factors, order=0)
    
    # Stack: (Channels, X, Y, Z)
    return np.stack([img_resampled, mask_resampled], axis=0)

# ==========================================
# 4. MAIN LOOP
# ==========================================
print("Starting download...")

for index, row in df_synapses.iterrows():
    synapse_id = str(row['id'])
    try:
        print(f"Downloading ID: {synapse_id}...")
        img_cxyz = download_cloud_data(row, vol_em, vol_seg, vol_cleft)
        
        # Validation
        print(f"   -> Final Shape: {img_cxyz.shape}")
        
        # SAVE
        out_filename = f"cloud_{synapse_id}.npy"
        save_path = os.path.join(proc_data_path, out_filename)
        np.save(save_path, img_cxyz)
        print(f"✅ Success! Saved to: {save_path}\n")
            
    except Exception as e:
        print(f"❌ Failed on {synapse_id}: {e}")
        import traceback
        traceback.print_exc()

print("Done!")