# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 10:15:10 2025

@author: Chen-Lab
"""

import pandas as pd
import numpy as np
import os
from caveclient import CAVEclient

# 1. Initialize Client
client = CAVEclient("jchen_mouse_cortex")
root_id = 720575941099131663

# --- QUERY DATA ---

# Query inputs (Postsynaptic)
input_syn_df = client.materialize.synapse_query(
    post_ids=root_id, 
    desired_resolution=[7.5, 7.5, 50]
)
print(f"Total number of input synapses for {root_id}: {len(input_syn_df)}")

# Query outputs (Presynaptic)
output_syn_df = client.materialize.synapse_query(
    pre_ids=root_id, 
    desired_resolution=[7.5, 7.5, 50]
)
print(f"Total number of output synapses for {root_id}: {len(output_syn_df)}")

# --- PROCESS DATA ---

# Add direction labels
input_syn_df['direction'] = 'input'
output_syn_df['direction'] = 'output'

# Combine into one DataFrame
combined_df = pd.concat([input_syn_df, output_syn_df], ignore_index=True)

# Inspect
print(f"Total combined synapses: {len(combined_df)}")
print(combined_df['direction'].value_counts())

# --- SAVE FILE ---

# 1. Define the specific folder path
# Using r'' tells Python to treat backslashes as literal characters
folder_path = r'C:\Users\jckgb\SynapseCLR'
file_name = 'combined_synapses.csv'

# 2. Create the folder if it doesn't exist
os.makedirs(folder_path, exist_ok=True)

# 3. Construct the full file path
save_path = os.path.join(folder_path, file_name)

# 4. Save the dataframe
combined_df.to_csv(save_path, index=False)

print(f"Success! File saved to: {save_path}")