# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 21:38:07 2025

@author: crazy
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from caveclient import CAVEclient
client = CAVEclient("PUT_OURS_HERE")

root_id = 78820140841712713

# Query synapse table with synapse_query()
input_syn_df = client.materialize.synapse_query(post_ids=root_id)

print(f"Total number of input synapses for {root_id}: {len(input_syn_df)}")
input_syn_df.head()

# Query synapse table with synapse_query()
output_syn_df = client.materialize.synapse_query(pre_ids=root_id)

print(f"Total number of output synapses for {root_id}: {len(output_syn_df)}")
output_syn_df.head()

# 1. Add a label column to distinguish the two sets
input_syn_df['direction'] = 'input'   # These are inputs (postsynaptic)
output_syn_df['direction'] = 'output' # These are outputs (presynaptic)

# 2. Combine them into one DataFrame
combined_df = pd.concat([input_syn_df, output_syn_df], ignore_index=True)

# Optional: Inspect the combined dataframe
print(f"Total combined synapses: {len(combined_df)}")
print(combined_df['direction'].value_counts())

# 3. Save to a single CSV file
combined_df.to_csv('combined_synapses.csv', index=False)

import os  # Standard library for operating system interactions

# 1. Define your folder and filename
folder_name = 'SynapseCLR'
file_name = 'combined_synapses.csv'

# 2. Create the folder if it doesn't exist yet
# 'exist_ok=True' prevents an error if you run the script multiple times
os.makedirs(folder_name, exist_ok=True)

# 3. Construct the full file path safely
# This creates something like "SynapseCLR/combined_synapses.csv"
save_path = os.path.join(folder_name, file_name)

# 4. Save the dataframe to that path
combined_df.to_csv(save_path, index=False)

print(f"Success! File saved to: {os.path.abspath(save_path)}")