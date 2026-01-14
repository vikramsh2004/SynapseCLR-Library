# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 14:20:02 2026

@author: Chen-Lab
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns 

# ==========================================
# CONFIGURATION
# ==========================================
# Update this path again if needed!
file_path = r"D:/SynapseCLR/training_data/features_drop_ann/extracted_features__node.0__epoch.99__encoder.fc.npy" 

estimated_clusters = 3 

# ==========================================
# 1. LOAD DATA
# ==========================================
if not "file_path" in locals() or not file_path:
    print("Please set the 'file_path' variable!")
else:
    # Load data
    embeddings = np.load(file_path)
    print(f"Loaded data shape: {embeddings.shape} (Samples, Features)")

    # ==========================================
    # 2. CREATE REGIONS (AUTO-CLUSTERING)
    # ==========================================
    print(f"Finding {estimated_clusters} natural clusters in the data...")
    kmeans = KMeans(n_clusters=estimated_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # ==========================================
    # 3. REDUCE TO 2D (t-SNE)
    # ==========================================
    print("Projecting to 2D using t-SNE...")
    
    # PCA step for stability
    n_components_pca = min(50, embeddings.shape[1])
    pca_50 = PCA(n_components=n_components_pca)
    pca_result = pca_50.fit_transform(embeddings)

    # Run t-SNE (Fixed: removed n_iter)
    perplexity_val = min(30, len(embeddings) - 1) 
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity_val, random_state=42)
    tsne_results = tsne.fit_transform(pca_result)

    # ==========================================
    # 4. PLOT
    # ==========================================
    plt.figure(figsize=(10, 8), dpi=100)
    
    scatter = sns.scatterplot(
        x=tsne_results[:, 0], 
        y=tsne_results[:, 1],
        hue=cluster_labels, 
        palette="viridis", 
        s=100, 
        legend="full"
    )

    plt.title(f"t-SNE Visualization of {len(embeddings)} Synapses\n(Colors = Auto-detected Clusters)", fontsize=15)
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)

    # Label the cluster centers
    for i in range(estimated_clusters):
        points = tsne_results[cluster_labels == i]
        if len(points) > 0:
            center = points.mean(axis=0)
            plt.text(center[0], center[1], f"Region {i}", 
                     fontsize=12, weight='bold', color='black', 
                     bbox=dict(facecolor='white', alpha=0.6, edgecolor='black'))

    plt.legend(title="Cluster ID")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ==========================================
    # 5. TEXT STATS
    # ==========================================
    print("\n" + "="*40)
    print(" REGION ANALYSIS")
    print("="*40)
    for i in range(estimated_clusters):
        count = np.sum(cluster_labels == i)
        print(f"Region {i}: Contains {count} images.")