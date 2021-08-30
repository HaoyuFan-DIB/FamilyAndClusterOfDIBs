# Make Heatmaps for Figure 2 (small) and Figure B1 (full size)
# r_regular on top, r_normalized at bottom

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from CorrDataPath import PicPath, DataPath
import os

# load r_tables and DIB sequence
normalized_filename = os.path.join(DataPath, "r_matrix_normalized.csv")
normalized_df = pd.read_csv(normalized_filename)
DIB_labels = normalized_df.label.to_list()
DIB_labels = np.array([str(item) for item in DIB_labels])
normalized_df = normalized_df.drop(columns=["label"])
normalized_matrix = normalized_df.to_numpy()

regular_filename = os.path.join(DataPath, "r_matrix_regular.csv")
regular_df = pd.read_csv(regular_filename)
regular_df = regular_df.drop(columns=["label"])
regular_matrix = regular_df.to_numpy()

cluster_filename = os.path.join(DataPath, "ClusteringResult.csv")
cluster_df = pd.read_csv(cluster_filename)
sort_idx = cluster_df["Sort_Idx"].to_numpy()

# resort labels and matrix
normalized_matrix = normalized_matrix[sort_idx][:, sort_idx]
regular_matrix = regular_matrix[sort_idx][:,sort_idx]
DIB_labels = DIB_labels[sort_idx]
# normalized in lower triangle, regular in upper triangle, ones on diagonal
r_matrix = np.tril(normalized_matrix, k=-1) + \
            np.triu(regular_matrix, k=1) + \
            np.diag(np.ones(len(normalized_matrix)))

# Clip NXs and prepare for big heatmap
idx_clip = [True if "N" not in item else False for item in DIB_labels]
r_matrix = r_matrix[idx_clip][:, idx_clip]
DIB_labels = DIB_labels[idx_clip]

# Plot big heatmap
fig = plt.figure(figsize=(8,6), dpi=200)
ax = fig.add_subplot(1,1,1)
im = ax.imshow(r_matrix)
cbar = ax.figure.colorbar(im, ax=ax)

# labels
ax.set_xticks(np.arange(len(DIB_labels)))
ax.set_xticklabels(DIB_labels, fontsize=7)
plt.plot([-0.5, 53.5], [45.5,45.5], color="0.5")
plt.plot([45.5, 45.5], [-0.5,53.5], color="0.5")
plt.setp(ax.get_xticklabels(), rotation=75, ha="right",
         rotation_mode="anchor")
ax.set_yticks(np.arange(len(DIB_labels)))
ax.set_yticklabels(DIB_labels, fontsize=8)

fig.tight_layout()
filename = os.path.join(PicPath, "FigureB1_BigHeatMap.png")
plt.savefig(filename)
plt.close()

# Clip and keep only the most anti-correlated DIBs, prepare for small heatmap
anti_idx = np.unique(np.where(r_matrix <= -0.5))
r_matrix = r_matrix[anti_idx][:, anti_idx]
DIB_labels = DIB_labels[anti_idx]

# Make small heatmap
fig = plt.figure(figsize=(8,6), dpi=200)
ax = fig.add_subplot(1,1,1)
im = ax.imshow(r_matrix)
cbar = ax.figure.colorbar(im, ax=ax)

# labels
ax.set_xticks(np.arange(len(DIB_labels)))
ax.set_xticklabels(DIB_labels, fontsize=7)
plt.setp(ax.get_xticklabels(), rotation=75, ha="right",
         rotation_mode="anchor")
ax.set_yticks(np.arange(len(DIB_labels)))
ax.set_yticklabels(DIB_labels, fontsize=8)

fig.tight_layout()
filename = os.path.join(PicPath, "Figure2_SmallHeatMap.png")
plt.savefig(filename)
plt.close()
