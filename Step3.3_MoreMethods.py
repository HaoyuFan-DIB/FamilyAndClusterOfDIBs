# Check results of t-Distributed Stochastic Neighbor Embedding and UMAP
# The figure will not go into the manuscript

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, TSNE
from umap import UMAP
from CorrDataPath import DataPath, PicPath
import os

# Load r-table and cluster result for labeling
r_filename = os.path.join(DataPath, "r_matrix_normalized.csv")
r_df = pd.read_csv(r_filename)
All_labels = r_df.label.to_list()
All_labels = np.array([str(item) for item in All_labels])
r_df = r_df.drop(columns=["label"])
dissimilarity_matrix = 1 - r_df.to_numpy()

cluster_filename = os.path.join(DataPath, "ClusteringResult.csv")
cluster_df = pd.read_csv(cluster_filename)
color_HAC = {}
color_Kmeans = {}
for i in range(len(cluster_df)):
    color_HAC[str(cluster_df["DIBs"].iloc[i])] = cluster_df["Norm HAC Color"].iloc[i]
    color_Kmeans[str(cluster_df["DIBs"].iloc[i])] = cluster_df["Norm Kmeans Color"].iloc[i]

# Plotting
font_size = 10
symbol_size = 30
label_size = 8

fig = plt.figure(figsize=(4,8), dpi=200)

methods = ["MDS", "tDSNE", "UMAP"]
for i, method in enumerate(methods):
    # prepare data from three methods
    if i == 0:
        X_transformed = MDS(n_components=2, dissimilarity='precomputed', random_state=31)\
            .fit_transform(dissimilarity_matrix)
    if i == 1:
        X_transformed = TSNE(n_components=2).fit_transform(dissimilarity_matrix)
    if i == 2:
        X_transformed = UMAP().fit_transform(dissimilarity_matrix)

    ax = fig.add_subplot(3, 1, i+1)
    for j, DIB in enumerate(All_labels):
        if color_HAC[DIB] == color_Kmeans[DIB]:
            ax.scatter(X_transformed[j][0], X_transformed[j][1],
                       color=color_HAC[DIB], s=symbol_size)
        else:
            ax.scatter(X_transformed[j][0], X_transformed[j][1],
                       color=color_Kmeans[DIB], s=symbol_size)
            ax.scatter(X_transformed[j][0], X_transformed[j][1],
                       color=color_HAC[DIB], s=symbol_size - 25)

    ax.set_ylabel(method, fontsize=font_size)
    ax.grid()

filename = os.path.join(PicPath, "Figure5_MoreMethods.png")
plt.tight_layout()
plt.savefig(filename)
plt.close(fig)