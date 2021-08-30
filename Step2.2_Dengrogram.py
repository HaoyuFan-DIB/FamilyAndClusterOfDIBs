# Clustering DIBs and output results to a .csv file
# Require data outputs of Step1_GetCorrData and Step2_Clustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.cluster import AgglomerativeClustering
from CorrDataPath import PicPath, DataPath
from scipy.cluster.hierarchy import dendrogram

# Get data
normalized_filename = os.path.join(DataPath, "r_matrix_normalized.csv")
normalized_df = pd.read_csv(normalized_filename)
normalized_df = normalized_df.loc[~normalized_df["label"].str.contains("N")]
DIB_labels = normalized_df.label.to_list()
DIB_labels = np.array([str(item) for item in DIB_labels])
normalized_df = normalized_df.drop(columns=["label", "NH2", "NCH", "NCN", "NC2"])
normalized_matrix = normalized_df.to_numpy()

regular_filename = os.path.join(DataPath, "r_matrix_regular.csv")
regular_df = pd.read_csv(regular_filename)
regular_df = regular_df.loc[~regular_df["label"].str.contains("N")]
regular_df = regular_df.drop(columns=["label", "NH2", "NCH", "NCN", "NC2"])
regular_matrix = regular_df.to_numpy()

# clustering result to set the colours
cluster_filename = os.path.join(DataPath, "ClusteringResult.csv")
cluster_df = pd.read_csv(cluster_filename)
cluster_df = cluster_df.loc[~cluster_df["DIBs"].str.contains("N")]

#####################
# r_norm
#####################
agglomerative = AgglomerativeClustering(distance_threshold=0,
                                        n_clusters=None)
agglomerative.labels_ = DIB_labels
model = agglomerative.fit(normalized_matrix)

counts = np.zeros(model.children_.shape[0])
n_samples = len(model.labels_)
for i, merge in enumerate(model.children_):
    current_count = 0
    for child_idx in merge:
        if child_idx < n_samples:
            current_count += 1  # leaf node
        else:
            current_count += counts[child_idx - n_samples]
    counts[i] = current_count
linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

# notes:
# * rows in linkage_matrix correspond to "inverted U" links that connect clusters
# * rows are ordered by increasing distance
# * if the colors of the connected clusters match, use that color for link
base_leaf_colors = {}
HAC_colors = cluster_df["Norm HAC Color"].to_numpy()
for key, value in zip(DIB_labels, HAC_colors):
    base_leaf_colors[key] = value

link_cols = {}
for i, i12 in enumerate(linkage_matrix[:,:2].astype(int)):
    c1, c2 = (link_cols[x] if x > len(linkage_matrix) else base_leaf_colors[DIB_labels[x]] for x in i12)
    link_cols[i + 1 + len(linkage_matrix)] = c1 if c1 == c2 else "C0"

fig = plt.figure(figsize=(10, 4), dpi=200)
ax1 = fig.add_subplot(1, 1, 1)
dendrogram(linkage_matrix, labels=DIB_labels, leaf_font_size=8, leaf_rotation=60,
           link_color_func=lambda x:link_cols[x])

fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
filename = os.path.join(PicPath, "Figure3_Dendrogram_Norm.png")
plt.savefig(filename)


#####################
# r_regular
#####################
model = agglomerative.fit(regular_matrix)

counts = np.zeros(model.children_.shape[0])
n_samples = len(model.labels_)
for i, merge in enumerate(model.children_):
    current_count = 0
    for child_idx in merge:
        if child_idx < n_samples:
            current_count += 1  # leaf node
        else:
            current_count += counts[child_idx - n_samples]
    counts[i] = current_count
linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

base_leaf_colors = {}
HAC_colors = cluster_df["Reg HAC Color"].to_numpy()
for key, value in zip(DIB_labels, HAC_colors):
    base_leaf_colors[key] = value

link_cols = {}
for i, i12 in enumerate(linkage_matrix[:,:2].astype(int)):
    c1, c2 = (link_cols[x] if x > len(linkage_matrix) else base_leaf_colors[DIB_labels[x]] for x in i12)
    link_cols[i + 1 + len(linkage_matrix)] = c1 if c1 == c2 else "C0"

fig = plt.figure(figsize=(10,4), dpi=200)
ax1 = fig.add_subplot(1, 1, 1)
dendrogram(linkage_matrix, labels=DIB_labels, leaf_font_size=8, leaf_rotation=60,
           link_color_func=lambda x:link_cols[x])

fig.tight_layout()
filename = os.path.join(PicPath, "FigureC1_Dendrogram_Reg.png")
plt.savefig(filename)