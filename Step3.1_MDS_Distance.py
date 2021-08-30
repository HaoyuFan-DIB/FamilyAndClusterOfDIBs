# Compare the 1D, 2D, and 3D MDS
# 3 panels, vertically aligned

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from scipy.spatial.distance import squareform, pdist
from CorrDataPath import DataPath, PicPath
import os

# read in r_matrix and create dissimilarity_matrix 1-r_norm
normalized_filename = os.path.join(DataPath, "r_matrix_normalized.csv")
normalized_df = pd.read_csv(normalized_filename)
All_labels = normalized_df.label.to_list()
All_labels = np.array([str(item) for item in All_labels])
normalized_df = normalized_df.drop(columns=["label"])
dissimilarity_matrix = 1 - normalized_df.to_numpy()

# get MDS and distance matrix
Dist_all = {}
for n in [1,2,3]:
    MDS_work = MDS(n_components=n, dissimilarity='precomputed', random_state=31)
    X_transformed = MDS_work.fit_transform(dissimilarity_matrix)
    Dist_all[n] = squareform(pdist(X_transformed, 'euclidean'))

# plot
font_size = 10
symbol_size = 6
label_size = 8

fig = plt.figure(figsize=(2,4), dpi=100)
for n in [1,2,3]:
    ax = fig.add_subplot(3, 1, n)
    real_dist = np.tril(dissimilarity_matrix).reshape(dissimilarity_matrix.shape[0] * dissimilarity_matrix.shape[1])
    prej_dist = np.tril(Dist_all[n]).reshape(Dist_all[n].shape[0] * Dist_all[n].shape[1])
    idx = np.where(prej_dist != 0)
    res_std = np.std(real_dist[idx] - prej_dist[idx])

    ax.scatter(real_dist[idx], prej_dist[idx], color="0.5", s=symbol_size)
    ax.plot([0,1.7],[0,1.7], linestyle="--", color="C0")
    ax.plot([0, 1.7], [0 - res_std, 1.7 - res_std], linestyle="--", color="C3")
    ax.plot([0, 1.7], [0 + res_std, 1.7 + res_std], linestyle="--", color="C3")
    ax.text(1.15, 0.08, "residual = {r:.2f}".format(r=res_std))
    ax.grid()
    ax.set_xlim([0, 1.8])
    ax.set_ylim([0, 2.0])
    ax.set_xticks(np.arange(4) * 0.5)
    ax.set_yticks(np.arange(5)*0.5)
    ax.tick_params(axis='both', which='major', labelsize=label_size)
    ax.set_ylabel("%iD Projected Distances" % n, fontsize=font_size)

    if n == 3:
        ax.set_xlabel("Original Dissimilarity (1 - $\\it{r}_{normalized}$)", fontsize=font_size)

plt.tight_layout()
filename = os.path.join(PicPath, "Figure4_MDS_Dist.png")
plt.savefig(filename)
