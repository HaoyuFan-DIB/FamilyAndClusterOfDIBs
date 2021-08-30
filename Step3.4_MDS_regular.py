# Make Figure C2, MDS plot from r_regular
# Same as Step3.2 except input, no linear regression, no output to ClusteringResult

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.linear_model import LinearRegression
from CorrDataPath import DataPath, PicPath
import os

# load r_matrix and clustering result get dissimilarity matrix
regular_filename = os.path.join(DataPath, "r_matrix_regular.csv")
regular_df = pd.read_csv(regular_filename)
All_labels = regular_df.label.to_list()
All_labels = np.array([str(item) for item in All_labels])
regular_df = regular_df.drop(columns=["label"])
dissimilarity_matrix = 1 - regular_df.to_numpy()

cluster_filename = os.path.join(DataPath, "ClusteringResult.csv")
cluster_df = pd.read_csv(cluster_filename)

# MDS
embedding = MDS(n_components=2, dissimilarity='precomputed', random_state=31)
X_transformed = embedding.fit_transform(dissimilarity_matrix)

# Plot
symbol_size = 50
line_width = 1.5
fig = plt.figure(figsize=(8, 6), dpi=200)
ax = fig.add_subplot(1, 1, 1)

for i, DIB in enumerate(All_labels):
    print(DIB, X_transformed[i][0], X_transformed[i][1])

    colors = [cluster_df["Reg HAC Color"].iloc[i], cluster_df["Norm HAC Color"].iloc[i],
              cluster_df["Reg Kmeans Color"].iloc[i], cluster_df["Norm Kmeans Color"].iloc[i]]
    colors = np.unique(colors)

    if DIB not in ["5780", "6196"]:
        if DIB not in ["5797", "6284", "6196", "4963", "4984"]:
            marker = "o"
        else:
            marker = "x"

        ax.scatter(X_transformed[i][0], X_transformed[i][1], marker=marker,
                   color=colors[0], s=symbol_size)
        if len(colors) == 2:
            ax.scatter(X_transformed[i][0], X_transformed[i][1], marker=marker,
                       color=colors[1], s=symbol_size - 40)
    else:
        ax.scatter(X_transformed[i][0], X_transformed[i][1],
                   color=colors[0], marker="x",
                   s=symbol_size + 50, linewidths=line_width + 1.0)
        ax.scatter(X_transformed[i][0], X_transformed[i][1],
                   color=colors[1], marker="x",
                   s=symbol_size, linewidths=line_width - 0.5)

# Add two circles to show 0.05 and 0.15 radius
ax.add_patch(plt.Circle((0.4, -0.2), 0.025, color="C0", alpha=0.6, ec=None))
ax.add_patch(plt.Circle((0.4, -0.2), 0.05, color="C0", alpha=0.2, ec=None))
#ax.set_ylabel("2D MDS", fontsize=16)
ax.set_xlim([-0.29, 0.55])
ax.set_ylim([-0.29, 0.38])
ax.grid()
filename = os.path.join(PicPath, "FigureC2_MDS_Regular.png")
plt.tight_layout()
plt.savefig(filename)
plt.close(fig)
