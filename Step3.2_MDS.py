# Make MDS plot Figure 5 and Figure B2
# All about r_normalized, MDS for r_regular is in Step 3.4
# Do a linear fitting to get a sequence of DIBs
# output X/Y_MDS and the sequence to ClusteringResult

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.linear_model import LinearRegression
from CorrDataPath import DataPath, PicPath
import os

# load r_matrix and clustering result get dissimilarity matrix
normalized_filename = os.path.join(DataPath, "r_matrix_normalized.csv")
normalized_df = pd.read_csv(normalized_filename)
All_labels = normalized_df.label.to_list()
All_labels = np.array([str(item) for item in All_labels])
normalized_df = normalized_df.drop(columns=["label"])
dissimilarity_matrix = 1 - normalized_df.to_numpy()

cluster_filename = os.path.join(DataPath, "ClusteringResult.csv")
cluster_df = pd.read_csv(cluster_filename)
color_HAC = {}
color_Kmeans = {}
for i in range(len(cluster_df)):
    color_HAC[str(cluster_df["DIBs"].iloc[i])] = cluster_df["Norm HAC Color"].iloc[i]
    color_Kmeans[str(cluster_df["DIBs"].iloc[i])] = cluster_df["Norm Kmeans Color"].iloc[i]

# 2D MDS
embedding = MDS(n_components=2, dissimilarity='precomputed', random_state=31)
X_transformed = embedding.fit_transform(dissimilarity_matrix)

# linear regression and DIB sequence
corr_all = X_transformed.T
x, y = corr_all[0], corr_all[1]
coe = np.polyfit(x[0:-4], y[0:-4], 1)

proj_corr = np.zeros(len(X_transformed))
main_axis = np.array([1,coe[0]]) / np.sqrt(1 + coe[0]**2)
for i, corr in enumerate(X_transformed):
    proj_corr[i] = np.dot(main_axis, corr)
idx_sort = np.argsort(proj_corr)
# Print out the sequence
print(", ".join(All_labels[idx_sort]))

idx_dict = {}
for idx_original, idx_sorted in enumerate(idx_sort):
    idx_dict[idx_sorted] = idx_original
idx_original = []
for i in range(len(All_labels)):
    idx_original.append(idx_dict[i])

# save sequence, sort_idx, MDS_X, MDS_Y to ClusteringResult
cluster_df["MDS_X"] = x
cluster_df["MDS_Y"] = y
cluster_df["Sequence_Idx"] = idx_original # as report in the paper， e.g. 7559 is the first one
cluster_df["Sort_Idx"] = idx_sort # DIB_labels[sort_idx] would give the right DIB sequence
with open(os.path.join(DataPath, "ClusteringResult.csv"), "w") as f:
    f.write(cluster_df.to_csv(index=False))



# Plot
# Small plot for text
symbol_size = 50
line_width = 1.5
fig = plt.figure(figsize=(8, 6), dpi=200)
ax = fig.add_subplot(1, 1, 1)

for i, DIB in enumerate(All_labels):
    if DIB not in ["5780", "5797", "6284", "6196", "4963", "4984"]:
        if color_HAC[DIB] == color_Kmeans[DIB]:
            ax.scatter(X_transformed[i][0], X_transformed[i][1],
                       color=color_HAC[DIB], s=symbol_size)
        else:
            ax.scatter(X_transformed[i][0], X_transformed[i][1],
                       color=color_Kmeans[DIB], s=symbol_size)
            ax.scatter(X_transformed[i][0], X_transformed[i][1],
                       color=color_HAC[DIB], s=symbol_size - 40)


for DIB in ["5780", "6284", "5797", "4963", "4984"]:
    idx = np.where(All_labels == DIB)
    ax.scatter(X_transformed[idx[0][0]][0], X_transformed[idx[0][0]][1],
               color=color_Kmeans[DIB], marker="x",
               s=symbol_size + 50, linewidth=line_width)

for DIB in ["6196"]:
    idx = np.where(All_labels == DIB)
    ax.scatter(X_transformed[idx[0][0]][0], X_transformed[idx[0][0]][1],
               color=color_Kmeans[DIB], marker="x",
               s=symbol_size + 50, linewidths=line_width + 1.0)
    ax.scatter(X_transformed[idx[0][0]][0], X_transformed[idx[0][0]][1],
               color=color_HAC[DIB], marker="x",
               s=symbol_size, linewidths=line_width - 0.5)

# add linear fit result
xrange = [-0.6, 0.95]
ax.plot([xrange[0], xrange[1]],
        [xrange[0]*coe[0]+coe[1], xrange[1]*coe[0]+coe[1]],
        linestyle="--", color="0.5")

# Add two circles to show 0.05 and 0.15 radius
ax.add_patch(plt.Circle((1., -0.5), 0.05, color="C0", alpha=0.6, ec=None))
ax.add_patch(plt.Circle((1., -0.5), 0.15, color="C0", alpha=0.2, ec=None))
ax.set_ylabel("2D MDS", fontsize=16)
ax.set_xlim([-0.91, 1.322])
ax.set_ylim([-0.848, 0.940])
ax.set_xticks(np.arange(start=-0.75, stop=1.5, step=0.25))
ax.set_yticks(np.arange(start=-0.75, stop=0.76, step=0.25))
ax.grid()
filename = os.path.join(PicPath, "Figure5_MDS.png")
plt.tight_layout()
plt.savefig(filename)
plt.close(fig)

# Redo the plot for the page-wide version
# All the same, just remove the regression line
fig = plt.figure(figsize=(8, 6), dpi=200)
ax = fig.add_subplot(1, 1, 1)

for i, DIB in enumerate(All_labels):
    if DIB not in ["5780", "5797", "6284", "6196", "4963", "4984"]:
        if color_HAC[DIB] == color_Kmeans[DIB]:
            ax.scatter(X_transformed[i][0], X_transformed[i][1],
                       color=color_HAC[DIB], s=symbol_size)
        else:
            ax.scatter(X_transformed[i][0], X_transformed[i][1],
                       color=color_Kmeans[DIB], s=symbol_size)
            ax.scatter(X_transformed[i][0], X_transformed[i][1],
                       color=color_HAC[DIB], s=symbol_size - 40)


for DIB in ["5780", "6284", "5797", "4963", "4984"]:
    idx = np.where(All_labels == DIB)
    ax.scatter(X_transformed[idx[0][0]][0], X_transformed[idx[0][0]][1],
               color=color_Kmeans[DIB], marker="x",
               s=symbol_size + 50, linewidth=line_width)

for DIB in ["6196"]:
    idx = np.where(All_labels == DIB)
    ax.scatter(X_transformed[idx[0][0]][0], X_transformed[idx[0][0]][1],
               color=color_Kmeans[DIB], marker="x",
               s=symbol_size + 50, linewidths=line_width + 1.0)
    ax.scatter(X_transformed[idx[0][0]][0], X_transformed[idx[0][0]][1],
               color=color_HAC[DIB], marker="x",
               s=symbol_size, linewidths=line_width - 0.5)

# Add two circles to show 0.05 and 0.15 radius
ax.add_patch(plt.Circle((1., -0.5), 0.05, color="C0", alpha=0.6, ec=None))
ax.add_patch(plt.Circle((1., -0.5), 0.15, color="C0", alpha=0.2, ec=None))
ax.set_ylabel("2D MDS", fontsize=16)
ax.set_xlim([-0.91, 1.322])
ax.set_ylim([-0.848, 0.940])
ax.set_xticks(np.arange(start=-0.75, stop=1.5, step=0.25))
ax.set_yticks(np.arange(start=-0.75, stop=0.76, step=0.25))
ax.grid()
filename = os.path.join(PicPath, "FigureB2_MDS.png")
plt.tight_layout()
plt.savefig(filename)
plt.close(fig)
