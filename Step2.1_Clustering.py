# Clustering DIBs and output results to a .csv file
# Requires r_value matrix

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans, AgglomerativeClustering
from CorrDataPath import DataPath


def GroupAndColor(DIB_labels, Y_Cluster):
    # assume four clusters
    color_map = ["C2", "C3", "C4", "C1"]
    cluster_id = np.asarray([-99] * len(DIB_labels))
    color = np.asarray(["C0"] * len(DIB_labels))
    for i in range(4):
        idx = Y_Cluster == i
        DIB_now = DIB_labels[idx]
        if "6284" in DIB_now:
            cluster_id_now = 0  # sigma
        elif "5797" in DIB_now:
            cluster_id_now = 2  # zeta
        elif "4984" in DIB_now:
            cluster_id_now = 3  # C2
        else:
            cluster_id_now = 1  # intermediate
        cluster_id[idx] = [cluster_id_now] * len(DIB_now)
        color[idx] = [color_map[cluster_id_now]] * len(DIB_now)
    return cluster_id, color


##########################################
# Get data
##########################################
n_cluster = 4    # sigma, inter, zeta, C2

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


##########################################
# HAC Clustering
##########################################
agglomerative = AgglomerativeClustering(n_clusters=n_cluster)

y_norm = agglomerative.fit_predict(normalized_matrix)
id_norm_HAC, color_norm_HAC = GroupAndColor(DIB_labels, y_norm)

y_reg = agglomerative.fit_predict(regular_matrix)
id_reg_HAC, color_reg_HAC = GroupAndColor(DIB_labels, y_reg)


##########################################
# K-Means
##########################################
kmeans = KMeans(n_clusters=n_cluster, n_init=20)
kmeans.fit(normalized_matrix)
y_norm = kmeans.predict(normalized_matrix)
id_norm_Kmeans, color_norm_Kmeans = GroupAndColor(DIB_labels, y_norm)

kmeans = KMeans(n_clusters=n_cluster, n_init=20)
kmeans.fit(regular_matrix)
y_reg = kmeans.predict(regular_matrix)
id_reg_Kmeans, color_reg_Kmeans = GroupAndColor(DIB_labels, y_reg)


##########################################
# Save_Result
##########################################
df = pd.DataFrame(list(zip(DIB_labels,
                           id_norm_HAC, color_norm_HAC,
                           id_norm_Kmeans, color_norm_Kmeans,
                           id_reg_HAC, color_reg_HAC,
                           id_reg_Kmeans, color_reg_Kmeans)),
                  columns=["DIBs",
                           "Norm HAC Cluster", "Norm HAC Color",
                           "Norm Kmeans Cluster", "Norm Kmeans Color",
                           "Reg HAC Cluster", "Reg HAC Color",
                           "Reg Kmeans Cluster", "Reg Kmeans Color"])

# Also add clustering result for known species here
NX_add = ["NH2", "NCH", "NCN", "NC2"]
cluster_add = [4]*len(NX_add)
color_add = ["b"]*len(NX_add)
df_add = pd.DataFrame(list(zip(NX_add,
                               cluster_add, color_add,
                               cluster_add, color_add,
                               cluster_add, color_add,
                               cluster_add, color_add)),
                      columns=["DIBs",
                               "Norm HAC Cluster", "Norm HAC Color",
                               "Norm Kmeans Cluster", "Norm Kmeans Color",
                               "Reg HAC Cluster", "Reg HAC Color",
                               "Norm Kmeans Cluster", "Norm Kmeans Color"])
df = df.append(df_add, ignore_index=True)

with open(os.path.join(DataPath, "ClusteringResult.csv"), "w") as f:
    f.write(df.to_csv(index=False))
