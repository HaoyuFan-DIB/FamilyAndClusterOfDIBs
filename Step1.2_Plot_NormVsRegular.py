# Creat Figure 1, Norm vs Regular
# Histogram on top for r_normalized and r_regular
# Three pairs of scatter plots for middle and bottom panels
# Require r_matrix and CorrData

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from CorrData import CorrData
from CorrDataPath import DataPath, PicPath

# Basic inputs
DIBpairs = [[4984, 7559], [5418, 7562], [6089, 6379]]
y_label_size = 12
x_label_size = 12
tick_label_size = 12


# Read-in Data, drop N(X)
normalized_df = pd.read_csv(os.path.join(DataPath, "r_matrix_normalized.csv"))
normalized_df = normalized_df.loc[~normalized_df["label"].str.contains("N")]
normalized_df = normalized_df.drop(columns=["label", "NH2", "NCH", "NCN", "NC2"])
normalized_matrix = normalized_df.to_numpy()

regular_df = pd.read_csv(os.path.join(DataPath, "r_matrix_regular.csv"))
regular_df = regular_df.loc[~regular_df["label"].str.contains("N")]
regular_df = regular_df.drop(columns=["label", "NH2", "NCH", "NCN", "NC2"])
regular_matrix = regular_df.to_numpy()

for i in range(len(normalized_matrix)):
    normalized_matrix[i][i] = 1
    regular_matrix[i][i] = 1

# plot
# initialize the figure
fig = plt.figure(figsize=(10, 7), dpi=200)

# histogram
# get number of DIB pairs in each bin
bins = np.arange(-0.7, 1.01, 0.1)
count_normalized, count_regular = [], []
for item in bins:
    count_normalized.append(len(normalized_matrix[
                        (normalized_matrix >= item - 0.05) &
                        (normalized_matrix < np.min([item + 0.05, 0.999]))
                        ]) / 2)
    count_regular.append(len(regular_matrix[
                          (regular_matrix >= item - 0.05) &
                          (regular_matrix < np.min([item + 0.05, 0.999]))
                          ]) / 2)

# make the plot to ax1
ax1 = fig.add_subplot(3, 1, 1)
ax1.set_xlabel("hist", fontsize=12)
ax1.bar(bins-0.02, count_regular, width=0.04,
        color="0.4", linewidth=1, edgecolor="k",
        label = "$\\it{r}_{regular}$")
ax1.bar(bins+0.02, count_normalized, width=0.04,
        color="0.8", linewidth=1, edgecolor="k",
        label = "$\\it{r}_{normalized}$")
ax1.grid(axis='y', alpha=0.75)
ax1.set_xticks(np.arange(-0.8, 1.01, 0.2))
ax1.set_xticklabels(["{a:.1f}".format(a=item) for item in np.arange(-0.8, 1.01, 0.2)])
ax1.set_yticks([0, 150, 300, 450])
ax1.set_xlabel("Pearson Correlation Coefficient (" + "$\\it{r}$" + " values)", fontsize=x_label_size)
ax1.set_ylabel("No. of DIB Pairs", fontsize=y_label_size)
ax1.tick_params(axis='both', which='major', labelsize=tick_label_size)
ax1.legend()

# correlation plots
dataset = CorrData()

for i, DIBpair in enumerate(DIBpairs):
    # regular/upper panel
    ax = fig.add_subplot(3, 3, i+4)
    df = dataset.getDoubleData(wave1=DIBpair[0], wave2=DIBpair[1], normalized=False)
    r = dataset.getPearsonr(wave1=DIBpair[0], wave2=DIBpair[1], normalized=False)
    # scatter and error bars
    for x, y, err_x, err_y in zip(df.EW1.to_list(), df.EW2.to_list(), df.err1.to_list(), df.err2.to_list()):
        plt.scatter(x, y, s=20, color="0.6")
        plt.plot([x - err_x, x + err_x], [y, y], color="k")
        plt.plot([x, x], [y - err_y, y + err_y], color="k")

    # r_values
    x_range, y_range = ax.get_xlim(), ax.get_ylim()
    plt.text(0.7 * x_range[1] + 0.3 * x_range[0], 0.05 * y_range[1] + 0.95 * y_range[0], "r = {r:.3f}".format(r=r))
    # x and y labels
    if i==0:
        ax.set_ylabel("Regular\n", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=tick_label_size)
    ax.grid()

    # normalized/lower panel
    ax = fig.add_subplot(3, 3, i+7)
    df = dataset.getDoubleData(wave1=DIBpair[0], wave2=DIBpair[1], normalized=True)
    r = dataset.getPearsonr(wave1=DIBpair[0], wave2=DIBpair[1], normalized=True)
    for x, y, err_x, err_y in zip(df.EW1.to_list(), df.EW2.to_list(), df.err1.to_list(), df.err2.to_list()):
        plt.scatter(x, y, s=20, color="0.6")
        plt.plot([x - err_x, x + err_x], [y, y], color="k")
        plt.plot([x, x], [y - err_y, y + err_y], color="k")

    x_range, y_range = ax.get_xlim(), ax.get_ylim()
    if i == 2:
        plt.text(0.7 * x_range[1] + 0.3 * x_range[0], 0.05 * y_range[1] + 0.95 * y_range[0], "r = {r:.3f}".format(r=r))
    else:
        plt.text(0.7 * x_range[1] + 0.3 * x_range[0], 0.9 * y_range[1] + 0.1 * y_range[0], "r = {r:.3f}".format(r=r))
    if i == 0:
        ax.set_ylabel("Normalized\n", fontsize=y_label_size)
    ax.set_xlabel("W({d1}) v.s W({d2})".format(d1=DIBpair[0], d2=DIBpair[1]), fontsize=x_label_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_label_size)
    ax.grid()

# finishing up
fig.tight_layout()
filename = os.path.join(PicPath, "Figure1_Norm_VS_Regular.png")
plt.savefig(filename)
plt.show()
