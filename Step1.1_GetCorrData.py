# Create r_normalized and r_regular matrix and save to .csv file
import numpy as np
import pandas as pd
from CorrData import CorrData
from CorrDataPath import DataPath
import os

DIB_all = [4429, 4501, 5780, 5797, 6196, 6379, 6284, 6613, 7224, 5849,
           6353, 4963, 4984, 6203, 6660, 6376, 6439, 6702, 6699, 5828,
           5766, 6185, 6089, 6377, 6993, 6367, 6330, 6622, 7367, 6449,
           5546, 6212, 6324, 6065, 6116, 6553, 4726, 6108, 6729, 5923,
           6520, 7562, 6362, 5705, 5418, 7559, 6234, 6397, 6270,
           4762, 5793, 5545, 5512, 6445, "NH2", "NCH", "NCN", "NC2"]
DIB_label = [str(item) for item in DIB_all]

data = CorrData()
r_matrix_regular = np.ones((len(DIB_all), len(DIB_all)))
r_matrix_normalized = np.ones_like(r_matrix_regular)
for i, DIB1 in enumerate(DIB_all):
    print(DIB1)
    for j, DIB2 in enumerate(DIB_all):
        r_matrix_normalized[i,j] = data.getPearsonr(DIB1, DIB2,
                                                    allowed_labels1=["D", "S"],
                                                    allowed_labels2=["D", "S"],
                                                    normalized=True, mode="linear")
        r_matrix_regular[i,j] = data.getPearsonr(DIB1, DIB2,
                                                    allowed_labels1=["D", "S"],
                                                    allowed_labels2=["D", "S"],
                                                    normalized=False, mode="linear")

with open(os.path.join(DataPath, "r_matrix_regular.csv"), "w") as f:
    f.write(",".join(["label"] + DIB_label) + "\n")
    for i in range(len(DIB_all)):
        data2write = [str(item) for item in list(r_matrix_regular[i,:])]
        f.write(",".join([DIB_label[i]] + data2write) + "\n")

with open(os.path.join(DataPath, "r_matrix_normalized.csv"), "w") as f:
    f.write(",".join(["label"] + DIB_label) + "\n")
    for i in range(len(DIB_all)):
        data2write = [str(item) for item in list(r_matrix_normalized[i,:])]
        f.write(",".join([DIB_label[i]] + data2write) + "\n")
