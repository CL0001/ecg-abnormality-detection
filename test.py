import os
import numpy as np

# label_path = "./data/ecg_resources/annotations.csv"
# labels = np.loadtxt(label_path, delimiter=',', skiprows=1, dtype=str)
#
# for row in labels:
#     print(row)

rec = np.load("./data/batch-399.npz", allow_pickle=True)
print(rec["labels"][1])
print(rec["signals"].shape)
print(rec["signals"][0][199][7])