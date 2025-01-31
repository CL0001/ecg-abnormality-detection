import torch
import numpy as np

# Trim and cast labels into wanted format
# labels = np.loadtxt("./ecg_resources/annotations.csv", delimiter=',', skiprows=1, dtype=str)
# trimed_labels = np.delete(labels, [0, 1, 2, 3, -1], axis=1)
# casted_labels = trimed_labels.astype(np.float32)
# print(casted_labels)

# Check if the saved .npz format is valid
# sample = np.load("./data/batch-99.npz")
# print(sample.files) # files -> ['signals', 'labels']
# print(sample["signals"].shape, sample["labels"].shape) # shapes -> signals(100, 2200, 8), labels(100, 6)
# print(sample["signals"].dtype, sample["labels"].dtype) # dtypes -> float32, float32

# Check if PyTorch dataset is valid
# data = torch.load('dataset.pt', weights_only=False)
# print(data["data"].shape, data["labels"].shape) # shapes -> data(298, 100, 2200, 8), labels(298, 100, 6)
# print(data["data"][0, :5], data["labels"][0, :5]) # fine

