import os
import numpy as np
import torch

input_path = "./data"

data = []
labels = []

for i, file in enumerate(os.listdir(input_path)):
    print(f"{i}. processing file: {file}")
    file_path = os.path.join(input_path, file)

    record = np.load(file_path)
    signals = record["signals"]
    label = record["labels"]

    data.append(signals)
    labels.append(label)

data = np.array(data)
labels = np.array(labels)

torch.save({'data': data, 'labels': labels}, 'dataset.pt')