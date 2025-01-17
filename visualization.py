import os
import numpy as np
import matplotlib.pyplot as plt

data_path = "./data"

disease_count = {
    '1dAVb': 0,
    'RBBB': 0,
    'LBBB': 0,
    'SB': 0,
    'AF': 0,
    'ST': 0
}

shape_count = {}

batch_files = os.listdir(data_path)

for batch_file in batch_files:
    batch_data = np.load(os.path.join(data_path, batch_file), allow_pickle=True)
    signals = batch_data['signals']
    labels = batch_data['labels']

    batch_shape = signals.shape

    if batch_shape not in shape_count:
        shape_count[batch_shape] = 0
    shape_count[batch_shape] += 1

    for i in range(len(labels)):
        if labels[i, -6] == '1':
            disease_count['1dAVb'] += 1
        if labels[i, -5] == '1':
            disease_count['RBBB'] += 1
        if labels[i, -4] == '1':
            disease_count['LBBB'] += 1
        if labels[i, -3] == '1':
            disease_count['SB'] += 1
        if labels[i, -2] == '1':
            disease_count['AF'] += 1
        if labels[i, -1] == '1':
            disease_count['ST'] += 1

fig, ax1 = plt.subplots(figsize=(10, 6))
diseases = list(disease_count.keys())
counts = list(disease_count.values())
ax1.bar(diseases, counts, color='skyblue')
ax1.set_xlabel('Disease')
ax1.set_ylabel('Count of Signals')
ax1.set_title('Count of Signals for Each Disease')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("./graphs/disease-statistics.png")
plt.show()

fig, ax2 = plt.subplots(figsize=(10, 6))
shapes = [str(shape) for shape in shape_count.keys()]
shape_counts = list(shape_count.values())
ax2.bar(shapes, shape_counts, color='salmon')
ax2.set_xlabel('Signal Shape')
ax2.set_ylabel('Count of Batches')
ax2.set_title('Count of Batches by Signal Shape')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("./graphs/batch-dimensions.png")
plt.show()