import os
import numpy as np
import wfdb

data_path = "./ecg_resources/data"
label_path = "./ecg_resources/annotations.csv"
output_path = "./data"

min_length = 2200 # this is length of the shortest timestamp
batch = []

labels = np.loadtxt(label_path, delimiter=',', skiprows=1, dtype=str)
trimed_labels = np.delete(labels, [0, 1, 2, 3, -1], axis=1) # keeps only these columns -> ['1dAVb' 'RBBB' 'LBBB' 'SB' 'AF' 'ST']
casted_labels = trimed_labels.astype(np.float32)

# labels preview
# [[0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0.]
#  [1. 0. 0. 0. 0. 0.]
#  ...
#  [0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0.]]

filename = "batch"
file_group = 1

os.makedirs(output_path, exist_ok=True)

while file_group <= 39_999:
    record_name = f"TNMG{file_group}_N1"
    print(f"Processing record: {record_name}")
    try:
        record_signal = wfdb.rdrecord(os.path.join(data_path, record_name)).p_signal

        if record_signal.shape[0] > min_length:
            trimmed_signal = record_signal[:min_length, :]
        else:
            trimmed_signal = record_signal

        batch.append(trimmed_signal)

    except Exception as e:
        print(f"Can't load file: {record_name}, error: {e}\n")
        file_group += 1
        continue

    if len(batch) == 100:
        numpy_array = np.array(batch, dtype=np.float32)
        trimmed_labels = casted_labels[:100]

        np.savez(os.path.join(output_path, f"{filename}-{int(file_group / 100)}.npz"),
                 signals=numpy_array, labels=trimmed_labels)

        print(f"Batch saved as {filename}-{int(file_group / 100)}.npz\n")

        batch = []
        casted_labels = casted_labels[100:]

    file_group += 1