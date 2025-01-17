import os
import numpy as np
import wfdb

data_path = "./data/ecg_resources/data"
label_path = "./data/ecg_resources/annotations.csv"
output_path = "./data"

batch = []
labels = np.loadtxt(label_path, delimiter=',', skiprows=1, dtype=str) # Label headers id_exam,id_patient,age,sex,1dAVb,RBBB,LBBB,SB,AF,ST,date_exam

# TNMG1_N1 -> TNMG2_N1 -> TNMG3_N1 -> ...
file_group = 1

while file_group <= 39_999:
    record_name = f"TNMG{file_group}_N1"
    print(f"processing record: {record_name}")

    try:
        record_signal = wfdb.rdrecord(os.path.join(data_path, record_name)).p_signal
        batch.append(record_signal)
    except Exception as e:
        print(f"can't load file: {record_name}, error: {e}\n")
        file_group += 1
        continue

    if len(batch) == 100:
        numpy_array = np.array(batch, dtype=object)
        np.savez(os.path.join(output_path, f"batch-{int(file_group/100)}.npz"), signals=numpy_array, labels=labels[:100])
        print(f"batch saved\n")
        batch = []
        labels = labels[100:]

    file_group += 1