import torch
from torch.utils.data import Dataset

class ECGDataset(Dataset):
    def __init__(self, data, labels):
        # Convert the data and labels to torch tensors
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)  # Assume labels are integers (class indices)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]  # Shape: (100, 2200, 8)
        y = self.labels[index]  # Shape: (100, 6) - multi-class labels for each sequence

        # Convert (100, 2200, 8) → (100, 8, 2200) to match input shape (channels, timesteps)
        x = x.permute(0, 2, 1)  # (100, 2200, 8) → (100, 8, 2200)

        # For multi-class classification, you need one label per sample, not one for each sequence
        # We will take the class label of the first sequence (or average it, depending on your task)
        y = y.argmax(dim=1)  # Assuming y is a one-hot vector, take the max index as the class

        return x, y