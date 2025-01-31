from torch import nn
from torch.nn import functional as F

class ECGCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(ECGCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1600, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(35200, 128)  # Adjust based on your input size
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        print(x.shape)
        #x = x.view(-1, 32 * 550 * 50)  # Flatten the tensor
        x = x.flatten()
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x