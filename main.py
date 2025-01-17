from model import SimpleECGClassifier
from torch import nn
from torch import optim

num_leads = 12
signal_length = 5000
num_classes = 5
model = SimpleECGClassifier()

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(),
                       lr=0.001)