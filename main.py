import torch
from model import ECGCNN
from torch import nn, optim
from dataset import ECGDataset
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

# Instance of model, loss function, and optimizer
model = ECGCNN().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=0.001)

# Create dataloader
loaded_data = torch.load('dataset.pt', weights_only=False)

dataset = ECGDataset(loaded_data['data'], loaded_data['labels'])  # shape -> data(298, 100, 2200, 8), labels(100)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
dataloader_length = len(dataloader)

# Training loop
epochs = 300
for epoch in range(epochs):
    running_loss = 0.0  # To accumulate loss for each epoch
    for i, (inputs, labels) in enumerate(dataloader):
        # Move data to the correct device
        inputs, labels = inputs.to(device), labels.to(device)

        # Reshape inputs to [batch_size, 8, 2200]
        inputs = inputs.permute(0, 3, 2, 1).reshape(-1, 8, 2200)

        # Forward pass
        outputs = model(inputs)
        print(outputs.shape, labels.shape)
        loss = loss_fn(outputs, labels.argmax(dim=1))  # Convert one-hot to class indices

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

    # Print average loss for the epoch
    epoch_loss = running_loss / dataloader_length
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')

# Evaluation
model.eval()  # Set the model to evaluation mode
with torch.inference_mode():
    correct = 0
    total = 0
    for inputs, labels in dataloader:
        # Move data to the correct device
        inputs, labels = inputs.to(device), labels.to(device)

        # Reshape inputs to [batch_size, 8, 2200]
        inputs = inputs.view(-1, 8, 2200)

        # Forward pass
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        # Calculate accuracy
        total += labels.size(0)
        correct += (predicted == labels.argmax(dim=1)).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total:.2f} %')

# Save trained model
torch.save(model.state_dict(), "ecg_classifier.pth")
print("Training complete, model saved!")
