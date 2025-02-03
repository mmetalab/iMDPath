import torch
from torch import nn, optim

def train_model(model, dataloader, epochs, criterion, optimizer, device):
    """
    Generic training loop for models.

    Args:
        model: PyTorch model to be trained.
        dataloader: DataLoader for training data.
        epochs: Number of training epochs.
        criterion: Loss function.
        optimizer: Optimizer for training.
        device: Computation device ('cuda' or 'cpu').

    Returns:
        Trained model.
    """
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader)}")
    return model
