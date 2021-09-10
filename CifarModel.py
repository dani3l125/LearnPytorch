# my implementation for logistic regression classifier based on CIFAR-10 dataset:
# used : https://jovian.ai/aakashns/mnist-logistic-minimal

import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split
from torch.utils.data import DataLoader

# Hyperparmeters
batch_size = 100
learning_rate = 0.003
epochs = 50


# Other constants
input_size = 32*32*3
num_classes = 10

# Download dataset and split for validation
dataset = CIFAR10(root='data/', train=True, transform=transforms.ToTensor(), download=True)
test_ds = CIFAR10(root='data/', train=False, transform=transforms.ToTensor(), download=True)
train_ds, val_ds = random_split(dataset, [40000, 10000])

train_subset = DataLoader(train_ds, batch_size, shuffle=True)
val_subset = DataLoader(val_ds, batch_size, shuffle=True)

# Define accuracy function (for programmer):
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# Define the model:
class CifarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, xb):
        xb = xb.reshape(-1, input_size)
        # try : xb = torch.flatten(xb)
        out = self.linear(xb)
        return out

    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

model = CifarModel()

# Training functions:


count = 0

def evaluate(model, val_subset):
    outputs = [model.validation_step(batch) for batch in val_subset]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_subset, val_subset, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        for batch in train_subset:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_subset)
        model.epoch_end(epoch, result)
        history.append(result)
    return history


history = fit(epochs, learning_rate, model, train_subset, val_subset)

print(history)




