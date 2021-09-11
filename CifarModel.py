# Solution for CIFAR-10 dataset, using NN with one hidden layer
# Daniel Yakovlev

import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.backends import cudnn
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split
from torch.utils.data import DataLoader

dtype = torch.cuda.FloatTensor
cudnn.benchmark = True


# Hyperparmeters
batch_size = 100
learning_rate1 = 0.01
learning_rate2 = 0.003
epochs1 = 7
epochs2 = 10

# Other constants
input_size = 32*32*3
hidden_size = 64
num_classes = 10

# Computing on GPU if possible
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    # Move tensor(s) to chosen device
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


# Download dataset and split for validation
dataset = CIFAR10(root='data/', train=True, transform=transforms.ToTensor(), download=True)
test_ds = CIFAR10(root='data/', train=False, transform=transforms.ToTensor(), download=True)
train_ds, val_ds = random_split(dataset, [40000, 10000])

train_dl = DeviceDataLoader(DataLoader(train_ds, batch_size, shuffle=True, pin_memory=True), get_default_device())
val_dl = DeviceDataLoader(DataLoader(val_ds, batch_size, shuffle=True, pin_memory=True), get_default_device())


# Define accuracy function (for programmer):
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# Define the model:
class CifarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, xb):
        xb = xb.view(xb.size(0), -1)
        out = self.linear1(xb)
        # activate function:
        out = F.relu(out)
        out = self.linear2(out)
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
to_device(model, get_default_device())

# Training functions:

def evaluate(model, val_dl):
    outputs = [model.validation_step(batch) for batch in val_dl]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_dl, val_dl, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        for batch in train_dl:

            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_dl)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

# Predicting:
def predict_image(img, model):
    xb = to_device(img.unsqueeze(0), get_default_device())
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    return preds[0].item()

history = fit(epochs1, learning_rate1, model, train_dl, val_dl)
history += fit(epochs2, learning_rate2, model, train_dl, val_dl)


# example
img, label = test_ds[0]
plt.imshow(img[0], cmap='gray')
print('Label:', label, ', Predicted:', predict_image(img, model))

print(history)




