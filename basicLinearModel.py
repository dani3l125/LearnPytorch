import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time


# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43],
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70],
                   [74, 66, 43],
                   [91, 87, 65],
                   [88, 134, 59],
                   [101, 44, 37],
                   [68, 96, 71],
                   [73, 66, 44],
                   [92, 87, 64],
                   [87, 135, 57],
                   [103, 43, 36],
                   [68, 97, 70]],
                  dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119],
                    [57, 69],
                    [80, 102],
                    [118, 132],
                    [21, 38],
                    [104, 118],
                    [57, 69],
                    [82, 100],
                    [118, 134],
                    [20, 38],
                    [102, 120]],
                   dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

train_ds = TensorDataset(inputs, targets)

batch_size = 5

train_dl = DataLoader(train_ds, batch_size, shuffle=True)

model = nn.Linear(3, 2)

loss_fn = F.mse_loss

opt = torch.optim.SGD(model.parameters(), lr=1e-5)


# Utility function to train the model
def fit(num_epochs, model, loss_fn, opt, train_dl):
    # Repeat for given number of epochs
    for epoch in range(num_epochs):

        # Train with batches of data
        for input, target in train_dl:
            # 1. Generate predictions
            pred = model(input)

            # 2. Calculate loss
            loss = loss_fn(pred, target)

            # 3. Compute gradients
            loss.backward()

            # 4. Update parameters using gradients
        opt.step()

            # 5. Reset the gradients to zero
        opt.zero_grad()

        # Print the progress
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

start = time.time()
fit(500, model, loss_fn, opt, train_dl)
end = time.time()

print()
print(model(inputs))
print()
print(targets)
print()
print(end-start)
