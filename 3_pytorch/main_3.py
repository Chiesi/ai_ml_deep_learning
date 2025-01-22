# Remember to run the following commands!
# pip install pandas
# pip install matplotlib
# pip install scikit-learn
# pip install jupyterlab
# pip install ipywidgets
# jupyter labextension enable widgetsnbextension
# pip install kagglehub

# pip install torch
# pip install torchvision

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch import nn, optim
from sklearn.metrics import accuracy_score

# Define the tensor transformation and load training and validation data
# The LeCun domain is having issues today, so we'll cheat a little bit
root_dir = './'
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,),)
    ]
)
trainset = datasets.MNIST(
    root=root_dir,
    download=True,
    train=True,
    transform=transform
)
valset = datasets.MNIST(
    root=root_dir,
    download=True,
    train=False,
    transform=transform
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# We'll build another multy-layer perceptron, with 784 input units (images are 28x28)
# Can you guess the rest of the architecture?
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

# Inspect the data whenever you can
images, labels = next(iter(trainloader))
pixels = images[0][0]
plt.imshow(pixels, cmap='gray')
plt.show()

# Let's define a loss function!
# We'll use cross-entropy, as we want the error in PD for all labels - again, Adam
# optimizer
criterion = nn.CrossEntropyLoss()
images = images.view(images.shape[0], -1)
output = model(images)
loss = criterion(output, labels)
optimizer = optim.Adam(model.parameters(), lr=0.003)

# Let's start the training proper
epochs = 15
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        images = images.view(images.shape[0], -1)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        # This is back-propagation explicitly being invoked!
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader) * 100))
        
# Let's finally test the accuracy of the validation data
valloader = torch.utils.data.DataLoader(
    valset,
    batch_size=valset.data.shape[0],
    shuffle=True
)
val_images, val_labels = next(iter(valloader))
val_images = val_images.view(val_images.shape[0], -1)
predictions = model (val_images)
predicted_labels = np.argmax(predictions.detach().numpy(), axis=1)
accuracy_score(val_labels.detach().numpy(), predicted_labels)

# Let's save this model too
torch.save(model, './model/torchpy_mnist_model.pt')
