# DL libraries
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import datasets, transforms

# Computational libraries
import math
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

T = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

"""
Note: You can update the path to point to the directory containing `MNIST` 
directory to avoid downloading the MNIST data again.
"""
mnist_train = datasets.MNIST("./", train=True, download=True, transform=T)
mnist_test = datasets.MNIST("./", train=False, download=True, transform=T)

"""
if you feel your computer can't handle too much data, you can reduce the batch
size to 64 or 32 accordingly, but it will make training slower. 

We recommend sticking to 128 but do choose an appropriate batch size that your
computer can manage. The training phase tends to require quite a bit of memory.
"""
train_loader = torch.utils.data.DataLoader(mnist_train, shuffle=True, batch_size=256)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=10000)

def get_accuracy(scores, labels):
    ''' accuracy metric '''
    _, predicted = torch.max(scores.data, 1)
    correct = (predicted == labels).sum().item()   
    return correct / scores.size(0)


class RawCNN(nn.Module):
    def __init__(self, classes):
        super().__init__()
        """
        classes: integer that corresponds to the number of classes for MNIST
        """
        self.net1 = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.MaxPool2d((2,2)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, (3, 3)),
            nn.MaxPool2d((2,2)),
            nn.LeakyReLU(0.1),
        )

        self.net2 = nn.Sequential(
            nn.Linear(1600, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, classes),
            nn.Softmax(1)
        )
        
    def forward(self, x):      
        x = self.net1(x)
        x = x.view(-1, 64 * 5 * 5) # Flattening
        x = self.net2(x)
        return x

def train_model_mnist(loader):
    """
    PARAMS
    loader: the data loader used to generate training batches
    
    RETURNS
        the final trained vanilla model on MNIST
    """

    """
    YOUR CODE HERE
    
    - create the model, loss, and optimizer
    """
    model = RawCNN(10)
    optimiser = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    epoch_losses = []
    for i in range(10):
        epoch_loss = 0
        
        for idx, data in enumerate(loader):
            x, y = data

            """
            YOUR CODE HERE
            
            - reset the optimizer
            - perform forward pass
            - compute loss
            - perform backward pass
            """
            optimiser.zero_grad()
            y_pred = model.forward(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimiser.step()

            # COMPUTE STATS
            epoch_loss += loss.sum()

        epoch_loss = epoch_loss / len(loader)
        epoch_losses.append(epoch_loss)
        print ("Epoch: {}, Loss: {}".format(i, epoch_loss))
        

    return model, epoch_losses

def test_model_mnist(model: RawCNN, test_loader):
    with torch.no_grad():
        # torch.save(model.state_dict(), "./saved_model.pt")
        # model.eval()
        for i, data in enumerate(test_loader):
            x, y = data
            pred = model(x)
            acc = get_accuracy(pred, y)
            print(f"vanilla acc: {acc}")
