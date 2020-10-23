# coding: utf-8

import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms
from deepcs.training import train
from deepcs.testing import test


class LinearNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearNet, self).__init__()
        self.classifier = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        y = self.classifier(x)
        return y

dataset_dir = os.path.join(os.path.expanduser("~"), 'Datasets', 'FashionMNIST')
batch_size = 64
num_workers = 4
device = torch.device('cpu')

train_valid_dataset = torchvision.datasets.FashionMNIST(root=dataset_dir,
                                                        train=True,
                                                        download=True,
                                                        transform=torchvision.transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_valid_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=num_workers)
test_dataset = torchvision.datasets.FashionMNIST(root=dataset_dir,
                                                 train=False,
                                                 download=True,
                                                 transform=torchvision.transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          pin_memory=True,
                                          num_workers=num_workers)

model = LinearNet(28*28, 10)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
metrics = {'CE': loss}

train(model, train_loader, loss, optimizer, device, metrics)
train_metrics = test(model, train_loader, device, metrics)
test_metrics = test(model, test_loader, device, metrics)
print(train_metrics, test_metrics)
