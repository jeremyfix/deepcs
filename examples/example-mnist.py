# coding: utf-8

import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms
import deepcs.display
from deepcs.training import train
from deepcs.testing import test
from deepcs.fileutils import generate_unique_logpath
from deepcs.metrics import accuracy


class LinearNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearNet, self).__init__()
        self.classifier = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        y = self.classifier(x)
        return y


# Parameters
dataset_dir = os.path.join(os.path.expanduser("~"), 'Datasets', 'MNIST')
batch_size = 64
num_workers = 4
n_epochs = 30
learning_rate = 0.01
device = torch.device('cpu')
logdir = generate_unique_logpath('./logs', 'linear')

# Datasets
train_valid_dataset = torchvision.datasets.MNIST(root=dataset_dir,
                                                 train=True,
                                                 download=True,
                                                 transform=torchvision.transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_valid_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=num_workers)
test_dataset = torchvision.datasets.MNIST(root=dataset_dir,
                                          train=False,
                                          download=True,
                                          transform=torchvision.transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          pin_memory=True,
                                          num_workers=num_workers)

# Model
model = LinearNet(28*28, 10)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
metrics = {
    'CE': loss,
    'accuracy': accuracy
}

# Display information about the model
summary_text = "Summary of the model architecture\n"+ \
        "=================================\n" + \
        f"{deepcs.display.torch_summarize(model)}\n"

print(summary_text)

# Callbacks
tensorboard_writer   = SummaryWriter(log_dir = logdir)
tensorboard_writer.add_text("Experiment summary", deepcs.display.htmlize(summary_text))

# Train
for e in range(n_epochs):

    train(model, train_loader, loss, optimizer, device, metrics)

    # Compute the metrics
    train_metrics = test(model, train_loader, device, metrics)
    test_metrics = test(model, test_loader, device, metrics)

    # Write the metrics to the tensorboard
    for m_name, m_value in train_metrics.items():
        tensorboard_writer.add_scalar(f'metrics/train_{m_name}', m_value, e)
    for m_name, m_value in test_metrics.items():
        tensorboard_writer.add_scalar(f'metrics/test_{m_name}', m_value, e)
