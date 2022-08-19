# coding: utf-8

# Standard imports
import os
import functools
import operator

# External imports
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms

# Local imports
import deepcs.display
from deepcs.training import train, ModelCheckpoint
from deepcs.testing import test
from deepcs.fileutils import generate_unique_logpath
from deepcs.metrics import BatchAccuracy, BatchCE, BatchF1


class LinearNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearNet, self).__init__()
        self.classifier = nn.Linear(
            functools.reduce(operator.mul, input_size), num_classes
        )

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        y = self.classifier(x)
        return y


def conv_relu_maxpool(dim_in, dim_out):
    return [
        nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(dim_out),
        nn.ReLU(),
        nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(dim_out),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    ]


class ConvNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ConvNet, self).__init__()
        conv_classifier = nn.Sequential(
            *conv_relu_maxpool(1, 12), *conv_relu_maxpool(12, 24)
        )
        dummy_tensor = torch.zeros(1, *input_size)
        output = conv_classifier(dummy_tensor)

        fc_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(functools.reduce(operator.mul, output.shape[1:]), 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes),
        )
        self.classifier = nn.Sequential(conv_classifier, nn.Flatten(), fc_classifier)

    def forward(self, x):
        return self.classifier(x)


# Parameters
dataset_dir = os.path.join(os.path.expanduser("~"), "Datasets", "MNIST")
batch_size = 64
num_workers = 4
n_epochs = 30
learning_rate = 0.01
device = torch.device("cpu")
logdir = generate_unique_logpath("./logs", "linear")
print(f"Logging into {logdir}")

# Datasets
train_valid_dataset = torchvision.datasets.MNIST(
    root=dataset_dir,
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
train_loader = torch.utils.data.DataLoader(
    dataset=train_valid_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=num_workers,
)
test_dataset = torchvision.datasets.MNIST(
    root=dataset_dir,
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=num_workers,
)

# Model
input_size = (1, 28, 28)
# model = LinearNet(input_size, 10)
model = ConvNet(input_size, 10)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
train_fmetrics = {"CE": BatchCE(), "accuracy": BatchAccuracy(), "F1": BatchF1()}
test_fmetrics = {"CE": BatchCE(), "accuracy": BatchAccuracy(), "F1": BatchF1()}

# Display information about the model
summary_text = (
    "Summary of the model architecture\n"
    + "=================================\n"
    + f"{deepcs.display.torch_summarize(model, (batch_size, ) + input_size)}\n"
)

print(summary_text)

# Callbacks
tensorboard_writer = SummaryWriter(log_dir=logdir)
tensorboard_writer.add_text("Experiment summary", deepcs.display.htmlize(summary_text))

model_checkpoint = ModelCheckpoint(model, os.path.join(logdir, "best_model.pt"))

# Train
for e in range(n_epochs):

    train(
        model,
        train_loader,
        loss,
        optimizer,
        device,
        train_fmetrics,
        tensorboard_writer=tensorboard_writer,
    )

    # Compute the metrics
    train_metrics = test(model, train_loader, device, test_fmetrics)
    for bname, bm in test_fmetrics.items():
        bm.tensorboard_write(tensorboard_writer, f"metrics/train_{bname}", e)
    metrics_msg = f"[{e}/{n_epochs}] Train : \n  "
    metrics_msg += "\n  ".join(
        f" {m_name}: {m_value}" for (m_name, m_value) in train_metrics.items()
    )
    print(metrics_msg)

    test_metrics = test(model, test_loader, device, test_fmetrics)
    for bname, bm in test_fmetrics.items():
        bm.tensorboard_write(tensorboard_writer, f"metrics/test_{bname}", e)

    updated = model_checkpoint.update(test_metrics["CE"])
    metrics_msg = f"[{e}/{n_epochs}] Test : \n  "
    metrics_msg += "\n  ".join(
        f" {m_name}: {m_value}"
        + ("[>> BETTER <<]" if updated and m_name == "CE" else "")
        for (m_name, m_value) in test_metrics.items()
    )
    print(metrics_msg)
