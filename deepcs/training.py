# coding: utf-8

from typing import Any, Callable, Dict, List

import torch
import torch.nn
import torch.utils.data
import torch.optim
from .display import progress_bar


Metric = Callable[[Any, Any], float]

def train(model: torch.nn.Module,
          loader: torch.utils.data.DataLoader,
          f_loss: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          device: torch.device,
          metrics: Dict[str, Metric]):
    """
        Train a model for one epoch, iterating over the loader
        using the f_loss to compute the loss and the optimizer
        to update the parameters of the model.

        Arguments :
        model     -- A torch.nn.Module object
        loader    -- A torch.utils.data.DataLoader
        f_loss    -- The loss function, i.e. a loss Module
        optimizer -- A torch.optim.Optimzer object
        device    -- A torch.device
        metrics

        Returns :

    """

    # We enter train mode. This is useless for the linear model
    # but is important for layers such as dropout, batchnorm, ...
    model.train()
    N = 0
    tot_metrics = {m_name: 0. for m_name in metrics}

    for i, (inputs, targets) in enumerate(loader):

        inputs, targets = inputs.to(device), targets.to(device)

        # Compute the forward propagation
        outputs = model(inputs)

        loss = f_loss(outputs, targets)

        # Accumulate the number of processed samples
        N += inputs.shape[0]

        # For the metrics, we assumed to be averaged over the minibatch
        for m_name, m_f in metrics.items():
            tot_metrics[m_name] += inputs.shape[0] * m_f(outputs, targets).item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        try:
            model.penalty().backward()
        except AttributeError:
            pass
        optimizer.step()

        # Display status
        metrics_msg = ",".join(f"{m_name}: {m_value/N:.4}" for(m_name, m_value) in tot_metrics.items())
        progress_bar(i, len(loader), msg = metrics_msg)

    # Normalize the metrics over the whole dataset
    for m_name, m_v in tot_metrics.items():
        tot_metrics[m_name] = m_v / N

    return tot_metrics
