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
          metrics: Dict[str, Metric],
          grad_clip = None,
          num_model_args=1):
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
        grad_clip
        num_model_args

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
        if num_model_args == 1:
            outputs = model(inputs)
        else:
            outputs = model(inputs, targets)

        loss = f_loss(outputs, targets)

        # Accumulate the number of processed samples
        if isinstance(inputs, torch.Tensor):
            batch_size = inputs.shape[0]
        elif isinstance(inputs, torch.nn.utils.rnn.PackedSequence):
            batch_size = inputs.data.shape[0]  # considering batch_first
        N += batch_size

        # For the metrics, we assumed to be averaged over the minibatch
        for m_name, m_f in metrics.items():
            tot_metrics[m_name] += batch_size * m_f(outputs, targets).item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        try:
            model.penalty().backward()
        except AttributeError:
            pass

        if grad_clip is not None:
            torch.nn.utils.clip_grad_value_(model.parameters(),
                                            clip_value=grad_clip)
            
        optimizer.step()

        # Display status
        metrics_msg = ",".join(f"{m_name}: {m_value/N:.4}" for(m_name, m_value) in tot_metrics.items())
        progress_bar(i, len(loader), msg = metrics_msg)

    # Normalize the metrics over the whole dataset
    for m_name, m_v in tot_metrics.items():
        tot_metrics[m_name] = m_v / N

    print("Train metrics :     {}".format(" | ".join([f"{m_name}: {m_value}" for m_name, m_value in tot_metrics.items()])))

    return tot_metrics
