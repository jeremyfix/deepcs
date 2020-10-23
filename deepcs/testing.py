# coding: utf-8

from typing import Any, Callable, Dict, List

import torch
import torch.nn
import torch.utils.data
import torch.optim
from .display import progress_bar


Metric = Callable[[Any, Any], float]

def test(model: torch.nn.Module,
         loader: torch.utils.data.DataLoader,
         device: torch.device,
         metrics: Dict[str, Metric]):
    """
    Test a model by iterating over the loader

    Arguments :

        model     -- A torch.nn.Module object
        loader    -- A torch.utils.data.DataLoader
        device    -- a torch.device object
        metrics   -- the metrics to be evaluated

    Returns :

        A dictionnary with the averaged metrics over the data

    """
    # We disable gradient computation which speeds up the computation
    # and reduces the memory usage
    with torch.no_grad():
        # We enter evaluation mode. This is useless for the linear model
        # but is important with layers such as dropout, batchnorm, ..
        model.eval()
        N = 0
        tot_metrics = {m_name: 0. for m_name in metrics}

        for i, (inputs, targets) in enumerate(loader):

            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            # Accumulate the number of processed samples
            N += inputs.shape[0]

            # For the metrics, we assumed to be averaged over the minibatch
            for m_name, m_f in metrics.items():
                tot_metrics[m_name] += inputs.shape[0] * m_f(outputs, targets).item()
    for m_name, m_v in tot_metrics.items():
        tot_metrics[m_name] = m_v / N
    return tot_metrics
