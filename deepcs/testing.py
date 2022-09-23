# coding: utf-8

# Standard imports
from typing import Any, Callable, Dict, List

# External imports
import torch
import torch.nn
import torch.utils.data
import torch.optim

# Local imports
from .display import progress_bar


Metric = Callable[[Any, Any], float]


def test(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    metrics: Dict[str, Metric],
    num_model_args: int = 1,
    dynamic_display: bool = True,
):
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
    for bname, bm in metrics.items():
        bm.reset()
    # We disable gradient computation which speeds up the computation
    # and reduces the memory usage
    with torch.no_grad():
        # We enter evaluation mode. This is useless for the linear model
        # but is important with layers such as dropout, batchnorm, ..
        model.eval()
        N = 0
        tot_metrics = {m_name: 0.0 for m_name in metrics}

        for i, (inputs, targets) in enumerate(loader):

            inputs, targets = inputs.to(device), targets.to(device)

            # Compute the forward propagation
            if num_model_args == 1:
                outputs = model(inputs)
            else:
                outputs = model(inputs, targets)

            # Accumulate the number of processed samples
            if isinstance(inputs, torch.Tensor):
                batch_size = inputs.shape[0]
            elif isinstance(inputs, torch.nn.utils.rnn.PackedSequence):
                # The minibatch size can be obtained as the number of samples for
                # the first time sample
                batch_size = inputs.batch_sizes[0]
            N += batch_size

            # Update the metrics
            for bname, bm in metrics.items():
                bm(outputs, targets)

            # Display status
            if dynamic_display:
                progress_bar(i, len(loader))
    # Compute the value of the batch metrics
    tot_metrics = {}
    for bname, bm in metrics.items():
        tot_metrics[bname] = bm.get_value()
    return tot_metrics
