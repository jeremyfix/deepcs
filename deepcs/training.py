# coding: utf-8

# Standard imports
from typing import Any, Callable, Dict, List, Union
from pathlib import Path
# External imports
import torch
import torch.nn
import torch.utils.data
import torch.optim
# Local imports
from .display import progress_bar


Metric = Callable[[Any, Any], float]

def train(model: torch.nn.Module,
          loader: torch.utils.data.DataLoader,
          f_loss: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          device: torch.device,
          metrics: Dict[str, Metric],
          grad_clip=None,
          num_model_args=1,
          num_epoch: int= 0,
          tensorboard_writer=None,
          dynamic_display=True):
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
        num_epoch -- The number of this epoch, used for determining
                     the current epoch for the tensorboard writer
        tensorboard_writer

        Returns :

    """

    # We enter train mode. This is useless for the linear model
    # but is important for layers such as dropout, batchnorm, ...
    model.train()
    N = 0
    tot_metrics = {m_name: 0. for m_name in metrics}

    # Get the total number of minibatches, i.e. of sub epochs
    tot_epoch = len(loader)

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
            # The minibatch size can be obtained as the number of samples for
            # the first time sample
            batch_size = inputs.batch_sizes[0]
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
            gradnorm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                      max_norm=grad_clip)
            tensorboard_writer.add_scalar(f'grad/norm', gradnorm, num_epoch + (i+1)/tot_epoch)

        optimizer.step()

        # Display status
        metrics_msg = ",".join(f"{m_name}: {m_value/N:.4}" for(m_name, m_value) in tot_metrics.items())
        if dynamic_display:
            progress_bar(i, len(loader), msg = metrics_msg)

        # Write the metrics on the tensorboard if one is provided
        if tensorboard_writer is not None:
            for m_name, m_value in tot_metrics.items():
                tensorboard_writer.add_scalar(f'metrics/train_{m_name}', m_value/N, num_epoch + (i+1)/tot_epoch)

    # Normalize the metrics over the whole dataset
    for m_name, m_v in tot_metrics.items():
        tot_metrics[m_name] = m_v / N

    print("Train metrics :     {}".format(" | ".join([f"{m_name}: {m_value}" for m_name, m_value in tot_metrics.items()])))

    return tot_metrics


class ModelCheckpoint(object):
    """
    Early stopping callback
    """

    def __init__(self,
                 model: torch.nn.Module,
                 savepath: Union[str, Path],
                 min_is_best: bool =True) -> None:
        self.model = model
        self.savepath = savepath
        self.best_score = None
        if min_is_best:
            self.is_better = self.lower_is_better
        else:
            self.is_better = self.higher_is_better

    def lower_is_better(self, score):
        return self.best_score is None or score < self.best_score

    def higher_is_better(self, score):
        return self.best_score is None or score > self.best_score

    def update(self, score):
        if self.is_better(score):
            torch.save(self.model.state_dict(),
                       self.savepath)
            self.best_score = score
            return True
        return False
