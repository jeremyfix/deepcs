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


def train(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    f_loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_metrics={},
    grad_clip=None,
    num_model_args=1,
    num_epoch: int = 0,
    tensorboard_writer=None,
    dynamic_display=True,
):
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
    batch_metrics
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
    for bname, bm in batch_metrics.items():
        bm.reset()

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

        # Update the metrics
        for bname, bm in batch_metrics.items():
            bm(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        try:
            model.penalty().backward()
        except AttributeError:
            pass

        if grad_clip is not None:
            gradnorm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=grad_clip
            )
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar(
                    "grad/norm", gradnorm, num_epoch + (i + 1) / tot_epoch
                )

        optimizer.step()

        # Display status
        if dynamic_display:
            metrics_msg = " | ".join(
                f"{bname}: {bm}" for (bname, bm) in batch_metrics.items()
            )
            progress_bar(i, len(loader), msg=metrics_msg)

        # Write the metrics on the tensorboard if one is provided
        # This is not working as expected
        # as these may call add_scalar which is not expecting
        # a fractional global step

        # if tensorboard_writer is not None:
        #     for bname, bm in batch_metrics.items():
        #         bm.tensorboard_write(
        #             tensorboard_writer,
        #             f"metrics/train_{bname}",
        #             num_epoch + (i + 1) / tot_epoch,
        #         )

    # Compute the value of the batch metrics
    tot_metrics = {}
    for bname, bm in batch_metrics.items():
        tot_metrics[bname] = bm.get_value()

    metrics_msg = "\n  ".join(
        f"{m_name}: {m_value}" for (m_name, m_value) in tot_metrics.items()
    )
    print(f"Sliding window train metrics: \n  {metrics_msg}")

    return tot_metrics


class ModelCheckpoint(object):
    """
    Early stopping callback
    """

    def __init__(
        self,
        model: torch.nn.Module,
        savepath: Union[str, Path],
        min_is_best: bool = True,
    ) -> None:
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
            torch.save(self.model.state_dict(), self.savepath)
            self.best_score = score
            return True
        return False
