import torch.nn as nn
import torch.nn.utils.rnn


class GenericBatchMetric:
    def __init__(self, metric):
        """
        Args:
            metric: a batch averaged metric to accumulate
        """
        self.metric = metric
        self.cum_metric = 0.0
        self.num_samples = 0

    def reset(self):
        self.cum_metric = 0
        self.num_samples = 0

    def __call__(self, predictions, targets):
        """
        predictions: (B, *)
        targets : (B, *)
        """
        # We suppose is batch averaged
        if isinstance(targets, torch.nn.utils.rnn.PackedSequence):
            B = targets.unsorted_indices.shape[0]
        else:
            B = targets.shape[0]
        self.cum_metric += B * self.metric(predictions, targets).item()
        self.num_samples += B

    def get_value(self):
        if self.num_samples == 0:
            raise ZeroDivisionError
        return self.cum_metric / self.num_samples

    def __str__(self):
        return f"{self.get_value():.3f}"

    def tensorboard_write(self, writer, prefix, global_step):
        writer.add_scalar(prefix, self.get_value(), global_step)


def BatchCE():
    return GenericBatchMetric(nn.CrossEntropyLoss(reduction="mean"))


def BatchAccuracy():
    return GenericBatchMetric(accuracy)


class BatchF1:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = None
        self.fp = None
        self.fn = None
        self.num_classes = 0

    def __call__(self, predictions, targets):
        """
        predictions: (B, ) logits or probabilities
        targets : (B,)

        or

        predictions: (B, K) logits or probabilities for multiple classes
        targets: (B, )

        """
        if len(predictions.shape) == 1:
            if self.tp is None:
                self.tp = self.fp = self.fn = 0
                self.num_classes = 2
            preds = predictions > 0.5
            self.tp += (preds * targets).sum()
            self.fp += (preds * (1 - targets)).sum()
            self.fn += ((1 - preds) * targets).sum()
        elif len(predictions.shape) >= 2:
            # Multi class case, possibly multi-dimensions
            if self.tp is None:
                self.num_classes = predictions.shape[1]
                self.tp = [0 for k in range(self.num_classes)]
                self.fp = [0 for k in range(self.num_classes)]
                self.fn = [0 for k in range(self.num_classes)]

            if len(predictions.shape) > 2:
                assert len(predictions.shape) == (len(targets.shape) + 1)
                # predictions is expected to be (B, C, d1, d2, ..)
                # targets is expected to be (B, d1, d2, ..)
                B = targets.shape[0]
                targets = targets.view(B, -1)
                predictions = predictions.view(B, self.num_classes, -1)

            preds = predictions.argmax(axis=1)  # (B, )
            for k in range(self.num_classes):
                preds_k = (preds == k).double()
                targs_k = (targets == k).double()
                self.tp[k] += (preds_k * targs_k).sum().item()
                self.fp[k] += (preds_k * (1.0 - targs_k)).sum().item()
                self.fn[k] += ((1 - preds_k) * targs_k).sum().item()

    def get_value(self):
        if self.num_classes == 2:
            return (
                self.tp / (self.tp + 0.5 * (self.fp + self.fn))
                if self.tp != 0 or self.fp != 0 or self.fn != 0
                else 1.0
            )  # else there are either no samples or just TN
        else:
            return [
                tp / (tp + 0.5 * (fp + fn))
                if tp != 0 or fp != 0 or fn != 0
                else 1.0  # else there are either no samples or just TN
                for tp, fp, fn in zip(self.tp, self.fp, self.fn)
            ]

    def __str__(self):
        return ",".join(f"{val:.2f}" for val in self.get_value())

    def tensorboard_write(self, writer, prefix, global_step):
        writer.add_scalar(
            prefix + "_macro", sum(self.get_value()) / self.num_classes, global_step
        )


def accuracy(probabilities, targets):
    """
    Computes the accuracy. Works with either PackedSequence or Tensor
    """
    with torch.no_grad():
        if isinstance(probabilities, torch.nn.utils.rnn.PackedSequence):
            probs = probabilities.data
        else:
            probs = probabilities
        if isinstance(targets, torch.nn.utils.rnn.PackedSequence):
            targ = targets.data
        else:
            targ = targets
        return (probs.argmax(axis=-1) == targ).double().mean()
