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
        B = targets.shape[0]
        self.cum_metric += B * self.metric(predictions, targets).item()
        self.num_samples += B

    def get_value(self):
        if self.num_samples == 0:
            raise ZeroDivisionError
        return self.cum_metric / self.num_samples


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
        elif len(predictions.shape) == 2:
            # Multi class case
            if self.tp is None:
                self.num_classes = predictions.shape[1]
                self.tp = [0 for k in range(self.num_classes)]
                self.fp = [0 for k in range(self.num_classes)]
                self.fn = [0 for k in range(self.num_classes)]

            preds = predictions.argmax(axis=1)  # (B, )
            for k in range(self.num_classes):
                preds_k = (preds == k).double()
                targs_k = (targets == k).double()
                self.tp[k] += (preds_k * targs_k).sum().item()
                self.fp[k] += (preds_k * (1.0 - targs_k)).sum().item()
                self.fn[k] += ((1 - preds_k) * targs_k).sum().item()
        else:
            raise ValueError(
                "Do not know how to handle predictions with more than 2 dimensions"
            )

    def get_value(self):
        if self.num_classes == 2:
            return self.tp / (self.tp + 0.5 * (self.fp + self.fn))
        else:
            return [
                tp / (tp + 0.5 * (fp + fn))
                for tp, fp, fn in zip(self.tp, self.fp, self.fn)
            ]

    def __str__(self):
        return ",".join(f"{val:.2f}" for val in self.get_value())


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
