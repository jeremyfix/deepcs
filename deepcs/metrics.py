import torch.nn.utils.rnn

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
