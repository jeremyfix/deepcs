

def accuracy(probabilities, targets):
    return (probabilities.argmax(axis=1) == targets).double().mean()
