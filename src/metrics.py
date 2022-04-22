def get_metric(args):
    if args.type == "binary_accuracy":
        return binary_accuracy


def binary_accuracy(y, yhat):
    y = y.cpu().numpy()
    yhat = yhat.argmax(1).cpu().numpy()

    metrics = {"valid_metric": (y == yhat).mean()}
    return metrics