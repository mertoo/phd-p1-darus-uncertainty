import numpy as np

def create_windows(data, history=20, horizon=10):
    """
    Convert time-series into sequences:
    past history → future horizon.
    """
    X, y = [], []

    for i in range(len(data) - history - horizon):
        past = data[i:i+history]
        future = data[i+history:i+history+horizon]
        X.append(past)
        y.append(future)

    return np.array(X), np.array(y)
