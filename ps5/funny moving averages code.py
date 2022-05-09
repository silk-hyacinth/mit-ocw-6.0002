def moving_average(y, window_length):
    return [sum(vals) / len(vals) for vals in [(y[max(i - window_length + 1, 0): i + 1]) for i in range(len(y))]]