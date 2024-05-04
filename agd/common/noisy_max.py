import numpy as np

def noisy_max(candidate, lmbda, bmin=False):
    scores = np.array(candidate)
    noise = np.random.exponential(lmbda, size=len(scores))

    # choose the minimum?
    if bmin:
        scores *= -1.0
        noise *= -1.0

    # add noise
    scores += noise
    idx = np.argmax(scores)

    return idx, candidate[idx]