import numpy as np

def fast_tanh(x):
    return np.tanh(x)

def softmax(x):
    x_shift = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x_shift)
    return exp / np.sum(exp, axis=-1, keepdims=True)

def forward_nn(x, w1, b1, w2, b2):
    # x: (F,)
    h = np.tanh(np.dot(x, w1) + b1)
    out = np.dot(h, w2) + b2
    return out
