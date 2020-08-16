import numpy as np 

def relu(Z):
    A = np.maximum(0, Z)
    assert(A.shape == Z.shape)
    return A, Z

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A, Z

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    assert(dZ.shape == Z.shape)
    return dZ


def sigmoid_backward(dA, Z):
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    assert(dZ.shape == Z.shape)
    return dZ