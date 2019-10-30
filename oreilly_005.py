# simple nrural network

import numpy as np

def hidden_W(x):

    W = np.array([[1, 2, 3],[4, 5, 6]])    
    y = np.dot(x,W)

    return y

x_ = np.array([4,3])

print("y =",hidden_W(x_))
