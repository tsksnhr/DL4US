# 3 leyer perceptron

import numpy as np
import step_func

def init_network():

    network = {}

    network['W1'] = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    network['b1'] = np.array([0.2, 0.2, 0.2])
    network['W2'] = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    network['b2'] = np.array([0.4, 0.4])
    network['W3'] = np.array([[0.6, 0.5], [0.8, 0.9]])
    network['b3'] = np.array([0.3, 0.3])

    return network

def forward(network, x):

    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    y1 = b1+np.dot(x_, W1)
    z1 = step_func.sigmoid(y1)

    y2 = b2+np.dot(z1, W2)
    z2 = step_func.sigmoid(y2)

    y3 = b3+np.dot(z2, W3)
    z3 = identify_func(y3)

    return z3

def identify_func(x):

    return x

x_ = np.array([1.0, 1.0])
network = init_network()
output = forward(network, x_)

print(output)
