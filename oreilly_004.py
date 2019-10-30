# step function

import numpy as np
import matplotlib.pyplot as plt

def step_function(x_):

    y = x_>0
    y = y.astype(np.int64)

    return y

def sigmoid(x):

    y = 1/(1+np.exp(-1*x))

    return y

def ReLu(x):

    y = np.maximum(0, x)

    return y

if __name__ == '__main__':

    x_ = np.arange(-5, 5, 0.01)
    y1 = step_function(x_)
    y2 = sigmoid(x_)
    y3 = ReLu(x_)

    plt.plot(x_, y1, linestyle='--', color='b')
    plt.plot(x_, y2, linestyle=':', color='r')
    plt.plot(x_, y3, color='g')
    #plt.ylim(-0.1,1.1)
    plt.show()
