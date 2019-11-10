import numpy as np
import matplotlib.pyplot as plt

def numerical_diff(f, x):
    delta = 10e-6
    df = (f(x+delta)-f(x-delta))/(2*delta)
    return df

def f(x):
    return x**2

if __name__ == "__main__":

    x = np.arange(-2, 5, 0.1)
    y1 = f(x)
    df = numerical_diff(f, 2)
    y2 = df*x - 4

    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.show()
