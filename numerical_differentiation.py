import numpy as np
import matplotlib.pyplot as plt

def numerical_diff(f, x):
    delta = 10e-6
    df = (f(x+delta)-f(x-delta))/(2*delta)
    return df

def f(x):
    return x**2

if __name__ == "__main__":

    x = np.arange(-2, 10, 0.1)
    y1 = f(x)
    df1 = numerical_diff(f, 2)
    df2 = numerical_diff(f, 4)
    y_1 = df1*x - 4
    y_2 = df2*x - 16

    plt.plot(x, y1)
    plt.plot(x, y_1)
    plt.plot(x, y_2)
    plt.show()
