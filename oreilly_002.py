# perceptron with bias

import numpy as np

def B_and(X):

    b = 0.7
    w_ = np.array([0.5, 0.5])

    out = sum(X*w_)-b

    if out >0:
        return 1
    else:
        return 0

def B_nand(X):

    b = 0.7
    w_ = np.array([0.5, 0.5])

    out = (-1)*(sum(X*w_)-b)

    if out >0:
        return 1
    else:
        return 0

def B_or(X):

    b = 0.7
    w_ = np.array([1.0, 1.0])

    out = sum(X*w_)-b

    if out >0:
        return 1
    else:
        return 0

if __name__ == "__main__":

    print("AND")
    print(B_and(np.array([0,0])))
    print(B_and(np.array([0,1])))
    print(B_and(np.array([1,0])))
    print(B_and(np.array([1,1])))
    print("NAND")
    print(B_nand(np.array([0,0])))
    print(B_nand(np.array([0,1])))
    print(B_nand(np.array([1,0])))
    print(B_nand(np.array([1,1])))
    print("OR")
    print(B_or(np.array([0,0])))
    print(B_or(np.array([0,1])))
    print(B_or(np.array([1,0])))
    print(B_or(np.array([1,1])))
