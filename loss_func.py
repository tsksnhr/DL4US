import numpy as np

def mean_square_error(test_x, pre_y):
    return 0.5*sum((pre_y-test_x)**2)

def cross_entropy_error(test_x, pre_y):
    delta = 1e-7
    return (-1)*sum(test_x*np.log(pre_y+delta))

if __name__ == "__main__":

    x = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    y = np.array([0.1, 0.1, 0.5, 0, 0.1, 0, 0, 0, 0, 0.2])
    
    print("ms-error")
    print(mean_square_error(x, y))
    print("ce-error")
    print(cross_entropy_error(x, y))
