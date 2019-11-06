import numpy as np

def mean_square_error(test_x, pre_y):
    return 0.5*np.sum((pre_y-test_x)**2)

def cross_entropy_error(test_x, pre_y, one_hot=True):

    if pre_y.ndim == 1:
        test_x = test_x.reshape(1, test_x.size)
        pre_y = pre_y.reshape(1, pre_y.size)

    batch_size = pre_y.shape[0]
    delta = 1e-7

    if one_hot:
        loss = (-1)*np.sum(test_x*np.log(pre_y+delta))/batch_size
    else:
        loss = (-1)*np.sum(np.log(pre_y[np.arange(batch_size), test_x]+delta))/batch_size
    return loss
    

if __name__ == "__main__":

    x = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    y = np.array([0.1, 0.1, 0.5, 0, 0.1, 0, 0, 0, 0, 0.2])
    
    print("ms-error")
    print(mean_square_error(x, y))
    print("ce-error")
    print(cross_entropy_error(x, y))
