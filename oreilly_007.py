# soft-max function

import numpy as np

def soft_max(x):

    a = np.exp(x)
    a_sum = sum(a)

    y = a/a_sum

    return y

if __name__ == '__main__':

    x_ = np.array([1, 2, 3])
    out = soft_max(x_)

    print(x_)
    print(out)
