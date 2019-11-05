# soft-max function

import numpy as np

def soft_max(x):

    x_max = max(x)
    a = np.exp(x-x_max)
    a_sum = sum(a)

    y = a/a_sum

    return y

if __name__ == '__main__':

    x_ = np.array([1000, 1200, 1100])
    out = soft_max(x_)

    print(x_)
    print(out)

    print(sum(out))