import numpy as np

def numerical_gradient(f, x):

    delta = 1e-6
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])

    while not it.finished:
        index = it.multi_index

        # calc f(x+h)
        temp = x[index]
        x[index] = temp + delta 
        f1 = f(x)

        # calc f(x-h)
        x[index] = temp - delta 
        f2 = f(x)

        # return array's value
        x[index] = temp

        # calc gradient
        grad[index] = (f1-f2)/(2*delta)
        it.iternext()

    return grad

if __name__ == "__main__":

    def func(x):
        return x[0,0]**2 + x[0,1]**2 + x[1,0]**2 + x[1,1]**2

#    x = np.array([3.0, 4.0])            # [3, 4] makes overflow since grad = (int)/(float) 
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    grad = numerical_gradient(func, x)
    print(grad)
