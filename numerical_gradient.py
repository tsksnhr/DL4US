import numpy as np

def numerical_gradient(f, x):

    delta = 1e-6
    grad = np.zeros_like(x)

    for index in range(np.size(x)):
        
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

    return grad

if __name__ == "__main__":

    def func(x):
        return x[0]**2 + x[1]**2

    x = np.array([3.0, 4.0])            # [3, 4] makes overflow since grad = (int)/(float) 
    grad = numerical_gradient(func, x)
    print(grad)
