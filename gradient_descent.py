import numpy as np
import numerical_gradient
import matplotlib.pyplot as plt


def gradient_descent(f, init_x, lr=0.01, step_num=1000):

    x = init_x
    traj = []

    for dummy in range(step_num):
        traj.append(x)
        grad = numerical_gradient.numerical_gradient(f, x)
        x = x - grad*lr

    return x, np.array(traj)


if __name__ == "__main__":

    x = [8, 3]

    xm = np.arange(-10, 10, 0.1)
    ym = np.arange(-10, 10, 0.1)

    def func(x):
        return x[0]**2 + 8*x[0] + x[1]**2

    out, traj = gradient_descent(func, x, lr=0.2)
    print(traj)

    xmesh, ymesh = np.meshgrid(xm, ym)
#    xx = np.r_[xmesh.reshape(1, -1), ymesh.reshape(1, -1)]
    xx = [xmesh, ymesh]
    z = func(xx)
    
    plt.scatter(x[0], x[1])
    plt.scatter(out[0], out[1], color="r")
    plt.plot(traj[:, 0], traj[:, 1], color="k", linewidth=1.0)
    plt.contour(xmesh, ymesh, z, colors="k")
    plt.show()
