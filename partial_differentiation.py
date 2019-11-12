import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numerical_differentiation

def f(x, y):
    return x**2+y**2

x_ = np.arange(-10,10,1)
y_ = np.arange(-10,10,1)
x, y = np.meshgrid(x_, y_)
z = f(x, y)

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x, y)")

ax.plot_wireframe(x, y, z)
plt.show()

def f_temp1(x):
    return f(x, 4)

def f_temp2(y):
    return f(3, y)

ans1 = numerical_differentiation.numerical_diff(f_temp1, 3)
ans2 = numerical_differentiation.numerical_diff(f_temp1, 4)

print(ans1)
print(ans2)
