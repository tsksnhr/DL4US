import numpy as np
import matplotlib.pyplot as plt

import loss_func as lf
import step_func as sf
import soft_max_func as smf
import numerical_gradient as n_gradient

class SimpleNet:
    def __init__(self):
        self.w = np.random.rand(2, 3)

    def predict(self, train_x):
        return np.dot(train_x, self.w)

    def loss(self, train_x, test_x):
        pre_y = self.predict(train_x)
        activated_y = smf.soft_max(pre_y)
        loss = lf.cross_entropy_error(test_x, activated_y)
        return loss

if __name__ == "__main__":
    net = SimpleNet()
    print("W =", net.w)

    train_x = np.array([1.0, 4.0])
    pre_y = net.predict(train_x)
    pre_label = np.argmax(pre_y)
    print("pre_y =", pre_y)
    print("max index =", pre_label)

    test_x = np.array([0, 1, 0])
    loss = net.loss(train_x, test_x)
    print("loss =", loss)

    def dummy_f(w):
        return net.loss(train_x, test_x)

    grad = n_gradient.numerical_gradient(dummy_f, net.w)
    print("loss gradient =", grad)
