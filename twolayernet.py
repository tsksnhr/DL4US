import numpy as np
import step_func
import soft_max_func
import loss_func
import numerical_gradient

class TwoLayersNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.parameters = {}
        self.parameters["w1"] = weight_init_std*np.random.randn(input_size, hidden_size)
        self.parameters["b1"] = np.zeros(input_size)
        self.parameters["w2"] = weight_init_std*np.random.randn(hidden_size, output_size)
        self.parameters["b2"] = np.zeros(hidden_size)

    def predict(self, x_train):
        w1, w2 = self.parameters["w1"], self.parameters["w2"]
        b1, b2 = self.parameters["b1"], self.parameters["b2"]

        y1 = np.dot(x_train, w1) + b1
        z1 = step_func.sigmoid(y1)
        y2 = np.dot(z1, w2) + b2
        z2 = soft_max_func.soft_max(y2)
        pred_y = z2

        return pred_y

    def loss(self, x_train, x_test):
        pred_y = self.predict(x_train)
        return loss_func.cross_entropy_error(x_test, pred_y)

    def acc_chk(self, x_train, x_test):
        pred_y = self.predict(x_train)
        y_chk = np.argmax(pred_y)
        x_chk = np.argmax(x_test)

        acc_cnt = 0
        if y_chk == x_chk:
            acc_cnt += 1

        acc = acc_cnt/float(np.size(x_test))
        return acc

    def numerical_gradient(self, x_train, x_test):

        def dummy_func(w):
            return self.loss(x_train, x_test)

        grads = {}

        grads["w1"] = numerical_gradient.numerical_gradient(dummy_func, self.parameters["w1"])
        grads["w2"] = numerical_gradient.numerical_gradient(dummy_func, self.parameters["w2"])
        grads["b1"] = numerical_gradient.numerical_gradient(dummy_func, self.parameters["b1"])
        grads["b2"] = numerical_gradient.numerical_gradient(dummy_func, self.parameters["b2"])

        return grads

if __name__ == "__main__":
    sample_net = TwoLayersNet(input_size=784, hidden_size=100, output_size=10)

    print(sample_net.parameters["w1"].shape)
    print(sample_net.parameters["w2"].shape)
    print(sample_net.parameters["b1"].shape)
    print(sample_net.parameters["b2"].shape)