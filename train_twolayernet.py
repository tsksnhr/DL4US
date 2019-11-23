import numpy as np
from load_mnist_dataset import load_mnist_dataset
from twolayernet import TwoLayersNet
import matplotlib.pyplot as plt

(train_images, train_label), (test_images, test_label) =\
    load_mnist_dataset(flatten=False)

train_loss_list = []

iters_num = 10000
train_size = train_images.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayersNet(input_size=784, hidden_size=100, output_size=10)

for i in range(iters_num):

    batch_mask = np.random.choice(train_size, batch_size)
    image_batch = train_images[batch_mask]
    label_batch = train_label[batch_mask]

    grad = network.numerical_gradient(image_batch, label_batch)

    for key in ("w1", "b1", "w2", "b2"):
        network.parameters[key] -= grad[key]*learning_rate

    loss = network.loss(image_batch, label_batch)
    train_loss_list.append(loss)

plt.plot(iters_num, train_loss_list)
plt.show()
