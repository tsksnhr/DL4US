# test mlp by sample waight
import os
import pickle
import numpy as np

import step_func
import soft_max_func
import load_mnist_dataset

def get_data():
    ((train_image, train_label), (test_image, test_label)) = load_mnist_dataset.load_mnist_dataset(one_hot=False, normarize=True, flatten=True)
    return test_image, test_label

def init_network():
    
    path = os.path.join('C:\\', 'Users', 'chest', 'Desktop', 'python', 'Learn_DL4US', 'datasets', 'sample_weight.pkl')

    if os.path.exists(path):
        with open(path, 'rb') as fp:
            network = pickle.load(fp)
    else:
        print("Invalid path.")

    return network

def predict(network, x_):

    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    y1 = b1+np.dot(x_, W1)
    z1 = step_func.sigmoid(y1)

    y2 = b2+np.dot(z1, W2)
    z2 = step_func.sigmoid(y2)

    y3 = b3+np.dot(z2, W3)
    z3 = soft_max_func.soft_max(y3)

    return z3


if __name__ == "__main__":

    test_image, test_label = get_data()
    network = init_network()

    accuracy_cnt = 0
    batch_size = 100

    for i in range(0, len(test_image), batch_size):
        test_image_batch = test_image[i:i+batch_size]
        out_batch = predict(network, test_image_batch)
        
        p = np.argmax(out_batch, axis=1)

        accuracy_cnt += np.sum(p == test_label[i:i+batch_size])

    print("Accuracy =", accuracy_cnt/len(test_image))
