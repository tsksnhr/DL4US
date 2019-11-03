import numpy as np
import os
import gzip
import pickle
import matplotlib.pyplot as plt

def load_image_mnist(value):

    # file path
    file_path = os.path.join(datasets_path, value)

    # open gz file
    with gzip.open(file_path, "rb") as fp:
        image = np.frombuffer(fp.read(), np.uint8, offset=16)

    image = image.reshape(-1,28**2)

    return image

def load_label_mnist(value):

    # file path
    file_path = os.path.join(datasets_path, value)

    # open gz file
    with gzip.open(file_path, "rb") as fp:
        label = np.frombuffer(fp.read(), np.uint8, offset=8)

    return label

if __name__ == "__main__":

    # set file-name
    file_dic = {}
    file_dic['train_images'] = 'train-images-idx3-ubyte.gz'
    file_dic['train_labels'] = 'train-labels-idx1-ubyte.gz'
    file_dic['test_images'] = 't10k-images-idx3-ubyte.gz'
    file_dic['test_labels'] = 't10k-labels-idx1-ubyte.gz'

    # directory path  (choose your username)
    # datasets_path = os.path.join('C:\\', 'Users', 'toshi', 'Desktop', 'python', 'Learn_DL4US', 'datasets')
    datasets_path = os.path.join('C:\\', 'Users', 'chest', 'Desktop', 'python', 'Learn_DL4US', 'datasets')

    # show dataset as a list
    print(load_image_mnist(file_dic['train_images']))
    print(load_label_mnist(file_dic['train_labels']))
    print(load_image_mnist(file_dic['test_images']))
    print(load_label_mnist(file_dic['test_labels']))

    # assign datasets to dictionary
    dataset_dic = {}
    dataset_dic['train_images'] = load_image_mnist(file_dic['train_images'])
    dataset_dic['train_labels'] = load_label_mnist(file_dic['train_labels'])
    dataset_dic['test_images'] = load_image_mnist(file_dic['test_images'])
    dataset_dic['test_labels'] = load_label_mnist(file_dic['test_labels'])

    # save datasets dictionary as pkl
    as_pickle = os.path.join(datasets_path, "mnist.pkl")
    with open(as_pickle, "wb") as fp:
        pickle.dump(dataset_dic, fp, -1)

    # load pkl data
    with open(as_pickle, "rb") as fp:
        pkl_datasets = pickle.load(fp)

    # number of files
    print(pkl_datasets['train_images'].shape)
    print(pkl_datasets['train_labels'].shape)

    # show graphic
    gra_train_img = pkl_datasets["train_images"][144].reshape((28, 28))
    plt.imshow(gra_train_img)
    plt.show()
    print(pkl_datasets["train_labels"][144])
