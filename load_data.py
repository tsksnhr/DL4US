import numpy as np
import os
import gzip
import pickle
import matplotlib.pyplot as plt

def load_image_mnist(datasets_path, value):

    # file path
    file_path = os.path.join(datasets_path, value)

    # open gz file
    with gzip.open(file_path, "rb") as fp:
        image = np.frombuffer(fp.read(), np.uint8, offset=16)

    image = image.reshape(-1,28**2)

    return image


def load_label_mnist(datasets_path, value):

    # file path
    file_path = os.path.join(datasets_path, value)

    # open gz file
    with gzip.open(file_path, "rb") as fp:
        label = np.frombuffer(fp.read(), np.uint8, offset=8)

    return label


def load_mnist_datasets_as_pkl():

    # directory path (in other env, change path)
    datasets_path = os.path.join('C:\\', 'Users', 'chest', 'Desktop', 'python', 'Learn_DL4US', 'datasets')

    # set file-name
    file_dic = {}
    file_dic['train_images'] = 'train-images-idx3-ubyte.gz'
    file_dic['train_labels'] = 'train-labels-idx1-ubyte.gz'
    file_dic['test_images'] = 't10k-images-idx3-ubyte.gz'
    file_dic['test_labels'] = 't10k-labels-idx1-ubyte.gz'

    # assign datasets to dictionary
    dataset_dic = {}
    dataset_dic['train_images'] = load_image_mnist(datasets_path, file_dic['train_images'])
    dataset_dic['train_labels'] = load_label_mnist(datasets_path, file_dic['train_labels'])
    dataset_dic['test_images'] = load_image_mnist(datasets_path, file_dic['test_images'])
    dataset_dic['test_labels'] = load_label_mnist(datasets_path, file_dic['test_labels'])

    # show datasets (use for debag)
    print(dataset_dic)

    # save datasets dictionary as pkl
    as_pickle = os.path.join(datasets_path, "mnist.pkl")
    with open(as_pickle, "wb") as fp:
        pickle.dump(dataset_dic, fp, -1)

    # load pkl data
    with open(as_pickle, "rb") as fp:
        pkl_datasets = pickle.load(fp)

    return pkl_datasets


def to_one_hot(label):

    temp_label = np.zeros((label.size, 10))

    for i in range(label.size):
        temp_label[i, label[i]] = 1

    return temp_label


def normarize(key):

    pkl_datasets[key] = pkl_datasets[key].astype(np.float32)
    pkl_datasets[key] /= 255 

    return pkl_datasets[key]


if __name__ == "__main__":

    pkl_datasets = load_mnist_datasets_as_pkl()

    # number of files
    print("\nnumber of files")
    print(pkl_datasets['train_images'].shape)
    print(pkl_datasets['train_labels'].shape)
    print(pkl_datasets['test_images'].shape)
    print(pkl_datasets['test_labels'].shape)

    # show one-hot-label
    print("\none-hot-label indicated 8")
    print(to_one_hot(pkl_datasets['train_labels'])[144])

    # normarized
    print("\nnormarized image-data")
    print(normarize("train_images"))

    # show graphic
    print("\ncorrect label is", pkl_datasets["train_labels"][144])
    gra_train_img = pkl_datasets["train_images"][144].reshape((28, 28))
    plt.imshow(gra_train_img)
    plt.show()

