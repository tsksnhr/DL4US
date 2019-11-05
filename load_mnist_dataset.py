# official
import numpy as np
import urllib.request
import os
import gzip
import pickle
import matplotlib.pyplot as plt

# set url
url_base = 'http://yann.lecun.com/exdb/mnist/'

# set file-name
file_dic = {}
file_dic['train_images'] = 'train-images-idx3-ubyte.gz'
file_dic['train_labels'] = 'train-labels-idx1-ubyte.gz'
file_dic['test_images'] = 't10k-images-idx3-ubyte.gz'
file_dic['test_labels'] = 't10k-labels-idx1-ubyte.gz'

# directory path
datasets_path = os.path.join('C:\\', 'Users', 'chest', 'Desktop', 'python', 'Learn_DL4US', 'datasets')

# pickle file-name
pickle_file = os.path.join(datasets_path, "mnist.pkl")


def download_datasets(datasets_path):

    if os.path.exists(datasets_path):
        count = 0

        # download files
        for value in file_dic.values():
            # debug sentences
            if count == 0:
                print("downloading", 'train_images...')
            elif count == 1:
                print("dounloading", "train_labels...")
            elif count == 2:
                print("downloading", "test_images...")
            else:
                print("downloading", "test_labels...")

            file_path = os.path.join(datasets_path, value)
            urllib.request.urlretrieve(url_base+value, file_path)
            count += 1

    # directory is not fine
    else:
        print("No directory, make directory")



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



def convert():

    # assign datasets to dictionary
    dataset_dic = {}
    dataset_dic['train_images'] = load_image_mnist(datasets_path, file_dic['train_images'])
    dataset_dic['train_labels'] = load_label_mnist(datasets_path, file_dic['train_labels'])
    dataset_dic['test_images'] = load_image_mnist(datasets_path, file_dic['test_images'])
    dataset_dic['test_labels'] = load_label_mnist(datasets_path, file_dic['test_labels'])

    return dataset_dic



def convert_to_pkl():

    dataset_dic = convert()

    # save datasets dictionary as pkl
    as_pickle = os.path.join(datasets_path, "mnist.pkl")
    with open(as_pickle, "wb") as fp:
        pickle.dump(dataset_dic, fp, -1)

    # load pkl data
    with open(as_pickle, "rb") as fp:
        pkl_datasets = pickle.load(fp)

    return pkl_datasets



def init_dataset():

    if os.path.exists(datasets_path):
        download_datasets(datasets_path)
        dataset_dic = convert()

        with open(pickle_file, "wb") as fp:
            pickle.dump(dataset_dic, fp, -1)
        print("Initialized.")
    
    else:
        print("Invalid path.")



def to_one_hot(pkl_datasets):

    temp_label = np.zeros((pkl_datasets.size, 10))

    for i in range(pkl_datasets.size):
        temp_label[i, pkl_datasets[i]] = 1

    return temp_label



def normarize(pkl_datasets, key):

    pkl_datasets[key] = pkl_datasets[key].astype(np.float32)
    pkl_datasets[key] /= 255 

    return pkl_datasets



def flatten(pkl_datasets, key):
    pkl_datasets[key].reshape(-1, 1, 28, 28)



def load_mnist_dataset(one_hot=True, normarize=True, flatten=True):

    # save mnist dataset as pkl file
    if  not os.path.exists(pickle_file):
        init_dataset()
    else:
        print("already exists pkl file.")

    # load pkl data
    with open(pickle_file, "rb") as fp:
        pkl_datasets = pickle.load(fp)

    if one_hot:
        pkl_datasets["train_labels"] = to_one_hot(pkl_datasets["train_labels"])
        pkl_datasets["test_labels"] = to_one_hot(pkl_datasets["test_labels"])

    if normarize:
        for key in ("train_images", "test_images"):
                
            pkl_datasets[key] = pkl_datasets[key].astype(np.float32)
            pkl_datasets[key] /= 255 

    if flatten:
        for key in ("train_images", "test_images"):
            pkl_datasets[key].reshape(-1, 1, 28, 28)

    return (pkl_datasets["train_images"], pkl_datasets["train_labels"]), (pkl_datasets["test_images"], pkl_datasets["test_labels"])
    

if __name__ == "__main__":
    print(load_mnist_dataset(one_hot=True, normarize=True, flatten=True))    
