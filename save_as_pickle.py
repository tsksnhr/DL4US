# save mnist datasets as pickle 

import gzip
import os
import numpy as np

# set file-name
file_dic = {}
file_dic['train_images'] = 'train-images-idx3-ubyte.gz'
file_dic['train_labels'] = 'train-labels-idx1-ubyte.gz'
file_dic['test_images'] = 't10k-images-idx3-ubyte.gz'
file_dic['test_labels'] = 't10k-labels-idx1-ubyte.gz'

# directory path  (choose your username)
# datasets_path = os.path.join('C:\\', 'Users', 'toshi', 'Desktop', 'python', 'Learn_DL4US', 'datasets')
datasets_path = os.path.join('C:\\', 'Users', 'chest', 'Desktop', 'python', 'Learn_DL4US', 'datasets')
print(datasets_path)

# vacant list for data-size
data_size = {}

# download files
for value in file_dic.values():

    # file path
    file_path = os.path.join(datasets_path, value)

    # open gz file
    with gzip.open(file_path, "rb") as fp:
        data = np.frombuffer(fp.read(), np.uint8)
        
#    data_size.append(len(data))
    data_size[value] = len(data)

print("data size:", data_size)

# data-size verification
print()
print("filesize_train-label =", data_size['train-images-idx3-ubyte.gz']//(28**2))
print("filesize_test-label =", data_size['t10k-images-idx3-ubyte.gz']//(28**2))
print()
print(data_size['train-images-idx3-ubyte.gz']%(28**2))
print(data_size['train-labels-idx1-ubyte.gz']%(28**2))
print(data_size['t10k-images-idx3-ubyte.gz']%(28**2))
print(data_size['t10k-labels-idx1-ubyte.gz']%(28**2))
