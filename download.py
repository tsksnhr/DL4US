# download

import urllib.request
import os

# set url
url_base = 'http://yann.lecun.com/exdb/mnist/'

# set file-name
file_dic = {}
file_dic['train_images'] = 'train-images-idx3-ubyte.gz'
file_dic['train_labels'] = 'train-labels-idx1-ubyte.gz'
file_dic['test_images'] = 't10k-images-idx3-ubyte.gz'
file_dic['test_labels'] = 't10k-labels-idx1-ubyte.gz'

# point directory out
datasets_path = os.path.join('C:\\', 'Users', 'toshi', 'Desktop', 'python', 'Learn_DL4US', 'datasets')
print(datasets_path)

# directory is fine, do download
if os.path.exists(datasets_path):
    
    count = 0

    # download files
    for value in file_dic.values():

        # debug sentences
        if count == 0:
            print("downloading", 'train_images')
        elif count == 1:
            print("dounloading", "train_labels")
        elif count == 2:
            print("downloading", "test_images")
        else:
            print("downloading", "test_labels")

        file_path = os.path.join(datasets_path, value)
        urllib.request.urlretrieve(url_base+value, file_path)

        count += 1

# directory is not fine
else:
    print("No directory, make directory")
