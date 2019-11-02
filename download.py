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
datasets_path = os.path.join('C\:', 'Users', 'toshi', 'Desktop', 'python', 'Learn_DL4US', 'datasets')
print(datasets_path)

if os.path.exists(datasets_path):
    
    # download files
    for value in file_dic.values():
        file_path = os.path.join(datasets_path, value)
        urllib.request.urlretrieve(url_base+value, file_path)

else:
    print("No directory")
