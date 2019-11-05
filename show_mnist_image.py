# official
import sys
import os
import numpy as np
from PIL import Image
# made by me
import load_mnist_dataset

sys.path.append(os.pardir)

def show_img(img):
    pil_img = Image.fromarray(img)
    pil_img.show()

((x_train, t_train), (x_test, t_test)) = load_mnist_dataset.load_mnist_dataset(one_hot=False, normarize=False, flatten=True)

img = x_train[0]
label = t_train[0]

print(label)

print(img.shape)
img = img.reshape((28, 28))
print(img.shape)

show_img(img)
