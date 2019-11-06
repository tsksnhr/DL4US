import load_mnist_dataset
import numpy as np

def random_batch_choice(x, y, batch_size):
    
    data_size = x.shape[0]
    batch_filter = np.random.choice(data_size, batch_size)

    x_batch = x[batch_filter]
    y_batch = y[batch_filter]

    return x_batch, y_batch


if __name__ == "__main__":

    ((train_images, train_labels),(test_images, test_labels)) = load_mnist_dataset.load_mnist_dataset()

    print("Non-batch")
    print(train_images.shape)
    print(train_labels.shape)
    print(test_images.shape)
    print(test_labels.shape)

    (x_batch, y_batch) = random_batch_choice(train_labels, train_labels, batch_size=100)

    print("batch")
    print(x_batch.shape)
    print(y_batch.shape)
