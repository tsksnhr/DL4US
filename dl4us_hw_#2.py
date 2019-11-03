import numpy as np
import pandas as pd

def load_cifar10():
    
    # 学習データ
    x_train = np.load('/root/userspace/public/lesson2/data/x_train.npy')
    y_train = np.load('/root/userspace/public/lesson2/data/y_train.npy')

    # テストデータ
    x_test = np.load('/root/userspace/public/lesson2/data/x_test.npy')
    
    x_train = x_train / 255.
    x_test = x_test / 255.
    
    y_train = np.eye(10)[y_train]
    
    return (x_train, x_test, y_train)

x_train, x_test, y_train = load_cifar10()


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential()

model.add(Conv2D(8, kernel_size=(5, 5), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(8, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',  metrics=['acc'])

history = model.fit(x=x_train, y=y_train, batch_size=1024, epochs=50, validation_split=0.1)

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, 1)

submission = pd.Series(y_pred, name='label')
submission.to_csv('/root/userspace/lesson2/submissions/submission.csv', header=True, index_label='id')


import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
