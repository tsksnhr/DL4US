import numpy as np
import pandas as pd
import os

def load_mnist():

    # 学習データ
    x_train = np.load('/root/userspace/public/lesson1/data/x_train.npy')
    y_train = np.load('/root/userspace/public/lesson1/data/y_train.npy')
    
    # テストデータ
    x_test = np.load('/root/userspace/public/lesson1/data/x_test.npy')

    x_train = x_train.reshape(-1, 784).astype('float32') / 255
    x_test = x_test.reshape(-1, 784).astype('float32') / 255
    y_train = np.eye(10)[y_train.astype('int32').flatten()]

    return (x_train, x_test, y_train)

x_train, x_test, y_train = load_mnist()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential()

model.add(Dense(512, input_shape=(784,), activation='relu', kernel_initializer='he_normal'))
model.add(Dense(256, activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(128, activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(32, activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.1, callbacks=[EarlyStopping(patience=0, verbose=1)])

pred_y = model.predict(x_test)

#score = model.evaluate(pred_y, y_test, verbose=0)
#print('test loss:', score[0])
#print('test accuracy', score[1])

pred_y = np.argmax(pred_y, 1)
    
submission = pd.Series(pred_y, name='label')
submission.to_csv('/root/userspace/lesson1/submissions/submission.csv', header=True, index_label='id')