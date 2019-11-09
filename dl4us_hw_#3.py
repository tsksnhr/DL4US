import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, SimpleRNN, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping

import pandas as pd
import matplotlib.pyplot as plt

# variables
Hid_dim = 10
look_back = 3
Drop_out_ratio = 0.5
Patience = 20
Val_split = 0.1
Batch_size = 1024
Epochs = 100

def load_dataset():
    # 学習データ
    x_train = np.load('/root/userspace/public/lesson3/data/x_train.npy')
    y_train = np.load('/root/userspace/public/lesson3/data/y_train.npy')
    y_train = to_categorical(y_train[:, np.newaxis], num_classes = 5)
    
    # テストデータ
    x_test = np.load('/root/userspace/public/lesson3/data/x_test.npy')

    return (x_train, x_test, y_train)

x_train, x_test, y_train = load_dataset()

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)

#def create_dataset(dataset, look_back=1):
#    X, Y = [], []
#    for i in range (len(dataset)-look_back-1):
#        temp = dataset[i:(i+look_back), 0]
#        X.append(temp)
#        Y .append(dataset[i+look_back, 0])
#    return np.array(X), np.array(Y)

#x_train, y_train = create_dataset(x_train, look_back)
#testX, testY = create_dataset(x_test, look_back)

# simple RNN
model_SRNN = Sequential()

model_SRNN.add(SimpleRNN(Hid_dim, input_shape=x_train.shape[1:]))
model_SRNN.add(Dense(y_train.shape[1], activation='softmax'))

model_SRNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history_SRNN = model_LSTM.fit(x_train, y_train, epochs=Epochs, batch_size=Batch_size, verbose=2, validation_split=Val_split, callbacks=[EarlyStopping(monitor='val_acc', patience=Patience, verbose=1)])

# LSTM
model_LSTM = Sequential()

model_LSTM.add(LSTM(Hid_dim, input_shape=x_train.shape[1:], dropout=Drop_out_ratio, recurrent_dropout=Drop_out_ratio, unroll=True))
model_LSTM.add(Dense(y_train.shape[1], activation='softmax'))

model_LSTM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history_LSTM = model_LSTM.fit(x_train, y_train, epochs=Epochs, batch_size=Batch_size, verbose=2, validation_split=Val_split, callbacks=[EarlyStopping(monitor='val_acc', patience=Patience, verbose=1)])

# GRU
model_GRU = Sequential()

model_GRU.add(GRU(Hid_dim, input_shape=x_train.shape[1:], dropout=Drop_out_ratio, recurrent_dropout=Drop_out_ratio, unroll=True))
model_GRU.add(Dense(y_train.shape[1], activation='softmax'))

model_GRU.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history_GRU = model_GRU.fit(x_train, y_train, epochs=Epochs, batch_size=Batch_size, verbose=2, validation_split=Val_split, callbacks=[EarlyStopping(monitor='val_acc', patience=Patience, verbose=1)])

# ensemble　learning

models = [model_SRNN, model_LSTM, model_GRU]

def ensembling_soft(models, X):
    preds_sum = None
    for model in models:
        if preds_sum is None:
            preds_sum = model.predict(X)
        else:
            preds_sum += model.predict(X)
    probs = preds_sum / len(models)
    return np.argmax(probs, 1)

y_pred = ensembling_soft(models, x_test)

submission = pd.Series(y_pred, name='label')
submission.to_csv('/root/userspace/lesson3/submissions/submission.csv', header=True, index_label='id')

# Plot training & validation accuracy values
plt.plot(history_SRNN.history['acc'])
plt.plot(history_SRNN.history['val_acc'])
plt.plot(history_LSTM.history['acc'])
plt.plot(history_LSTM.history['val_acc'])
plt.plot(history_GRU.history['acc'])
plt.plot(history_GRU.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test','Train', 'Test','Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history_SRNN.history['loss'])
plt.plot(history_SRNN.history['val_loss'])
plt.plot(history_LSTM.history['loss'])
plt.plot(history_LSTM.history['val_loss'])
plt.plot(history_GRU.history['loss'])
plt.plot(history_GRU.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test''Train', 'Test','Train', 'Test'], loc='upper left')
plt.show()
