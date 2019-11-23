import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM
import csv
from tensorflow.keras.callbacks import EarlyStopping

def load_data():
    # 学習データ
    x_train = np.load('/root/userspace/lesson4/data/x_train.npy')
    y_train = np.load('/root/userspace/lesson4/data/y_train.npy')
    tokenizer_en = np.load('/root/userspace/lesson4/data/tokenizer_en.npy').item()
    tokenizer_ja = np.load('/root/userspace/lesson4/data/tokenizer_ja.npy').item()
 
    # テストデータ
    x_test = np.load('/root/userspace/lesson4/data/x_test.npy')

    return (x_train, y_train, tokenizer_en, tokenizer_ja, x_test)

x_train, y_train, tokenizer_en, tokenizer_ja, x_test = load_data()

emb_dim = 256
hid_dim = 256

en_vocab_size = len(tokenizer_en.word_index) + 1
ja_vocab_size = len(tokenizer_ja.word_index) + 1

seqX_len = len(x_train[0])
seqY_len = len(y_train[0])

encoder_inputs = Input(shape=(seqX_len,))
encoder_embedded = Embedding(en_vocab_size, emb_dim, mask_zero=True)(encoder_inputs)
_, *encoder_states = LSTM(hid_dim, return_state=True)(encoder_embedded)

decoder_inputs = Input(shape=(seqY_len,))
decoder_embedding = Embedding(ja_vocab_size, emb_dim)
decoder_embedded = decoder_embedding(decoder_inputs)
decoder_lstm = LSTM(hid_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedded, initial_state=encoder_states)
decoder_dense = Dense(ja_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

train_target = np.hstack((y_train[:, 1:], np.zeros((len(y_train),1), dtype=np.int32)))
model.fit([x_train, y_train], np.expand_dims(train_target, -1), batch_size=1024, epochs=20, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_acc', patience=2, verbose=1)])

encoder_model = Model(encoder_inputs, encoder_states)

decoder_states_inputs = [Input(shape=(hid_dim,)), Input(shape=(hid_dim,))]
decoder_inputs = Input(shape=(1,))
decoder_embedded = decoder_embedding(decoder_inputs)
decoder_outputs, *decoder_states = decoder_lstm(decoder_embedded, initial_state=decoder_states_inputs)
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

def decode_sequence(input_seq, bos_eos, max_output_length):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.array(bos_eos[0])
    output_seq = bos_eos[0][:]

    while True:
        output_tokens, *states_value = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = [np.argmax(output_tokens[0, -1, :])]
        output_seq += sampled_token_index

        if (sampled_token_index == bos_eos[1] or len(output_seq) > max_output_length):
            break

        target_seq = np.array(sampled_token_index)

    return output_seq

bos_eos = tokenizer_ja.texts_to_sequences(["<s>", "</s>"])
output = [decode_sequence(x_test[i][np.newaxis,:], bos_eos, 100)[1:-1] for i in range(len(x_test))]

with open('/root/userspace/lesson4/submission/submission.csv', 'w') as file:
    writer = csv.writer(file, lineterminator='\n')
    writer.writerows(output)