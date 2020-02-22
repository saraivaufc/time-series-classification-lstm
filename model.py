from tensorflow.keras.layers import Dense, LSTM, \
    Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.models import Sequential

rnn_width = 200


def model_fn(n_classes, sequence_size, n_features):
    model = Sequential()

    model.add(Conv1D(filters=32,
                     kernel_size=3,
                     padding='same',
                     activation='relu',
                     input_shape=(sequence_size, n_features)))

    model.add(MaxPooling1D(pool_size=2))

    model.add(Dropout(0.2))

    model.add(LSTM(rnn_width))

    model.add(Dropout(0.2))

    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())

    return model
