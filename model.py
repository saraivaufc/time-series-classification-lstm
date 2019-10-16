from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential

rnn_width = 500


def model_fn(n_classes, sequence_size, n_features):
    model = Sequential()

    model.add(LSTM(rnn_width, input_shape=(sequence_size, n_features)))

    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())

    return model
