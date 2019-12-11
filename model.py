from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

rnn_width = 200


def model_fn(n_classes, sequence_size, n_features):
    model = Sequential()

    model.add(LSTM(units=rnn_width, input_shape=(sequence_size, n_features)))

    model.add(Dense(n_classes, activation='softmax'))

    optimizer = Adam(learning_rate=0.001)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    print(model.summary())

    return model
