from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.models import Sequential

def model_fn(sequence_size, n_features):
    model = Sequential()

    model.add(LSTM(60, input_shape=(sequence_size, n_features)))

    model.add(Dense(60, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())

    return model
