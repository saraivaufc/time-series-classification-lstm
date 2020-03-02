from tensorflow import keras


def LSTM(n_classes, sequence_size, n_features):
    model = keras.models.Sequential()

    model.add(keras.layers.Bidirectional(
        keras.layers.LSTM(128,
                          recurrent_dropout=0.5,
                          return_sequences=True),
        input_shape=(sequence_size, n_features)
    ))

    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Bidirectional(
        keras.layers.LSTM(128, recurrent_dropout=0.5)
    ))

    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(n_classes))

    model.add(keras.layers.Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())

    return model

def DeepConvLSTM(n_classes, sequence_size, n_features):
    model = keras.models.Sequential()

    model.add(keras.layers.Conv1D(filters=64,
                                  kernel_size=3,
                                  padding='same',
                                  activation='relu',
                                  input_shape=(sequence_size, n_features)))

    model.add(keras.layers.MaxPooling1D(pool_size=2))

    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Conv1D(filters=64,
                                  kernel_size=3,
                                  padding='same',
                                  activation='relu'))

    model.add(keras.layers.MaxPooling1D(pool_size=2))

    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.LSTM(128,
                                return_sequences=True,
                                recurrent_dropout=0.5))

    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.LSTM(128,
                                recurrent_dropout=0.5))

    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(n_classes))

    model.add(keras.layers.Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())

    return model

def DeepConvLSTM2(n_classes, sequence_size, n_features):
    model = keras.models.Sequential()

    model.add(keras.layers.Conv1D(filters=64,
                                  kernel_size=3,
                                  padding='same',
                                  activation='relu',
                                  input_shape=(sequence_size, n_features)))

    model.add(keras.layers.MaxPooling1D(pool_size=2))

    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.LSTM(128,
                                return_sequences=True,
                                recurrent_dropout=0.5))

    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Conv1D(filters=128,
                                  kernel_size=3,
                                  padding='same',
                                  activation='relu'))

    model.add(keras.layers.MaxPooling1D(pool_size=2))

    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.LSTM(128, recurrent_dropout=0.5))

    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(n_classes))

    model.add(keras.layers.Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())

    return model


model_fn = DeepConvLSTM2