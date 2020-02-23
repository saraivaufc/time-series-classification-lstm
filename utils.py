import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing import sequence


def load_data(path, n_classes, train_size=0.9):
    df = pd.read_csv(path)

    df = shuffle(df)

    x_values = []
    y_values = []
    for index, row in df.iterrows():
        serie_values = []
        serie_class = np.zeros(n_classes)

        for columnName, columnData in row.iteritems():
            if pd.isna(columnData):
                continue
            else:
                if columnName == "CLASS":
                    serie_class[int(columnData)] = 1
                else:
                    try:
                        value = [float(columnData)]
                    except Exception as e:
                        continue
                        # value = [0]
                    serie_values.append(value)

        serie_values = np.array(serie_values)

        serie_values = remove_zeros(serie_values)

        x_values.append(serie_values)
        y_values.append(serie_class)

    x = int(len(x_values) * train_size)
    y = int(len(y_values) * train_size)

    x_train = np.array(x_values[:x])
    y_train = np.array(y_values[:y])

    x_test = np.array(x_values[x:])
    y_test = np.array(y_values[y:])

    print(x_train[0])
    print(x_train.shape)

    return (x_train, y_train), (x_test, y_test)


def remove_zeros(serie_values):
    raw_shape = serie_values.shape
    serie_values = serie_values[~np.all(serie_values == 0, axis=1)]

    if len(serie_values) >= 1:
        # Normalize
        serie_mean = float(np.mean(serie_values))
        serie_std = float(np.std(serie_values)) + 0.000001

        serie_values = (serie_values - serie_mean) / serie_std
        # End Normalize

    serie_values = sequence.pad_sequences([serie_values],
                                          maxlen=raw_shape[0],
                                          dtype='float32')[0]

    if serie_values.shape != raw_shape:
        serie_values = serie_values.reshape(raw_shape)

    return serie_values