import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def load_data(path, n_classes, train_size=0.75):
    df = pd.read_csv(path)

    df.fillna(0)

    df = shuffle(df)

    x_values = []
    y_values = []
    for index, row in df.iterrows():
        serie_values = []
        serie_class = np.zeros(n_classes)
        for columnName, columnData in row.iteritems():
            if columnName == "class":
                serie_class[int(columnData)] = 1
            else:
                serie_values.append(columnData)
        x_values.append(serie_values)
        y_values.append(serie_class)

    x = int(len(x_values) * train_size)
    y = int(len(y_values) * train_size)

    x_train = np.array(x_values[:x])
    y_train = np.array(y_values[:y])

    x_test = np.array(x_values[x:])
    y_test = np.array(y_values[y:])

    return (x_train, y_train), (x_test, y_test)
