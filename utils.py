import numpy as np
import pandas as pd
from sklearn.utils import shuffle


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
                        value = int(columnData)
                    except:
                        value = 0
                    serie_values.append(value)

        serie_values = np.array(serie_values)

        serie_max = np.max(serie_values)
        serie_min = np.min(serie_values)

        serie_values = (2 * ((serie_values - serie_min)/(
                serie_max-serie_min))) -1

        if len(serie_values) >= 1:
            x_values.append(serie_values)
            y_values.append(serie_class)

        x_values.append(serie_values)
        y_values.append(serie_class)

    x = int(len(x_values) * train_size)
    y = int(len(y_values) * train_size)

    x_train = np.array(x_values[:x])
    y_train = np.array(y_values[:y])

    x_test = np.array(x_values[x:])
    y_test = np.array(y_values[y:])

    return (x_train, y_train), (x_test, y_test)
