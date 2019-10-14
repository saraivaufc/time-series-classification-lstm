import numpy as np
import tensorflow as tf
from osgeo import gdal
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing import sequence

from model import model_fn
from utils import load_data


class Classification(object):
    def __init__(self, n_classes, sequence_size, n_features, model_dir):
        self.__n_classes = n_classes
        self.__sequence_size = sequence_size
        self.__n_features = n_features
        self.__model_dir = model_dir

        self.__checkpoint_path = "{dir}/model.ckpt".format(dir=model_dir)

        self.__model = model_fn(n_classes=self.__n_classes,
                                sequence_size=self.__sequence_size,
                                n_features=self.__n_features)

        latest = tf.train.latest_checkpoint(self.__model_dir)

        if latest:
            self.__model.load_weights(latest)
            print("Model loaded!")

        cp_callback = ModelCheckpoint(filepath=self.__checkpoint_path,
                                      save_weights_only=True,
                                      save_best_only=True)

        tensorboard_callback = TensorBoard(log_dir=self.__model_dir)

        self.__callbacks = [cp_callback, tensorboard_callback]

    def train(self, path, epochs=34, batch_size=255):
        (X_train, y_train), (X_test, y_test) = load_data(
            path=path,
            n_classes=self.__n_classes
        )

        X_train = sequence.pad_sequences(X_train,
                                         maxlen=self.__sequence_size)

        X_train = X_train.reshape(
            (X_train.shape[0], X_train.shape[1], self.__n_features))

        self.__model.fit(X_train, y_train,
                         validation_split=0.25,
                         shuffle=True,
                         epochs=epochs,
                         batch_size=batch_size,
                         callbacks=self.__callbacks)

        X_test = sequence.pad_sequences(X_test,
                                        maxlen=self.__sequence_size)

        X_test = X_test.reshape(
            (X_test.shape[0], X_test.shape[1], self.__n_features))

        scores = self.__model.evaluate(X_test, y_test,
                                       verbose=0)

        print("Accuracy: %.2f%%" % (scores[1] * 100))

    def predict(self, image_path, predicted_path):
        dataSource = gdal.Open(image_path)

        image = dataSource.ReadAsArray()

        flat_image = image.reshape(image.shape[0],
                                   image.shape[1] * image.shape[2])

        flat_image = flat_image.transpose()

        flat_image = sequence.pad_sequences(flat_image,
                                            maxlen=self.__sequence_size)

        flat_image = flat_image.reshape((flat_image.shape[0],
                                         flat_image.shape[1],
                                         self.__n_features))

        flat_predicted = self.__model.predict(flat_image, batch_size=255)

        flat_predicted = np.argmax(flat_predicted, axis=1)

        predicted_image = flat_predicted.reshape((image.shape[1],
                                                  image.shape[2]))

        # save results
        driver = dataSource.GetDriver()
        output_dataset = driver.Create(predicted_path,
                                       predicted_image.shape[1],
                                       predicted_image.shape[0],
                                       1,
                                       gdal.GDT_Byte,
                                       ['COMPRESS=DEFLATE'])
        output_dataset.SetGeoTransform(dataSource.GetGeoTransform())
        output_dataset.SetProjection(dataSource.GetProjection())
        output_dataset.GetRasterBand(1).WriteArray(predicted_image, 0, 0)
        output_dataset.FlushCache()
        print("Results saved!")
