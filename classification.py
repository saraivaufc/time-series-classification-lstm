import numpy as np
import tensorflow as tf
from osgeo import gdal
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing import sequence

from model import model_fn
from utils import load_data, remove_zeros


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
            n_classes=self.__n_classes,
            train_size=0.9
        )

        X_train = sequence.pad_sequences(X_train,
                                         maxlen=self.__sequence_size,
                                         dtype='float32')

        self.__model.fit(X_train, y_train,
                         validation_split=0.10,
                         shuffle=True,
                         epochs=epochs,
                         batch_size=batch_size,
                         callbacks=self.__callbacks)

        X_test = sequence.pad_sequences(X_test,
                                        maxlen=self.__sequence_size,
                                        dtype='float32')

        y_test = y_test.reshape((y_test.shape[0]))

        scores = self.__model.evaluate(X_test, y_test,
                                       verbose=0)

        print("Accuracy: %.2f%%" % (scores[1] * 100))

    def predict(self, image_path, predicted_path, batch_size=255):
        data_source = gdal.Open(image_path)

        image = data_source.ReadAsArray()

        flat_image = image.reshape(image.shape[0],
                                   image.shape[1] * image.shape[2])

        flat_image = flat_image.transpose()

        flat_image = flat_image.reshape((flat_image.shape[0],
                                         flat_image.shape[1],
                                         self.__n_features))

        flat_image = np.array(flat_image).astype(float)

        for index, serie_values in enumerate(flat_image):
            serie_values = remove_zeros(serie_values)
            flat_image[index] = serie_values

        flat_image = sequence.pad_sequences(flat_image,
                                            maxlen=self.__sequence_size,
                                            dtype='float32')

        flat_predicted = self.__model.predict(flat_image,
                                              batch_size=batch_size)

        print(flat_predicted[0])

        flat_predicted = np.argmax(flat_predicted, axis=1)

        print(flat_predicted[0])

        predicted_image = flat_predicted.reshape((image.shape[1],
                                                  image.shape[2]))

        # save results
        driver = data_source.GetDriver()
        output_dataset = driver.Create(predicted_path,
                                       predicted_image.shape[1],
                                       predicted_image.shape[0],
                                       1,
                                       gdal.GDT_Byte,
                                       ['COMPRESS=DEFLATE'])
        output_dataset.SetGeoTransform(data_source.GetGeoTransform())
        output_dataset.SetProjection(data_source.GetProjection())
        output_dataset.GetRasterBand(1).WriteArray(predicted_image, 0, 0)
        output_dataset.FlushCache()
        print("Results saved!")
