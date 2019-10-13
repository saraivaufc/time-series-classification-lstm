# https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
# https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
import tensorflow as tf
from osgeo import gdal
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing import sequence

from model import model_fn
from utils import load_data


class Classification():
    def __init__(self, sequence_size, n_features, model_dir):
        self.__sequence_size = sequence_size
        self.__n_features = n_features
        self.__model_dir = model_dir

        self.__checkpoint_path = "{dir}/model.ckpt".format(dir=model_dir)

        self.__model = model_fn(sequence_size=self.__sequence_size,
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

    def train(self, path, epochs=100, batch_size=255):
        (X_train, y_train), (X_test, y_test) = load_data(path)

        X_train = sequence.pad_sequences(X_train,
                                         maxlen=self.__sequence_size)

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], self.__n_features))

        self.__model.fit(X_train, y_train,
                         validation_split=0.25,
                         shuffle=True,
                         epochs=epochs,
                         batch_size=batch_size,
                         callbacks=self.__callbacks)

        X_test = sequence.pad_sequences(X_test,
                                        maxlen=self.__sequence_size)

        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], self.__n_features))

        scores = self.__model.evaluate(X_test, y_test,
                                       verbose=0)

        print("Accuracy: %.2f%%" % (scores[1] * 100))

    def predict(self, image_path, predicted_path):
        dataSource = gdal.Open(image_path)

        array = dataSource.ReadAsArray()

        saved_shape = array.shape

        reshaped = array.reshape(saved_shape[0],
                                 saved_shape[1] * saved_shape[2])

        reshaped = reshaped.transpose()

        print(reshaped.shape)

        reshaped = sequence.pad_sequences(reshaped,
                                          maxlen=self.__sequence_size)

        reshaped = reshaped.reshape((reshaped.shape[0],
                                     reshaped.shape[1],
                                     self.__n_features))

        predicted = self.__model.predict(reshaped, batch_size=255)
        predicted[predicted > 0.5] = 1
        predicted[predicted <= 0.5] = 0
        predicted = predicted.astype(int)

        predicted_image = predicted.reshape((saved_shape[1], saved_shape[2]))

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
