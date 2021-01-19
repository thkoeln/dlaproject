from models.music_lstm import get_model
import matplotlib.pyplot as plt
import tensorflow as tf
from utils.plot import show_plot, multi_step_plot
import pandas as pd
import numpy as np

# datasets
from datasets.music_dataset import BASE_BPM, get_dataset as get_dataset_music


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.show()


class TrainerMusic:
    def __init__(self, epochs, learning_rate, val_split, train_split, **kwargs):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.val_split = val_split
        self.train_split = train_split
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
            # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
                print(e)

    
    def predictionToArr(self, pred):
        arr = np.zeros((len(pred), 88+1), dtype=np.int16)
        for i in range(0, len(pred)):
            arr[i][0] = int(BASE_BPM * pred[i][0]) 
            for n in range(1, 88+1):
                count = 1
                max = 0.0
                maxIndex = 0
                for k in range(0,3):
                    if pred[i][count] > max:
                        maxIndex = k
                    count = count + 1
                arr[i][n] = maxIndex
        return arr


    def train(self, plot=True, batch_size=32, lstm_layers=16, composer=None, **kwargs):
        music = True
        if music:
            future_target = 265 # output size: bestimmt die größe des letzten Dense layer (u.a.)
            plot_multi_variate = False
            single_step_prediction = True # Will break model.fit() with False
            # get dataset
            training_set, validation_set, shape = get_dataset_music(future_target=future_target,
                                                                    single_step=single_step_prediction,
                                                                    batch_size=batch_size, composer=composer,train_split_pct=self.train_split)
        else:
            raise NotImplementedError()

        # tensorboard callback to log model training
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./log")

        # get model
        model = get_model(input_shape=shape, lr=self.learning_rate, future_target=future_target, lstm_layers=lstm_layers)

        # train model
        history = model.fit(
            training_set,
            validation_data=validation_set,
            epochs=self.epochs,
            callbacks=[tensorboard_callback]
        )

        print("Parameters: {}".format(model.count_params()))
        #print(model.get_weights())

        if plot:
            cee = history.history['categorical_crossentropy']
            val_cee = history.history['val_categorical_crossentropy']

            acc = history.history['mse']
            val_acc = history.history['mse']

            loss = history.history['loss']
            val_loss = history.history['val_loss']

            epochs_range = range(self.epochs)

            plt.figure(figsize=(8, 12))
            plt.subplot(1, 3, 1)
            plt.plot(epochs_range, cee, label='Training CCE')
            plt.plot(epochs_range, val_cee, label='Validation CCE')
            plt.legend(loc='lower right')
            plt.title('Training and Validation CCE')

            plt.subplot(1, 3, 2)
            plt.plot(epochs_range, acc, label='Training MSE')
            plt.plot(epochs_range, val_acc, label='Validation MSE')
            plt.legend(loc='lower right')
            plt.title('Training and Validation MSE')

            plt.subplot(1, 3, 3)
            plt.plot(epochs_range, loss, label='Training Loss')
            plt.plot(epochs_range, val_loss, label='Validation Loss')
            plt.legend(loc='upper right')
            plt.title('Training and Validation Loss')
            plt.show()

            testset =  validation_set.take(100)
            test_iterator = testset.as_numpy_iterator()
            for test in test_iterator:
                # print(test)
                # print(type(test))
                prediction = model.predict(test[0])
                print(prediction.shape)
                arr = self.predictionToArr(prediction)
                #print(arr)
                #print(prediction)
                # print(prediction[0])
                # print(prediction[1])
                # print(prediction[2])
                arr_dataframe = pd.DataFrame(arr)
                print(arr_dataframe.shape)
                arr_dataframe.to_csv("test_arr.csv", header=False, index=False)
                pd.DataFrame(prediction).to_csv("test_pred.csv", header=False, index=False)
