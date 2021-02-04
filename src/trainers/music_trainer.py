from datasets.MidiParser import MidiParser
from models.music_lstm import get_model
import matplotlib.pyplot as plt
import tensorflow as tf
from utils.plot import show_plot, multi_step_plot
import pandas as pd
import numpy as np

# datasets
from datasets.music_dataset import BASE_BPM, BPM_MODIFIER, FEATURE_SIZE, get_dataset as get_dataset_music
PREDICTION_LENGTH = 100

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
        policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
        tf.keras.mixed_precision.experimental.set_policy(policy)
        tf.keras.backend.clear_session()

    def predictionToArr(self, pred):
        arr = np.zeros((len(pred), 88+1), dtype=np.int16)
        for i in range(0, len(pred)):
            arr[i][0] = int((BASE_BPM * pred[i][0])+BPM_MODIFIER) 
            for n in range(1, 88+1):                
                if pred[i][n*2 - 1] >= 0.1:
                    arr[i][n] = 1
                    continue
                if pred[i][n*2+1 - 1] >= 0.1:
                    arr[i][n] = 2
                    continue
        return arr


    def train(self, plot=True, batch_size=32, lstm_layers=16, composer=None, **kwargs):
        music = True
        if music:
            future_target = FEATURE_SIZE # output size: bestimmt die größe des letzten Dense layer (u.a.)
            plot_multi_variate = False
            single_step_prediction = True # Will break model.fit() with False
            # get dataset
            training_set_gen, validation_set_gen, shape, test_dataset = get_dataset_music(future_target=future_target,
                                                                    single_step=single_step_prediction,
                                                                    batch_size=batch_size, composer=composer,train_split_pct=self.train_split)
        else:
            raise NotImplementedError()

        # tensorboard callback to log model training
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./log")

        input_shape = {}
        input_shape["shape_key"] = (200,176)
        input_shape["shape_metro"] = (200,1)

        # get model
        model = get_model(input_shape=input_shape, lr=self.learning_rate, future_target=future_target, lstm_layers=lstm_layers)

        # train model
        history = model.fit(
            training_set_gen,
            validation_data=validation_set_gen,
            epochs=self.epochs,
            callbacks=[tensorboard_callback]
        )

        print("Parameters: {}".format(model.count_params()))
        #print(model.get_weights())

        if plot:
            cee = history.history['binary_crossentropy']
            val_cee = history.history['val_binary_crossentropy']

            acc = history.history['mae']
            val_acc = history.history['val_mae']

            loss = history.history['loss']
            val_loss = history.history['val_loss']

            epochs_range = range(self.epochs)

            plt.figure(figsize=(8, 12))
            plt.subplot(1, 3, 1)
            plt.plot(epochs_range, cee, label='Training BCE')
            plt.plot(epochs_range, val_cee, label='Validation BCE')
            plt.legend(loc='lower right')
            plt.title('Training and Validation BCE')

            plt.subplot(1, 3, 2)
            plt.plot(epochs_range, acc, label='Training MAE')
            plt.plot(epochs_range, val_acc, label='Validation MAE')
            plt.legend(loc='lower right')
            plt.title('Training and Validation MSE')

            plt.subplot(1, 3, 3)
            plt.plot(epochs_range, loss, label='Training Loss')
            plt.plot(epochs_range, val_loss, label='Validation Loss')
            plt.legend(loc='upper right')
            plt.title('Training and Validation Loss')
            plt.show()

            #[anzahl_zeilen, 64, 177]
            zeilen = test_dataset.shape[0]
            spalten = test_dataset.shape[1]
            batches = int(zeilen/PREDICTION_LENGTH)

            pred_dataset = np.empty((int(zeilen/batches),batches,spalten))
            print(pred_dataset.shape)

            test_dataset = test_dataset.tolist()

            for z in range(int(zeilen/batches)):
                for b in range(batches):
                    pred_dataset[z][b] = test_dataset.pop(0)
                    #print("{} {} {}".format(z,b, pred_dataset[z][b]))
                
            #test_dataset = np.reshape(test_dataset, (test_dataset.shape[0], 1, test_dataset.shape[1]))
            pred_input = pred_dataset #test_dataset#:PREDICTION_LENGTH
            prediction = model.predict(pred_input)
            print(prediction.shape)
            arr = self.predictionToArr(prediction)
            arr_dataframe = pd.DataFrame(arr)
            print(arr_dataframe.shape)
            arr_dataframe.to_csv("test_arr.csv", header=False, index=False)
            pd.DataFrame(prediction).to_csv("test_pred.csv", header=False, index=False)
            arr_file = arr_dataframe.to_numpy()
            print(arr_file[0])
            print(arr_file[1])
            print(arr_file)
            arr_corr = MidiParser().validateCSV(arr_file)
            pd.DataFrame(arr_corr).to_csv("test_corr.csv", header=False, index=False)
            MidiParser().arrayToMidi(arr_file,"test_arr.mid")
            MidiParser().arrayToMidi(arr_corr,"test_corr.mid")