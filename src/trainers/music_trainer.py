from models.music_lstm import get_model
import matplotlib.pyplot as plt
import tensorflow as tf
from utils.plot import show_plot, multi_step_plot

# datasets
from datasets.music_dataset import get_dataset as get_dataset_music


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.show()


class TrainerRNN:
    def __init__(self, epochs, learning_rate, val_split, train_split, **kwargs):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.val_split = val_split
        self.train_split = train_split

    def train(self, plot=True, image_width=180, image_height=180, batch_size=32, lstm_layers=16, **kwargs):
        music = True
        if music:
            future_target = 0
            plot_multi_variate = False
            single_step_prediction = True
            # get dataset
            training_set, validation_set, shape = get_dataset_music(future_target=future_target,
                                                                    single_step=single_step_prediction,
                                                                    batch_size=batch_size)
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

        if plot:
            mae = history.history['mae']
            val_mae = history.history['val_mae']

            loss = history.history['loss']
            val_loss = history.history['val_loss']

            epochs_range = range(self.epochs)

            plt.figure(figsize=(8, 8))
            plt.subplot(1, 2, 1)
            plt.plot(epochs_range, mae, label='Training MAE')
            plt.plot(epochs_range, val_mae, label='Validation MAE')
            plt.legend(loc='lower right')
            plt.title('Training and Validation MAE')

            plt.subplot(1, 2, 2)
            plt.plot(epochs_range, loss, label='Training Loss')
            plt.plot(epochs_range, val_loss, label='Validation Loss')
            plt.legend(loc='upper right')
            plt.title('Training and Validation Loss')
            plt.show()

            for x, y in validation_set.take(3):
                if plot_multi_variate:
                    plot = multi_step_plot(x[0].numpy(), y[0].numpy(), model.predict(x)[0], future_target)
                else:
                    plot = show_plot([x[0].numpy(), y[0].numpy(),
                                     model.predict(x)[0]], 0, 'Music LSTM model')
                plot.show()