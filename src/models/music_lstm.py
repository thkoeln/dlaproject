import tensorflow as tf
from tensorflow.python.keras.layers.core import Dropout
from tensorflow import keras

from datasets.music_dataset import FEATURE_SIZE

                                     # output size (for us=(88)*2 + 1  = 177)
def get_model(input_shape, lr=0.001, future_target=FEATURE_SIZE, summary=True, lstm_layers=32):
    print("Input Shape: {}".format(input_shape))

    input_keys = keras.Input(shape=input_shape["shape_key"])
    input_metro = keras.Input(shape=input_shape["shape_metro"])

    lstm_keys = keras.layers.LSTM(lstm_layers, dropout=0.3, unroll=False, use_bias=True,name="KeysLSTM")
    lstm_keys_output = lstm_keys(input_keys)

    lstm_metro = keras.layers.LSTM(lstm_layers, dropout=0.3, unroll=False, use_bias=True, name="MetronomeLSTM")
    lstm_metro_output = lstm_metro(input_metro)

    concat_1 = keras.layers.Concatenate(name="ConcatenationLayer")
    concat_1_output = concat_1([lstm_keys_output, lstm_metro_output])

    #dense_1 = tf.keras.layers.Dense(future_target)
    #dense_1_output = dense_1(concat_1_output)

    dense_keys = tf.keras.layers.Dense(future_target-1, activation="sigmoid", name="KeysDenseLayer")
    dense_keys_output = dense_keys(concat_1_output)

    dense_metronome = tf.keras.layers.Dense(1, activation="relu", name="MetronomeDenseLayer")
    dense_metronome_output = dense_metronome(concat_1_output)

    music_lstm_model = keras.Model(inputs=[input_metro, input_keys], outputs=[dense_metronome_output, dense_keys_output], name="music_lstm_model")

    # Kann auskommentiert werden
    keras.utils.plot_model(music_lstm_model, "music_lstm_model.png", show_shapes=True, expand_nested=True)

    #music_lstm_model = tf.keras.models.Sequential([
    #    tf.keras.layers.LSTM(lstm_layers, input_shape=input_shape, dropout=0.3, unroll=False, use_bias=True),#, return_sequences=True # If another lstm is stacked behind the first LSTM, we need to set return_sequences to True
    #    # Try out Bidirection layers to also take into account the following values not only the preceeding
    #    #tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_layers, dropout=0.3, unroll=False, use_bias=True, return_sequences=True)), # recurrent_dropout=0.1 cannot be set, because cuDNN cannot be used then
    #    #tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_layers, dropout=0.3, unroll=False, use_bias=True)), # recurrent_dropout=0.1 cannot be set, because cuDNN cannot be used then
    #    # Additional Dense Layer(s) didn't help yet
    #    tf.keras.layers.Dense(future_target), #, activation="relu"
    #    # tf.keras.layers.Dropout(0.3),
    #    tf.keras.layers.Dense(future_target, activation="sigmoid"), # <- Only relu and sigmoid will yield values between 0 and 1 that seem reasonable, but softmax will create the sum of 1 over alle 265 values, which isn't feasible
    #])

                    # Liu, Bach in 2014 shows better performance with rmsprop             # checked with mse, categorical_crossentropy, binary_crossentropy
    music_lstm_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr), loss=[keras.losses.BinaryCrossentropy(), keras.losses.MeanSquaredError()], metrics=['binary_crossentropy', 'mae'])

    if summary:
        # Print out model information on parameters (important for calculating the size of VRAM)
        music_lstm_model.summary()

    return music_lstm_model