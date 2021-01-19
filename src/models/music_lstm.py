import tensorflow as tf
from tensorflow.python.keras.layers.core import Dropout

                                     # output size (for us=(88)*3 + 1  = 265)
def get_model(input_shape, lr=0.001, future_target=265, summary=True, lstm_layers=16):
    music_lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(lstm_layers, input_shape=input_shape, return_sequences=True, dropout=0.3, unroll=False, use_bias=True), # If another lstm is stacked behind the first LSTM, we need to set return_sequences to True
        # Try out Bidirection layers to also take into account the following values not only the preceeding
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_layers, dropout=0.3, unroll=False, use_bias=True, return_sequences=True)), # recurrent_dropout=0.1 cannot be set, because cuDNN cannot be used then
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_layers, dropout=0.3, unroll=False, use_bias=True)), # recurrent_dropout=0.1 cannot be set, because cuDNN cannot be used then
        # Additional Dense Layer(s) didn't help yet
        tf.keras.layers.Dense(future_target, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(future_target, activation="relu"), # <- Only Relu will yield values between 0 and 1 that seem reasonable, but softmax should too, but doesn't
    ])

                    # Liu, Bach in 2014 shows better performance with rmsprop             # checked with mse, categorical_crossentropy, binary_crossentropy
    music_lstm_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr), loss='categorical_crossentropy', metrics=['categorical_crossentropy', 'mse'])

    if summary:
        # Print out model information on parameters (impornant for calculating the size of VRAM)
        music_lstm_model.summary()

    return music_lstm_model