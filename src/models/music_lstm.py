import tensorflow as tf

                                     # output size (for us=(88)*3 + 1  = 265)
def get_model(input_shape, lr=0.001, future_target=265, summary=True, lstm_layers=16):
    music_lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(lstm_layers, input_shape=input_shape),
        tf.keras.layers.Dense(future_target)
    ])

    music_lstm_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse', metrics=['mae'])

    if summary:
        # Print out model information on parameters (impornant for calculating the size of VRAM)
        music_lstm_model.summary()

    return music_lstm_model