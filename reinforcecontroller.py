import tensorflow as tf


def controllermodel(controller_lstm_dim,controller_input_shape,controller_classes):
    model= tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(controller_lstm_dim, return_sequences=True, input_shape = controller_input_shape, name='main_input'))
    model.add(tf.keras.layers.Dense(controller_classes, activation = 'softmax', name = 'main_output'))
    return model