"""
Created 29 September 2020
Bjarne Grimstad, bjarne.grimstad@gmail.no
"""

import tensorflow as tf


def build_model(hidden_units, reg=1e-10):
    """
    Build Neural network
    :param hidden_units:
    :param reg:
    :return:
    """
    layers = []

    for units in hidden_units:
        layers.append(tf.keras.layers.Dense(units=units, activation='relu',
                                            kernel_initializer=tf.keras.initializers.he_normal(),
                                            bias_initializer=tf.keras.initializers.constant(0.1),
                                            kernel_regularizer=tf.keras.regularizers.l2(reg),
                                            bias_regularizer=tf.keras.regularizers.l2(reg)))

    layers.append(tf.keras.layers.Dense(units=1, activation=None,
                                        kernel_initializer=tf.keras.initializers.he_normal(),
                                        bias_initializer=tf.keras.initializers.constant(0.0),
                                        kernel_regularizer=tf.keras.regularizers.l2(reg),
                                        bias_regularizer=tf.keras.regularizers.l2(reg)))

    return tf.keras.models.Sequential(layers)


def compile_model(model, learning_rate=0.01):
    """
    Compile NN model using Adam solver
    :param model: Keras model
    :param learning_rate: Learning rate
    :return: Compiled model
    """
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.mean_squared_error,
                  metrics=['mse', 'mae'])
    return model


def build_shallow_well_model():
    hidden_units = [20] * 2  # Shallow model
    model = build_model(hidden_units=hidden_units)
    compile_model(model, learning_rate=0.001)
    return model


def build_deep_well_model():
    hidden_units = [10] * 4  # Deep model
    model = build_model(hidden_units=hidden_units)
    compile_model(model, learning_rate=0.001)
    return model


def build_shallow_flowline_model():
    hidden_units = [50] * 2  # Shallow model
    model = build_model(hidden_units=hidden_units)
    compile_model(model, learning_rate=0.001)
    return model


def build_deep_flowline_model():
    hidden_units = [20] * 5  # Deep model
    model = build_model(hidden_units=hidden_units)
    compile_model(model, learning_rate=0.001)
    return model
