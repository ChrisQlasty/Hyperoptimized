from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
import keras
import tensorflow as tf
from keras import backend as K


# Set early stopping callback
my_callback = keras.callbacks.EarlyStopping(monitor='val_acc',
                                            min_delta= 1. / 100,  # [%]
                                            patience=10,
                                            verbose=0,
                                            mode='max')


# Define a remote function that takes a set of hyperparameters as well as the
# data, constructs and trains a network, and returns the validation accuracy.
@ray.remote
def train_ffn_and_compute_accuracy(params, _info,
                                   x_train, y_train_onehot,
                                   x_valid, y_valid_onehot,
                                   x_tes, y_tes_onehot):
    # Extract the hyperparameters from the params dictionary.

    # Get sampled values
    _features = 105 #params['_features']
    _layers = 2 #params['_layers']
    _l1nn = 8 #params['_l1nn'] + 2
    _l2nn = 8 #params['_l2nn'] + 2
    _act = 'tanh' # params['_act']
    #_lr = params['_lr']

    features_list = _info.loc[_features]['list']
    n_inputs = _info.loc[_features]['sum']

    # select sampled set of input features
    x_train_tmp = x_train[:, features_list]
    x_valid_tmp = x_valid[:, features_list]
    x_tes_tmp = x_tes[:, features_list]

    num_classes = 3

    from keras.models import Sequential
    from keras.layers import Dense

    model = Sequential()
    model.add(Dense(_l1nn, input_shape=(n_inputs,), activation=_act))
    if (_layers == 2):
        model.add(Dense(_l2nn, activation=_act))

    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=_lr, clipnorm=1.),
                  metrics=['accuracy'])
    model.fit(x_train_tmp, y_train_onehot,
              validation_data=(x_valid_tmp, y_valid_onehot),
              batch_size=128, epochs=100, verbose=0,
              shuffle=True,
              callbacks=[my_callback])

    score, acc = model.evaluate(x_tes_tmp, y_tes_onehot, verbose=0)
    K.clear_session()

    return float(acc)

