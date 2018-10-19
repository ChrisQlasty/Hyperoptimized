from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import time

from keras import backend as K
from hyperopt import hp

from helpers import *

import ray
from ray.tune import run_experiments, register_trainable
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import HyperOptSearch

ray.init(redirect_output=True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Dynamic memory allocate, used with GPU tests
# src: https://github.com/keras-team/keras/issues/4161#issuecomment-366031228
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess)

#--- Read data from file. Worksheet contains 10 coulmns - 9 features and 1 for class labels.
myData = pd.read_csv('/home/qlastyonthegol/Documents/tune tester/dtagain.csv', sep=',')

#--- Change labels from strings to integers (there are three classes in the dataset).
num_classes = 3
myData['class'].loc[myData['class'] == '1FS']=0
myData['class'].loc[myData['class'] == '2FJ']=1
myData['class'].loc[myData['class'] == '4FJ']=2

myData['class']= pd.to_numeric(myData['class'])

col_names = myData.columns.tolist()
col_names = col_names[-1:] + col_names[:-1]
myData = myData[col_names] # now class label is the first column in df

reset_seeds()

x_train, x_valid, x_tes, y_train_onehot, y_valid_onehot,\
y_tes_onehot, y_train, y_valid, y_tes = get_data(myData, num_classes, ratio=0.7)

_info = features_info(n_features=len(col_names)-1, min_allowed=3)

allowed_indices = list(_info.index)

# Set early stopping callback
my_callback = keras.callbacks.EarlyStopping(monitor='val_acc',
                                            min_delta= 1. / 100,  # [%]
                                            patience=10,
                                            verbose=0,
                                            mode='max')


def create_model(params, reporter):
    reset_seeds()
    from keras.models import Sequential
    from keras.layers import Dense

    # Get sampled values
    _features =  105 #params['_features']
    _layers = 2 #params['_layers']
    _l1nn = params['_l1nn']+2
    _l2nn = params['_l2nn']+2
    _act =  params['_act']
    _lr = params['_lr']

    features_list = _info.loc[_features]['list']
    n_inputs = _info.loc[_features]['sum']

    # select sampled set of input features
    x_train_tmp = x_train[:,features_list]
    x_valid_tmp = x_valid[:,features_list]
    x_tes_tmp   = x_tes[:,features_list]

    model = Sequential()
    model.add(Dense(_l1nn, input_shape=(n_inputs,), activation=_act))
    if(_layers==2):
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

    #return acc, STATUS_OK, model
    reporter(mean_acc=acc)


register_trainable("exp", create_model)


# Hyperparameter space
space={
    #'_features':hp.choice('_features',allowed_indices)
    #'_layers' : hp.choice('_layers',[1,2]),
    '_l1nn'   : hp.randint('_l1nn',9),
    '_l2nn'   : hp.randint('_l2nn',9),
    '_act'    : hp.choice('_act',['relu','tanh']),
    '_lr'     : hp.uniform('_lr',0.001,0.05)
}

config = {
    "my_exp": {
            "run": "exp",
            "num_samples": 1000
            }
        }

start = time.time()
algo = HyperOptSearch(space, max_concurrent=10, reward_attr="mean_acc")
scheduler = AsyncHyperBandScheduler(reward_attr="mean_acc")
train_results = run_experiments(config, search_alg=algo, scheduler=scheduler)
end = time.time()



results = [vvv['mean_acc'] for i, vvv in enumerate(item.last_result for item in train_results)]
configs = [vvv for i, vvv in enumerate(item.config for item in train_results)]

rdf = pd.DataFrame(results)
cdf = pd.DataFrame(configs)

cdf['_acc'] = rdf

cdf.to_csv('Res_tune_hype.csv', index=False)

print(end-start)
