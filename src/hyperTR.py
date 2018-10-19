import os
import tensorflow as tf

from keras import backend as K
from sklearn.metrics import confusion_matrix
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp

from helpers import *

import time
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


def create_model(params):
    reset_seeds()
    from keras.models import Sequential
    from keras.layers import Dense

    # Get sampled values
    _features = params['_features']
    _layers = 2# params['_layers']
    _l1nn = 8 # params['_l1nn']+2
    _l2nn = 8 #params['_l2nn']+2
    _act = 'tanh' # params['_act']
    #_lr = params['_lr']

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
                  optimizer=keras.optimizers.Adam(clipnorm=1.),
                  metrics=['accuracy'])
    model.fit(x_train_tmp, y_train_onehot,
              validation_data=(x_valid_tmp, y_valid_onehot),
              batch_size=128, epochs=100, verbose=0,
              shuffle=True,
              callbacks=[my_callback])

    score, acc = model.evaluate(x_tes_tmp, y_tes_onehot, verbose=0)

    return acc, STATUS_OK, model


def train_model(params):
    acc, status, _ = create_model(params)
    K.clear_session()
    return {'loss': -acc, 'status': status}

# Hyperparameter space
space={
    '_features':hp.choice('_features', allowed_indices)
    #'_layers' : hp.choice('_layers', [1,2]),
    #'_l1nn'   : hp.randint('_l1nn', 9),
    #'_l2nn'   : hp.randint('_l2nn', 9),
    #'_act'    : hp.choice('_act', ['relu','tanh']),
    #'_lr'     : hp.uniform('_lr', 0.001,0.1)
}

trials = Trials()

start = time.time()
best = fmin(train_model, space, algo=tpe.suggest, max_evals=1000, trials=trials,
            verbose=2, rstate=np.random.RandomState(2018))
end = time.time()


_results = [-tdict['loss'] for tdict in trials.results]

rdf = pd.DataFrame(_results)
cdf = pd.DataFrame(trials.vals)

cdf['_acc'] = rdf

cdf.to_csv('Res_hype.csv', index=False)

# In order to train model with the obtained values of hyperparameters 'choice' based
# values need to be assigned manually as indexes are sampled and interpreted by fmin().
'''
best['_features'] = allowed_indices[best['_features']]
best['_layers']=best['_layers']+1
if (best['_act'] == 0):
    best['_act'] = 'relu'
else:
    best['_act'] = 'tanh'
'''

acc, STATUS_OK, model = create_model(best)

features_list = _info.loc[best['_features']]['list']
x_tes_tmp   = x_tes[:,features_list]

y_predicted=model.predict(x_tes_tmp)
y_predicted=np.argmax(y_predicted,axis=1)
_cm = confusion_matrix(y_tes, y_predicted)

print('-----------------------------------------')
print('Best parameters: {0}'.format(best))
print('-----------------------------------------')
print('Values: {0}'.format(trials.vals))
print('-----------------------------------------')
print('Results: {0}'.format(np.sort(_results)))
print('-----------------------------------------')
print('Best model accuracy: {0:0.00}[%]'.format(acc*100))
print('-----------------------------------------')
print('Confusion matrix:')
print(_cm)

vals = np.sum(_cm,axis=1)
print(_cm*100.0/vals)

CM = pd.DataFrame(_cm)
CM.to_csv('CM.csv')

print(end-start)