import keras
import numpy as np
import pandas as pd

# Reset seeds to make reproducible results
def reset_seeds():
    from numpy.random import seed
    seed(1)
    from tensorflow import set_random_seed
    set_random_seed(2)

# Get data from a dataframe, shuffle and divide it into training/validation/testing sets
def get_data(myData, num_classes, ratio):
    myData = myData.as_matrix()
    np.random.shuffle(myData)
    train_prc = int(len(myData) * ratio)
    valid_prc = int(len(myData) * (1-ratio)/2)
    test_prc = int(len(myData) * (1-ratio)/2)

    x_train = myData[0:train_prc, 1:]
    y_train = myData[0:train_prc, 0]

    x_valid = myData[train_prc:train_prc + valid_prc, 1:]
    y_valid = myData[train_prc:train_prc + valid_prc, 0]

    x_tes = myData[train_prc + valid_prc:, 1:]
    y_tes = myData[train_prc + valid_prc:, 0]

    y_train_onehot = keras.utils.to_categorical(y_train, num_classes)
    y_valid_onehot = keras.utils.to_categorical(y_valid, num_classes)
    y_tes_onehot = keras.utils.to_categorical(y_tes, num_classes)

    return x_train, x_valid, x_tes, \
           y_train_onehot, y_valid_onehot, y_tes_onehot, \
           y_train, y_valid, y_tes

# Function for finding a list of '1's in a zero padded binary representation of a number
# e.g. : output  = list_indices('00101')
# print(output)
# >> [2,4]
def list_indices(stri):
    return [_ind for _ind, _val in enumerate(stri) if _val=='1']

# Function for finding allowed combinations of input features.
# n_features - number of input features considered
# min_allowed - minimal number of features in input combination
# Assume we have n_features and we are interested in checking how models learn
# with different combinations (e.g. features 1,2,4; 1,4,6,9 etc.). However, we may
# not want to consider combinations composed of less than min_allowed features.
# Function outputs lookup tables and the _info.index is the argument for defining
# a choice list in hyperparameter space.
def features_info(n_features, min_allowed):
    # total number of combinations
    combinations = 2 ** n_features
    # Construction of DF of the length equal to initial combinations number
    _info = pd.DataFrame(data={'binary': range(combinations)})
    # Overwrite 'binary' column with binary representation of a row number
    _info['binary'] = _info.index.map(np.binary_repr)
    # Count how many '1's (features to select) are in a given binary representation
    _info['sum'] = _info['binary'].apply(str.count, args=('1'))
    # Drop all rows, which have not enough number of selected features ('1's)
    _info = _info.drop(_info.index[_info['sum'] < min_allowed])
    _info = _info.drop(_info.index[_info['sum'] > 4])
    # Make zero padding of binary representations to have n_features long strings
    _info.binary = _info['binary'].apply(str.rjust, args=(n_features, '0'))
    # Find indices where '1's are present - this will select proper columns from data
    _info['list'] = _info.binary.apply(list_indices)

    return _info