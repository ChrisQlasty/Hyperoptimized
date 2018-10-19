import os
import ray
import time
import objective
import tensorflow as tf

from helpers import *
from collections import defaultdict

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


ray.init()

#--------------------------
num_starting_segments = 10
num_segments = 1000
steps = 50

#---------------------------

reset_seeds()

x_train, x_valid, x_tes, y_train_onehot, y_valid_onehot,\
y_tes_onehot, y_train, y_valid, y_tes = get_data(myData, num_classes, ratio=0.7)

_info = features_info(n_features=len(col_names)-1, min_allowed=3)

allowed_indices = list(_info.index)


x_train = ray.put(x_train)
x_valid = ray.put(x_valid)
x_tes = ray.put(x_tes)

y_train_onehot = ray.put(y_train_onehot)
y_valid_onehot = ray.put(y_valid_onehot)
y_tes_onehot = ray.put(y_tes_onehot)


# Keep track of the accuracies that we've seen at different numbers of
# iterations.
accuracies_by_num_steps = defaultdict(lambda: [])

# Keep track of all of the experiment segments that we're running. This
# dictionary uses the object ID of the experiment as the key.
experiment_info = {}
# Keep track of the curently running experiment IDs.
remaining_ids = []

# Keep track of the best hyperparameters and the best accuracy.
best_hyperparameters = None
best_accuracy = 0

# A function for generating random hyperparameters.
def generate_hyperparameters():
    return {
            #'_features':np.random.choice(allowed_indices)
            #'_layers' : np.random.choice([1,2]),
            #'_l1nn'   : np.random.randint(9),
            #'_l2nn'   : np.random.randint(9),
            #'_act'    : np.random.choice(['relu','tanh']),
            '_lr'     : np.random.uniform(0.001,0.1)
         }

# Launch some initial experiments.
for _ in range(num_starting_segments):
    hyperparameters = generate_hyperparameters()
    experiment_id = objective.train_ffn_and_compute_accuracy.remote(
        hyperparameters, _info,
        x_train, y_train_onehot, x_valid, y_valid_onehot, x_tes, y_tes_onehot)
    experiment_info[experiment_id] = {"hyperparameters": hyperparameters,
                                      "total_num_steps": steps,
                                      "accuracies": []}
    remaining_ids.append(experiment_id)

start = time.time()

for _ in range(num_segments):
    # Wait for a segment of an experiment to finish.
    ready_ids, remaining_ids = ray.wait(remaining_ids, num_returns=1)
    experiment_id = ready_ids[0]
    # Get the accuracy and the weights.
    accuracy = ray.get(experiment_id)
    # Update the experiment info.
    previous_info = experiment_info[experiment_id]
    previous_info["accuracies"].append(accuracy)

    # Update the best accuracy and best hyperparameters.
    if accuracy > best_accuracy:
        best_hyperparameters = previous_info["hyperparameters"]
        best_accuracy = accuracy

    # --- here the hyperparameter optimization should come ---
    new_hyperparameters = generate_hyperparameters()
    new_info = {"hyperparameters": new_hyperparameters,
                "total_num_steps": steps,
                "accuracies": []}


    # Start running the next segment.
    new_experiment_id = objective.train_ffn_and_compute_accuracy.remote(
        new_hyperparameters, _info,
        x_train, y_train_onehot, x_valid, y_valid_onehot, x_tes, y_tes_onehot)
    experiment_info[new_experiment_id] = new_info
    remaining_ids.append(new_experiment_id)

    # Update the set of all accuracies that we've seen.
    accuracies_by_num_steps[previous_info["total_num_steps"]].append(
        accuracy)

end = time.time()

# Record the best performing set of hyperparameters.
print("""Best accuracy was {:.3} with
      learning_rate: {}    
  """.format(100 * best_accuracy, best_hyperparameters))

print(end-start)


results = [experiment_info[v]['accuracies'] for i,v in enumerate(experiment_info)]
configs = [experiment_info[v]['hyperparameters'] for i,v in enumerate(experiment_info)]

rdf = pd.DataFrame(results)
cdf = pd.DataFrame(configs)

cdf['_acc'] = rdf
cdf = cdf.dropna()

cdf.to_csv('Res_tune_rand.csv', index=False)

print('Finished')