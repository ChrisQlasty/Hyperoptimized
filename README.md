# Hyperoptimized
Once I utilized a grid-search method to find good set of hyperparameters and input features for a small feedforward neural network. Almost 50 000 trials were considered and the trainings lasted for around 2 days (I used MATLAB on not very fast computer for this purpose). The aim for the models was to classify instances from dataset containing observations of 3 hand poses. Each pose was described as a set of features and a label. The achieved accuracy on the testing set was 90.02%. <br>
More details can be found in [[1]](http://ieeexplore.ieee.org/document/8004989/). <br><br>
In this approach the task is to achieve more than 90% accuracy using significantly lower number of trials and in much shorter time thanks to the _hyperopt_ library [[2]](https://github.com/hyperopt/hyperopt) and by trying to boost calculations speed using _ray_ [[3]](https://github.com/ray-project/ray) (and faster computer).<br>
Neural network models are to be constructed utilizing _Keras_ with the Tensorflow backend [[4]](https://github.com/keras-team/keras).

## Approaches to build training scripts
* First approach utilizes the hyperopt library only and runs training trials sequentially. The hyperparameter sampling is optimized. The code to the script can be found in the [hyperTR.py](./src/hyperTR.py) file.
* Second approach utilizes the ray library only and runs training trials in parallel. The hyperparameters here are sampled randomly through the whole experiment. The code to the script can be found in the [tuneTR.py](./src/tuneTR.py) file.
* Third approach utilizes both the hyperopt and ray libraries. It runs training trials in parallel and optimizes the hyperparameters. The code to the script can be found in the [hypertuneTR.py](./src/hypertuneTR.py) file. <br><br>
The second and third approach code is based on tutorials presented in ray repository.

### Input features combinations
In order to try different combinations of model input features a binary representation of a number was used. As there are 9 features in given dataset, one can have 2^9-1=511 combinations (0 features not considered). We can simply sample an integer from the range <1, 511>, convert it to binary, convert it to string and make left zero padding, e.g.:
 * number sampled: 123
 * binary representation: 1111011
 * zero padding (to make string of length 9 as this is the number of features): '001111011'
 * from the data array select only those columns where '1's appear <br>
We can also put additional restrictions and e.g. consider only combinations, which contain more than _n_min_ features and/or less than _n_max_ features. These operations are performed by funciton _features_info_ in the helpers.py file.

## Analysis and results
Attached [notebook](Performance_comparison.ipynb) compares performance of described approaches and presents some analysis of the obtained results.

### Other files
* [helpers.py](./src/helpers.py) contains some functions, which help keeping main scripts clear
  * _reset_seeds_ function is self explanatory 
  * _get_data_ performs division into training, validation and testing sets
  * _list_indices_ returns a list of positions in a string, where '1's are located
  * _features_info_ produces a list of allowed combinations of input features
 * [objective.py](./src/objective.py) is associated with a [tuneTR.py](./src/tuneTR.py) script
  
---
### References
[1] [*"Pose classification in the gesture recognition using the linear optical sensor"* K. Czuszynski, J. Ruminski, J. Wtorek](http://ieeexplore.ieee.org/document/8004989/)  
[2] [*Hyperopt: Distributed Asynchronous Hyper-parameter Optimization*](https://github.com/hyperopt/hyperopt)  
[3] [*Ray: A system for parallel and distributed Python that unifies the ML ecosystem*](https://github.com/ray-project/ray)  
[4] [*Keras: Deep Learning for humans*](https://github.com/keras-team/keras)
