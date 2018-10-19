# Hyperoptimized
Once I utilized a grid-search method to find good set of hyperparameters and input features for a small feed-forward neural network. The trainings were performed in MATLAB and almost 50 000 combinations were considered. The aim for the models was to classify dataset containing observations of 3 hand poses. Each pose was described as a set of features and a label. The achieved accuracy on the testing set was 90.02%. More details can be found in [[1]](http://ieeexplore.ieee.org/document/8004989/). <br><br>
In this approach the task is to achieve more than 90% accuracy using significantly lower number of trials thanks to the hyperopt library [[2]](https://github.com/hyperopt/hyperopt) and by trying to boost calculations speed using Ray [[3]](https://github.com/ray-project/ray). Neural network models are to be constructed utilizing Keras with the Tensorflow backend [[4]](https://github.com/keras-team/keras).

## Approaches to build training scripts
* First approach utilizes the hyperopt library only and runs training trials sequentially. The hyperparameter sampling is optimized. The code to the script can be found in the [hyperTR.py](./src/hyperTR.py) file.
* Second approach utilizes the ray library only and runs training trials in parallel. The hyperparameters here are sampled randomly through the whole experiment. The code to the script can be found in the [tuneTR.py](./src/tuneTR.py) file.
* Third approach utilizes both the hyperopt and ray libraries. It runs training trials in parallel and optimizes the hyperparameters. The code to the script can be found in the [hypertuneTR.py](./src/hypertuneTR.py) file.


---
### References
[1] [*"Pose classification in the gesture recognition using the linear optical sensor"* K. Czuszynski, J. Ruminski, J. Wtorek](http://ieeexplore.ieee.org/document/8004989/)  
[2] [*Hyperopt: Distributed Asynchronous Hyper-parameter Optimization*](https://github.com/hyperopt/hyperopt)  
[3] [*Ray: A system for parallel and distributed Python that unifies the ML ecosystem*](https://github.com/ray-project/ray)  
[4] [*Keras: Deep Learning for humans*](https://github.com/keras-team/keras)
