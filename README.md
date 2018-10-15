# Hyperoptimized
Once I utilized a grid-search method to find good set of hyperparameters and input features for a small feed-forward neural network. The trainings were performed in MATLAB and almost 50 000 combinations were considered. The aim for the models was to classify dataset containing observations of 3 hand poses. Each pose was described as a set of features and a lablel. The achieved accuracy on the testing set was 90.02%. More details can be found in [1]. <br><br>
In this approach the task is to achieve more than 90% accuracy using significantly lower number of trials thanks to the hyperopt library [2]. Neural network models are to be constructed utilizing Keras with the Tensorflow backend [3].


---
### References
[1] [*"Pose classification in the gesture recognition using the linear optical sensor"* K. Czuszynski, J. Ruminski, J. Wtorek](http://ieeexplore.ieee.org/document/8004989/)  
[2] [*Hyperopt: Distributed Asynchronous Hyper-parameter Optimization*](https://github.com/hyperopt/hyperopt)  
[3] [*Keras: Deep Learning for humans*](https://github.com/keras-team/keras)
