# Deep Learning with Keras
Authors: Moorissa Tjokro, Jager Hartman

## Overview
We will be using the [Keras Sequential Interface](https://keras.io/getting-started/sequential-model-guide/) and the following datasets to run two multilayer perceptron models and two convolutional neural network models for building models:
* Iris
* MNIST
* SVHN
* Pets dataset with a variety of dog types.

## Tasks
*Codes can be found in github folders*

### Task 1
We model using a multilayer perceptron (feed forward neural network) with two hidden layers and rectified linear nonlinearities on the iris dataset. We then evaluated our model selection on an independent test-set.

#### Evaluation
* Test loss score: 0.121
* Test accuracy score: 0.974
![](task1/task1.png)


### Task 2
We train a multilayer perceptron on the MNIST dataset. We then compare a “vanilla” model with a model using drop-out. The scoring evaluation and result visualization of the learning curves can be found below:

#### Model with No Dropout
* Test loss for No Dropout: 0.237
* Test Accuracy for No Dropout: 0.975

#### Model with Dropout
* Test loss for Dropout: 0.119
* Test Accuracy for Dropout: 0.976

#### Learning curves between drop out and no dropout:
![Dropout](task2/task2a.png)
![No Dropout](task2/task2b.png)


### Task 3
We first train a convolutional neural network on the SVHN dataset using a single digit classification. We then build a model using batch normalization, then compare it against other approaches.

#### Base model without Batch Normalization
* Accuracy on test set: 86.34%
![Without Batch](task3/task3_withoutBatch.png)

#### Base model with Batch Normalization
* Accuracy on test set: 85.23%
![With Batch](task3/task3_withBatch.png)


### Task 4
We will be working with the 37 class classification task using [pets dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/). We will be using the weights of a pre-trained convolutional neural network like AlexNet or VGG for feature extraction and linear modeling. The weights are loaded into keras and features are computed using a forward pass to be then stored in disk. Then a linear model or MLP will be trained on the resulting features.

* Accuracy score on test set: 0.8765
![](task4/task4.png)
