import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
% matplotlib inline
plt.rcParams["figure.dpi"] = 120
np.set_printoptions(precision=3, suppress=True)
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

np.random.RandomState(seed=0)


from keras.datasets import mnist
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784).astype('float32') / 255
X_test = X_test.reshape(10000, 784).astype('float32') / 255

print (X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

def make_model(optimizer='adam', hidden_size=32):
    model = Sequential([
        Dense(hidden_size, input_shape=[784,]),
        Activation('relu'),
        Dense(hidden_size),
        Activation('tanh'),
        Dense(10),
        Activation('softmax')
    ])
        
    model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])
    return model

clf = KerasClassifier(make_model)

param_grid = {'epochs' : [1, 3, 5],
              'hidden_size':  [10, 32, 64]}

grid = GridSearchCV(clf, param_grid = param_grid, cv = 5)
grid.fit(X_train, y_train)

print(grid.best_params_)
#hidden size = 256, epochs = 10
res = pd.DataFrame(grid.cv_results_)
res.pivot_table(index=["param_epochs", "param_hidden_size"],
                values=['mean_train_score', "mean_test_score"])


def make_model(optimizer='adam', hidden_size=64, dropout=0.5):
    model = Sequential([
        Dense(hidden_size, input_shape=[784,]),
        Activation('relu'),
        Dropout(rate = dropout),
        Dense(hidden_size),
        Activation('relu'),
        Dropout(rate = dropout),
        Dense(10),
        Activation('softmax')
    ])
        
    model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])
    return model

clf = KerasClassifier(make_model)

param_grid = {'epochs' : [5],
              'dropout' : [0.1, 0.2, 0.3]}

grid = GridSearchCV(clf, param_grid = param_grid, cv = 5)
grid.fit(X_train, y_train)

print(grid.best_params_)
#hidden size = 256, epochs = 10
res = pd.DataFrame(grid.cv_results_)
res.pivot_table(index=["param_epochs", "param_dropout"],
                values=['mean_train_score', "mean_test_score"])

import tensorflow as tf
#This doesn't work :(
num_classes = 10
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
print(y_train.shape, X_train.shape)

model_do = Sequential([
    Dense(64, input_shape=[784,]),
    Activation('relu'),
    Dropout(rate = 0.2),
    Dense(64),
    Activation('relu'),
    Dropout(rate = 0.2),
    Dense(10),
    Activation('softmax')
    ])

model_do.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

with tf.device('/gpu:0'):
    history_callback_do = model_do.fit(X_train, y_train, epochs=100, validation_split=.1)

model_nodo = Sequential([
    Dense(64, input_shape=[784,]),                                                                                      
    Activation('relu'),
    Dense(64),
    Activation('relu'),
    Dense(10),
    Activation('softmax')
    ])

model_nodo.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

with tf.device('/gpu:0'):
    history_callback_nodo = model_nodo.fit(X_train, y_train, epochs=100, validation_split=.1)

score = model_nodo.evaluate(X_test, y_test, batch_size=3, verbose=0)

print("\nTest loss for No Dropout: {:.3f}".format(score[0]))
print("Test Accuracy for No Dropout: {:.3f}".format(score[1]))

score = model_do.evaluate(X_test, y_test, batch_size=3, verbose=0)

print("\nTest loss for Dropout: {:.3f}".format(score[0]))
print("Test Accuracy for Dropout: {:.3f}".format(score[1]))

def plot_history(logger, title):
    df = pd.DataFrame(logger.history)
    df[['acc', 'val_acc']].plot()
    plt.ylabel("accuracy")
    df[['loss', 'val_loss']].plot(linestyle='--', ax=plt.twinx())
    plt.ylabel("loss")
    plt.title(title)
    plt.show()

plot_history(history_callback_do, title='DropOut')
plot_history(history_callback_nodo, title = 'No DropOut')
