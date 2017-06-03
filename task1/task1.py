import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
% matplotlib inline
plt.rcParams["figure.dpi"] = 120
np.set_printoptions(precision=3, suppress=True)
import keras

np.random.RandomState(seed=0)

from sklearn.datasets import load_iris
data = load_iris()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    data['data'], data['target'], random_state=0)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def make_model(optimizer='adam', hidden_size=32):
    model = Sequential([
        Dense(hidden_size, input_dim = 4),
        Activation('relu'),
        Dense(hidden_size),
        Activation('relu'),
        Dense(3),
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

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import SGD

def make_model(optimizer='adam', hidden_size=64):
    model = Sequential([
        Dense(hidden_size, input_dim = 4),
        Activation('relu'),
        Dense(hidden_size),
        Activation('relu'),
        Dense(3),
        Activation('softmax')
    ])        
    model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])
    return model

num_classes = 3
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = make_model()
model.fit(X_train, y_train,
          epochs=100,
          batch_size=3,
          validation_split=.1)
print('\nSummary:\n\n')
model.summary()

score = model.evaluate(X_test, y_test, batch_size=3, verbose=0)

print("\nTest loss: {:.3f}".format(score[0]))
print("Test Accuracy: {:.3f}".format(score[1]))

history_callback = model.fit(X_train, y_train, batch_size=64,
                             epochs=100, verbose=1, validation_split=.1)
def plot_history(logger):
    df = pd.DataFrame(logger.history)
    df[['acc', 'val_acc']].plot()
    plt.ylabel("accuracy")
    df[['loss', 'val_loss']].plot(linestyle='--', ax=plt.twinx())
    plt.ylabel("loss")

plot_history(history_callback)
plt.show()
