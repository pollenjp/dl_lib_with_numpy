import os,sys
import numpy as np
home_path = ".."
sys.path.append(home_path)

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.datasets import mnist # MNIST dataset

from layers.dense import Dense
import activations

epsilon = np.finfo(np.float32).eps
epochs = 100
batch_size = 64


################################################################################
#  Dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Flatten
X_train = X_train.reshape(X_train.shape[0], -1)
X_test  = X_test.reshape(X_test.shape[0], -1)
# Scaling
X_train = X_train.astype('float32') / 255
X_test  = X_test.astype('float32') / 255
# one-hot encoding
y_train = np.eye(N=10)[y_train]
y_test  = np.eye(N=10)[y_test]

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=10000)


################################################################################
def f_props(layers, x):
    for layer in layers:
        x = layer(x)
    return x


def b_props(layers, delta):
    batch_size = delta.shape[0]
    for i, layer in enumerate(layers[::-1]):
        if i == 0:
            layer.delta = delta
            layer.compute_grad()
        else:
            delta = layer.b_prop(delta, W)
            layer.compute_grad()
        W = layer.W

def update_params(layers, lr):
    for layer in layers:
        layer.W -= lr * layer.dW
        layer.b -= lr * layer.db

################################################################################
#  Model
layers = [
    Dense(input_dim=784, output_dim=100, activation=activations.Relu()),
    Dense(input_dim=100, output_dim=100, activation=activations.Relu()),
    Dense(input_dim=100, output_dim=10,  activation=activations.Softmax()),
]


def train(x, t, lr=0.01):
    y = f_props(layers, x)
    cost = (- t * np.log(np.clip(a=y, a_min=epsilon, a_max=y))).sum(axis=1).mean()

    delta = y - t
    b_props(layers, delta)
    update_params(layers, lr=lr)

    return cost

def estimate_valid(x, t):
    y = f_props(layers, x)
    cost = (- t * np.log(np.clip(a=y, a_min=epsilon, a_max=y))).sum(axis=1).mean()
    return cost, y


################################################################################
for epoch in range(epochs):
    X_train, y_train = shuffle(X_train, y_train)
    # オンライン学習
    N_data = X_train.shape[0]
    for idx in range(0, N_data, batch_size):
    #for _X, _y in zip(X_train, y_train):
        cost = train(x=X_train[idx:idx+batch_size], t=y_train[idx:idx+batch_size], lr=0.01)
    
    cost, y_pred = estimate_valid(X_valid, y_valid)
    accuracy = accuracy_score(y_valid.argmax(axis=1), y_pred.argmax(axis=1))
    print('EPOCH: {}, Valid Cost: {:.3f}, Valid Accuracy: {:.3f}'.format(epoch + 1, cost, accuracy))

