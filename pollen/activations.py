import numpy as np

################################################################################
class Relu():
    def __init__(self):
        pass

    def forward(self, x):
        return np.maximum(x, 0)

    def backward(self, x):
        return (x > 0).astype(x.dtype)

########################################
class Tanh():
    def forward(self, x):
        return np.tanh(x)

    def backward(self, x):
        return 1 - np.tanh(x) ** 2

################################################################################
class Sigmoid():
    def forward(self, x):
        return np.tanh(x * 0.5) * 0.5 + 0.5

    def backward(self, x):
        return self.forward(x) * (1 - self.forward(x))

########################################
class Softmax():
    def forward(self, x):
        x -= x.max(axis=1, keepdims=True)
        x_exp = np.exp(x)
        return x_exp / np.sum(x_exp, axis=1, keepdims=True)
    def deriv_softmax(x):
        return self.forward(x) * (1 - self.forward(x))


