import numpy as np

################################################################################
class Dense():
    """
    Parameters
    ----------
    - input_dim : int
    - output_dim : int
    - activation : class instance
        - methods
            - forward
            - backward
    """
    #def __init__(self, shape, ):
    def __init__(self, input_dim, output_dim, activation):
        self.W = np.random.uniform(low=-0.08, high=0.08, size=(input_dim, output_dim)).astype('float64')
        self.b = np.zeros(output_dim).astype('float64')

        self.activation = activation

        self.x = None
        self.u = None

        self.dW = None
        self.db = None

        self.params_idxs = np.cumsum([self.W.size, self.b.size])

    def __call__(self, x):
        self.x = x
        self.u = np.matmul(self.x, self.W) + self.b
        return self.activation.forward(self.u)

    def b_prop(self, delta, W):
        self.delta = self.activation.backward(self.u) * np.matmul(delta, W.T)
        return self.delta
    
    def compute_grad(self):
        batch_size = self.delta.shape[0]
        
        self.dW = np.matmul(self.x.T, self.delta) / batch_size
        self.db = np.matmul(np.ones(batch_size), self.delta) / batch_size

    def get_params(self):
        return np.concatenate([self.W.ravel(), self.b], axis=0)
    
    def set_params(self, params):
        _W, _b = np.split(params, self.params_idxs)[:-1]
        self.W = _W.reshape(self.W.shape)
        self.b = _b
    


