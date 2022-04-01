import numpy as np
from loss import SquareError

class Env(object):
    """
    Environment of ML task, stores dataset and implements the loss for ML
    """

    def __init__(self):
        ()

    def predict(self, param, data):
        """
        Predict label based given param and data
        """
        return NotImplementedError
    
    def get_loss(self, param, t):
        """
        Given parameter and round index t (1-indexing), 
        return $$\ell(param, z_t)$$
        """
        return NotImplementedError

    def get_grad(self, param, t):
        """
        Wrapper around get_grad_call to return flattened gradient.
        Returns $$\ell(w,z_t)$$.
        """
        return self.get_grad_call(param, t).ravel()

    def get_grad_call(self, param, t):
        return NotImplementedError

class LinearRegression(Env):
    """
    Linear regression task.
    Predicts $$f(w,x)=<w,x>+b$$ with loss $$\ell(w,x)=(<w,x>+b-y)^2$$
    """

    def __init__(self, X, Y):
        """
        X: (N,d); y: (N,).
        """
        self.X = np.concatenate( (X, np.ones((len(X),1)) ), axis=1 )  # add 1 for bias terms
        self.Y = Y

    def predict(self, param, x):
        return np.dot(param[:-1], x) + param[-1]

    def get_loss(self, param, t):
        """Outputs loss: float"""
        x = self.X[t-1]
        y = self.Y[t-1]
        return (np.dot(param, x) - y)**2

    def get_grad_call(self, param, t):
        """Outputs grad: (1,d) array"""
        x = self.X[t-1]
        y = self.Y[t-1]
        return 2 * (np.dot(param, x) - y) * x