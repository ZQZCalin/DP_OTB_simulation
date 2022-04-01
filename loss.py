import numpy as np 

class Loss(object):
    """
    implements a loss function $$\ell$$, 
    it has two methods:
        get_loss: return $$\ell(*args)$$.
        get_grad: return gradient $$\nabla\ell(*args)$$.
    """

    def __init__(self):
        return None 

    def get_loss(self, *args):
        return NotImplementedError

    def get_grad(self, *args):
        """wrapper around get_grad_call, returns a flattened (d,) array"""
        return self.get_grad_call(*args).ravel()

    def get_grad_call(self, *args):
        return NotImplementedError

# ============ Losses for Online Learners ============

class LinearLoss(Loss):
    """
    Linear loss $$\ell(w) = <g,w>$$.
    """

    def __init__(self, grad):
        self.grad = grad 
    
    def get_loss(self, w):
        return np.dot(self.grad, w)

    def get_grad_call(self, w):
        return self.grad

class L2RegLinearLoss(Loss):
    """
    Linear loss with 2-norm regularization,
    $$\ell(w) = <g,w> + R*\|w\|^2$$.
    """

    def __init__(self, grad, reg):
        self.grad = grad
        self.reg = reg

    def get_loss(self, w):
        return np.dot(self.grad, w) + self.reg * np.linalg.norm(w)^2

    def get_grad_call(self, w):
        return self.grad + 2*self.reg*w

# ============ ML Losses ============

class SquareError(Loss):
    """
    $$\ell(w;x,y) = (f(w;x)-y)^2$$
    Input:
        f: prediction function, predicts y_hat given (w,x).
        grad_f: returns gradient of f (w.r.t. w) at (w,x).
    """

    def __init__(self, f, grad_f):
        self.f = f 
        self.grad_f = grad_f

    def get_loss(self, w, x, y):
        return (self.f(w,x)-y)**2

    def get_grad_call(self, w, x, y):
        return 2*(self.f(w,x)-y)*self.grad_f(w,x)