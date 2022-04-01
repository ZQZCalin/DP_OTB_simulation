import numpy as np 

class Learner(object):
    """
    Online Learner.
    In each round, receives loss function and updates w_{t+1}.
    ** for consistency, parameter w is of size (d,)
    """

    def __init__(self, d, w=np.array([])):
        self.d = d 
        if len(w) > 0:
            assert w.shape == (d,), f"invalid parameter shape, should be {(d,)}"
        else:
            w = np.zeros(d)
        self.w = np.array(w, dtype=float)

    def get_param(self):
        """get current parameter w"""
        return self.w
    
    # def update(self, loss):
    #     """update_call wrapper"""
    #     self.update_call(loss)
    #     return self.get_param()

    def update(self, loss):
        """
        Update parameter w_t+1 given a loss function;
        and update acumulated loss (if necessary).
        Input:
            loss: object that implements a loss function $$\ell$$ fed to the learner, with two methods:
                1. return_loss: given w, return $$\ell(w)$$.
                2. return_grad: given w, return gradient $$\nabla\ell(w)$$.
        """
        raise NotImplementedError

class OSD(Learner):
    """
    Online Subgradient Descent.
    $$w_{t+1} = w_t - \eta_t g_t$$
    """

    def __init__(self, d, lr=0.01):
        super(OSD, self).__init__(d)
        self.lr = lr

    def update(self, loss):
        self.w -= self.lr * loss.get_grad(self.w)