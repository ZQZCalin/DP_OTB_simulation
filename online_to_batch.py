import numpy as np 
import math
from matplotlib import pyplot as plt
from utils import get_parent_nodes
from loss import LinearLoss
from online_learner import Learner, OSD
from environment import Env, LinearRegression

def online_to_batch(T, dimen, alpha, learner:Learner, env:Env):
    """
    DP Online-to-batch conversion.
    Input:
        T: int, total number of rounds.
        dimen: int, dimension of parameter w.
        alpha: function, returns alpha(t) = alpha_t.
        learner: object, online learner.
        env: object, ML task environment, returns loss and gradient in the current round.
    Output:
        x: (1,d) array, final parameter $$x_T$$.
        ML_losses: list of losses in each round.
    """
    sum_alpha = 0   # running sum of alpha_t, i.e., $$\alpha_{1:t}$$
    sum_x = np.zeros(dimen)       # running sum of alpha_i*w_i, i.e., $$\alpha_{1:t}x_t$$
    x = np.zeros(dimen)

    grad_est = np.zeros(dimen)  # gradient estimate $$g_t=g_{t-1}+\delta_t$$

    tree_noise = np.random.normal(0, 0.01, (T, dimen))   # T x d, stores all noises in tree mechanism

    ML_losses = []

    for t in range(1, T+1):
        # 1. get w_t and compute x_t
        x_prev = x
        w = learner.get_param()
        """ $$\alpha_{1:t}x_t = \alpha_{1:t-1}x_{t-1} + \alpha_tw_t$$ """
        sum_alpha += alpha(t)
        sum_x += alpha(t) * w
        x = sum_x / sum_alpha

        # 2. compute delta_t, g_t, gamma_t
        grad_diff = alpha(t) * env.get_grad(x, t) - alpha(t-1) * env.get_grad(x_prev, t)
        grad_est += grad_diff
        noise = np.sum(tree_noise[get_parent_nodes(t)], axis=0).ravel()

        # 3. send loss to online learner and update
        """A few ways to implement linear loss fed to online learner (for test purpose),
        and the first two seem to experience high numerical instability."""
        # OL_loss = LinearLoss(grad_est + noise)              # noised gradient difference methodd
        # OL_loss = LinearLoss(grad_est)                      # un-noised gradient difference method
        OL_loss = LinearLoss(alpha(t)*env.get_grad(x, t))   # anytime OTB
        learner.update(OL_loss)

        # 4. performance analysis
        ML_losses.append(env.get_loss(x, t))

    return x, ML_losses

# test
if __name__ == "__main__":
    print("=====testing=====")

    T = 10000
    dimen = 2   # including bias term 
    alpha = lambda t : 1
    learner = OSD(dimen, lr=1/math.sqrt(T))

    """test on simple function y = x-5 with square error"""

    """
    # deterministic sampling is NOT working, we have to sample randomly
    indices = np.arange(T)
    np.random.shuffle(indices)
    X = np.linspace(10, 100, T)
    y = X-5
    """

    X = np.random.uniform(-10,10, (T))
    y = X - 5

    X = X.reshape((T,1))

    env = LinearRegression(X, y)

    x_final, ML_losses = online_to_batch(T, dimen, alpha, learner, env)
    print("="*50)
    print("online-to-batch completed")
    print(x_final)

    plt.yscale("log")
    plt.xscale("log")
    plt.plot(np.arange(T), ML_losses)
    plt.show()