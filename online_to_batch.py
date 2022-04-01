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

    tree_noise = np.random.normal(0, 10, (T, dimen))   # T x d, stores all noises in tree mechanism

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
        grad_now = env.get_grad(x, t)
        grad_prev = env.get_grad(x_prev, t)
        if t == 1:
            grad_diff = alpha(t) * grad_now
        else:
            grad_diff = alpha(t) * grad_now - alpha(t-1) * grad_prev
        grad_est += grad_diff
        noise = np.sum(tree_noise[get_parent_nodes(t)], axis=0).ravel()

        """testing mode"""
        if False:
            print(f"x: {x}; x_prev: {x_prev}")
            print(f"grad_now: {grad_now}")
            print(f"grad_prev: {grad_prev}")
            print(f"grad_diff: {grad_diff}")
            print(f"grad_est: {grad_est}")
            print(f"noise: {noise}")
            print("="*40)

        # 3. send loss to online learner and update
        """A few ways to implement linear loss fed to online learner (for test purpose)."""
        OL_loss = LinearLoss(grad_est + noise)              # noised gradient difference methodd
        # OL_loss = LinearLoss(grad_est)                      # un-noised gradient difference method (stable)
        # OL_loss = LinearLoss(alpha(t)*env.get_grad(x, t))   # anytime OTB (stable)
        learner.update(OL_loss)

        # 4. performance analysis
        ML_losses.append(env.get_loss(x, t))

    return x, ML_losses

# test
if __name__ == "__main__":
    print("="*50)
    print("start testing...")

    T = 100000
    dimen = 2   # including bias term 
    alpha = lambda t : t
    """we need to re-tune the step-size for most learners.
    If we set alpha_t=t, then eta=T^{-3/2} so that $$R_T(u)/\alpha_{1:T} = O(T^{-1/2})$$"""
    lr = 1/T**1.5
    learner = OSD(dimen, lr=lr)

    """test on simple function y = x-5 with square error"""

    """
    # deterministic sampling is NOT working, we have to sample randomly
    indices = np.arange(T)
    np.random.shuffle(indices)
    X = np.linspace(10, 100, T)
    y = X-5
    """

    # random sampling
    a = 1; b = -5
    X = np.random.uniform(0, 10, (T))
    y = a*X + b
    X = X.reshape((T,1))

    env = LinearRegression(X, y)

    x_final, ML_losses = online_to_batch(T, dimen, alpha, learner, env)
    print("="*50)
    print("online-to-batch completed")
    print(f"final weight is: {x_final} (should be {(a,b)})")

    plt.yscale("log")
    plt.xscale("log")
    start = 1000
    plt.plot(np.arange(start, T), ML_losses[start:])
    plt.show()