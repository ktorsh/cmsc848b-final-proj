from tools import *
import numpy as np

def tv(img_nse, noise, param):
    M, N, T = img_nse.shape

    if T == 1:
        param._lambda = getoptions(param, 'lambda', 0.075)
    else:
        param._lambda = getoptions(param, 'lambda', 0.0375)

    img_fil = tv_primal_dual(img_nse, noise, param)

def tv_primal_dual(A, noise, param):
    n, m, t = A.shape

    _lambda = getoptions(param, 'lambda', 1)
    W = getoptions(param, 'W', np.ones(n, m, t))
    N_iter = getoptions(param, 'N_iter', 1000)
    nsteps = getoptions(param, 'nsteps', N_iter)

    if t == 1:
        sigma = 1 / 8**0.5
        tau = 1 / 8**0.5
    else:
        alpha = 1.5
        sigma = 1 / (8 + 4 * alpha**2)**0.5
        tau = 1 / (8 + 4 * alpha**2)**0.5
    theta = 1

    prox_Fet = lambda x: x / np.max(np.abs(x), 1)
    prox_G = lambda u: (u + tau / (_lambda * noise.sig**2) * W * A) / (1 + W * tau / (_lambda * noise.sig**2))
    G = lambda C: (W * (C - A)**2) / (2 * noise.sig**2)

    if t == 1:
        wgrad = np.gradient
        wdiv = grad
        wdiv = div
    else:
        WeightedField = lambda x: np.concatenate((x[:, :, :, 1:2], alpha * x[:, :, :, 3]), axis=3)
        wgrad = lambda x: WeightedField(grad(x))
        wdiv = lambda x: div(WeightedField(x))

    z = np.mean(A[:] * np.ones(A.shape))
    x_old = z
    y = wgrad(z)

    k_iter = 1
    step = nsteps - N_iter
    while k_iter <= N_iter:
        step += 1
        y = prox_Fet(y + sigma * wgrad(z))
        x = prox_G(x_old + tau * wdiv(y))
        z = x + theta * (x - x_old)
        x_old = x

        k_iter += 1