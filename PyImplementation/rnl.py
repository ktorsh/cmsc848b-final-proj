from tools import *
from nlmeans import nlmeans
from tv import tv

def rnl(ima_nse, noise, param):
    M, N, T = ima_nse.shape

    param.dejitter = getoptions(param, 'dejitter', True)
    if T == 1:
        param.w = getoptions(param, 'w', 10)
        param.w_temp = getoptions(param, 'w_temp', 0)
        param._lambda = getoptions(param), 'lambda', 0.0015
    else:
        param.w = getoptions(param, 'w', 4)
        param.w_temp = getoptions(param, 'w_temp', 4)
        param._lambda = getoptions(param), 'lambda', 0.0075
    match noise.type:
        case 'gamma':
            param.N_iter = getoptions(param, 'N_iter', 500)
        case _:
            param.N_iter = getoptions(param, 'N_iter', 100)
    param.nsteps = getoptions(param, 'nsteps', (2 * param.w + 1)**2 * (2 * param.w_temp + 1) + param.N_iter)

    ima_res, ima_reduction = nlmeans(ima_nse, noise, param)
    param.w = ima_reduction
    ima_res = tv(ima_res, noise, param)
    return (ima_res, ima_reduction)