import numpy as np
from gaussfilter import *

def criteria(I, J, noise):
    match noise.type:
        case 'gauss':
            return (I - J) ** 2 / (2 * noise.sig ** 2)
        case 'poisson':
            return xlogx(I / noise.Q) + xlogx(J / noise.Q) - 2 * xlogx((I + J) / noise.Q / 2)
        case 'gamma':
            return (I - J) ** 2 / (noise.nlf(I) + noise.nlf(J))
    
    def xlogx(x): 
        y = x * np.log(x)
        y[x == 0] = 0

def build_patch_shape(M, N, T, hP, ht, shape):
    (X, Y, P) = meshgridperiodic(M, N, T)
    match shape:
        case 'circular':
            return Y ** 2 + X ** 2 <= (hP + 0.5) ** 2 and P <= ht
        case 'square':
            return np.abs(Y) <= hP and np.abs(X) <= hP and np.abs(P) <= ht
        case 'gaussian':
            return np.exp(-(Y ** 2 + X ** 2) / (0.6 * hP ** 2)) and np.abs(P) <= ht
        case 'triangle':
            return np.abs(Y) <= hP and np.abs(X) <= hP and Y <= X and np.abs(P) <= ht
    
    def meshgridperiodic(M, N=1, T=1):
        (Y, X, Z) = np.meshgrid(np.min((1..N) - 1, N + 1 - (1..N)), \
                                np.min((1..M) - 1, M + 1 - (1..M)), \
                                np.min((1..T) - 1, T + 1 - (1..T)))
        return (X, Y, Z)

def central_pix(img_sample, param):
    p = param.p
    ptemp = param.p_temp
    shape = param.shape
    noise = param.noise
    blur = param.blur
    _, _, T = img_sample.shape
    x
    if T == 1:
        x = np.mean(img_sample[:]) * np.ones((512, 512))
    else:
        x = np.mean(img_sample[:]) * np.ones(64, 64, np.min(T, 64))
    M, N, T = x.shape

    patch_shape = np.zeros(M, N, T)
    patch_shape = build_patch_shape(M, N, T, p, ptemp, shape)
    patch_size = np.sum(patch_shape[:])
    # if np.sum(patch_shape(patch_shape))

    img_nse1 = noisegen(x, noise)
    img_nse2 = noisegen(x, noise)
    img_nse1 = gaussfilter(img_nse1, blur)
    img_nse2 = gaussfilter(img_nse2, blur)

    patch_nse1 = np.zeros(M * N * T, patch_size)
    patch_nse2 = np.zeros(M * N * T, patch_size)
    n = 1
    for k in range(-(p + 1), p + 1):
        for l in range(-(p + 1), (p + 1)):
            for t in range(-(ptemp + 1), ptemp + 1):
                if patch_shape[np.mod(k, M) + 1, np.mod(l, N) + 1, np.mod(t, T) + 1] == 1:
                    patch_nse1[:, n] = np.reshape(circshift(img_nse1, (k, l, t)), (M * N * T, 1))
                    patch_nse2[:, n] = np.reshape(circshift(img_nse2, (k, l, t)), (M * N * T, 1))
                    n += 1
    
    d = criteria(patch_nse1, patch_nse2[randperm(M * N * T), :], noise)
    d = np.mean(d, 2)

    m_criteria = np.mean(d)
    s_criteria = np.std(d)

def nlmeans(ima_nse, noise, param):
    M, N, T = ima_nse.shape

    

    patch_shape = build_patch_shape(M, N, T, hP, ht, shape)
    patch_size = np.sum(patch_shape[:])
    patch_shape = patch_shape / patch_size
    patch_shape = np.conj(np.fft.fftn(patch_shape))

    ima_cmp = gaussfilter(ima_nse, blur)

    sum_w = np.zeros(M, N, T)
    sum_w2 = np.zeros(M, N, T)
    sum_wI = np.zeros(M, N, T)
    if dejitter:
        sum_wI2 = np.zeros(M, N, T)
    hT = np.min(hT, T)

    step = 0
    for dx in range(-hW, hW):
        for dy in range(-hW, hW):
            for dz in range(-hT, hT):
                step += 1

                if (dx == 0 and dy == 0 and dz == 0) or dx^2 + dy^2 > (hW+0.5)^2:
                    continue
                x2range = np.mod(range(1, M) + dx - 1, M) + 1
                y2range = np.mod(range(1, N) + dy - 1, N) + 1
                z2range = np.mod(range(1, T) + dz - 1, T) + 1

                diff = criteria(ima_cmp, ima_cmp(x2range, y2range, z2range), noise)
                diff = np.real(np.fft.ifftn(patch_shape * np.fft.fftn(diff)))

                w = np.exp(-np.abs(diff - m_criteria) / (s_criteria * tau ** 2))

                if block:
                    w = np.real(np.fft.ifftn(patch_size * patch_shape * np.fft.ifftn(w)))

                sum_w = sum_w + w
                sum_w2 = sum_w2 + w
                sum_wI = sum_wI + w * ima_nse(x2range, y2range, z2range)
                if dejitter:
                    sum_wI2 = sum_wI2 + w * ima_nse(x2range, y2range, z2range) ** 2
                
    w_center = 1
    sum_w = sum_w2 + w_center ** 2
    sum_wI = sum_wI + w_center * ima_nse
    if dejitter:
        sum_wI2 = sum_wI2 + w_center * ima_nse ** 2

    ima_fil = sum_wI / sum_w
    ima_nbpixels = sum_w ** 2 / sum_w2

    if dejitter:
        ima_var = sum_wI2 / sum_w - ima_fil ** 2
        ima_var_predict = noise.nlf(ima_fil)
        ima_var_x = np.abs(ima_var - ima_var_predict)
        alpha = ima_var_x / (ima_var_x + ima_var_predict)
        ima_fil = (1 - alpha) * ima_fil + alpha * ima_nse
        ima_nbpixels = ima_nbpixels / \
            ((1-alpha) ** 2 + \
            ((alpha ** 2 + \
            2 * alpha * (1 - alpha) / sum_w) * ima_nbpixels))
        
    ima_reduction = (ima_nbpixels) ** (1/2)
