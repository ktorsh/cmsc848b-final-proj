import numpy as np


def gaussfilter(img, s):
    M, N, T = img.shape
    s = s * np.sqrt(2) / 2
    cs = np.ceil(s)

    cx = np.min(cs, np.floor(M / 2))
    cy = np.min(cs, np.floor(N / 2))
    cz = np.min(cs, np.floor(T / 2))

    Y, X, Z = np.meshgrid(range(-cx, cx), range(-cy, cy), range(-cz, cz))
    disk = np.exp(-np.pi * (X**2 + Y**2 + Z**2)) / (s + 1/2)**2
    kernel = disk / np.sum(disk[:])

    kernel_zp = np.zeros(M, N, T)
    kernel_zp[np.mod(range(-cx, cx), M) + 1, np.mod(range(-cy, cy), M) + 1, np.mod(range(-cz, cz), M) + 1] = kernel
    kernel_zp = np.conj(np.fft.fftn(kernel_zp))
    img = np.real(np.fft.ifftn(np.fft.fftn(img) * kernel_zp))