import numpy as np
import cv2

def grad(I):
    nd = I.ndim

    # Gradient
    if nd == 1:
        n = I.shape[0]
        dx = np.concatenate((I[1:], I[:1]), axis=0) - I
        dx[-1] = 0
        G = dx
    elif nd == 2:
        n, m = I.shape
        dx = np.concatenate((I[1:], I[:1]), axis=0) - I
        dy = np.concatenate((I[:, 1:], I[:, :1]), axis=1) - I
        G = np.stack((dx, dy), axis=nd)
        G[-1, :, 0] = 0
        G[:, -1, 1] = 0
    elif nd == 3:
        n, m, t = I.shape
        dx = np.concatenate((I[1:], I[:1]), axis=0) - I
        dy = np.concatenate((I[:, 1:], I[:, :1]), axis=1) - I
        dz = np.concatenate((I[:, :, 1:], I[:, :, :1]), axis=2) - I
        G = np.stack((dx, dy, dz), axis=nd)
        G[-1, :, :, 0] = 0
        G[:, -1, :, 1] = 0
        G[:, :, -1, 2] = 0
    else:
        raise ValueError(f'Grad for dimension {nd} not implemented')

    return G



def div(I):
    s = I.shape
    nd = I.ndim - 1
    if s[-1] != nd:
        raise ValueError(f"Div requires the vector field to be {nd} dimensional")
    n = s[0]
    
    # Divergence
    if nd == 1:
        tx = np.concatenate((np.array([n]), np.arange(1, n)))
        div = I - I[tx]
        div[0] = I[0]
        div[-1] = -I[-2]
    elif nd == 2:
        m = s[1]
        tx = np.concatenate((np.array([n]), np.arange(1, n)))
        ty = np.concatenate((np.array([m]), np.arange(1, m)))
        divx = I[:, :, 0] - I[tx, :, 0]
        divx[0] = I[0, :, 0]
        divx[-1] = -I[-2, :, 0]
        divy = I[:, :, 1] - I[:, ty, 1]
        divy[:, 0] = I[:, 0, 1]
        divy[:, -1] = -I[:, -2, 1]
        div = divx + divy
    elif nd == 3:
        m = s[1]
        t = s[2]
        tx = np.concatenate((np.array([n]), np.arange(1, n)))
        ty = np.concatenate((np.array([m]), np.arange(1, m)))
        tz = np.concatenate((np.array([t]), np.arange(1, t)))
        divx = I[:, :, :, 0] - I[tx, :, :, 0]
        divx[0, :, :] = I[0, :, :, 0]
        divx[-1, :, :] = -I[-2, :, :, 0]
        divy = I[:, :, :, 1] - I[:, ty, :, 1]
        divy[:, 0, :] = I[:, 0, :, 1]
        divy[:, -1, :] = -I[:, -2, :, 1]
        divz = I[:, :, :, 2] - I[:, :, tz, 2]
        divz[:, :, 0] = I[:, :, 0, 2]
        divz[:, :, -1] = -I[:, :, -2, 2]
        div = divx + divy + divz
    
    return div

def resample(img, Mnew, Nnew):
    M, N, T = img.shape
    fimg = np.fft.fft2(img)
    fimg = np.pad(fimg, ((0, 1), (0, 1), (0, 0)), mode='constant')
    
    hM = min(M // 2, Mnew // 2)
    if hM == hM // 1:
        hMn = hM - 1
        hMp = hM
    else:
        hMn = int(hM)
        hMp = int(hM)
        
    hN = min(N // 2, Nnew // 2)
    if hN == hN // 1:
        hNn = hN - 1
        hNp = hN
    else:
        hNn = int(hN)
        hNp = int(hN)
        
    fimg = fimg[:(hMp+1), :(hNp+1), :].copy()
    fimg = np.pad(fimg, ((0, max(Mnew-M, 0)), (0, max(Nnew-N, 0)), (0, 0)), mode='constant')
    fimg[hMn+1:, hNn+1:, :] = fimg[M-hMn:, N-hNn:, :]
    
    img = np.real(np.fft.ifft2(fimg))
    
    return img

def psnr(img1, img2): 
    return cv2.PSNR(img1, img2)

