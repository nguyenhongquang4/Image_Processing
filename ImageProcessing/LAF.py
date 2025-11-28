import numpy as np
from scipy.ndimage import map_coordinates

def advect(I, u, v):
    H, W = I.shape
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    x_new = x - u
    y_new = y - v

    x_new = np.clip(x_new, 0, H - 1)
    y_new = np.clip(y_new, 0, W - 1)

    I_new = map_coordinates(I, [y_new, x_new], order=1, mode='nearest')

    return I_new

def LAF(I, n, tau):
    I = I.astype(np.float32)

    for _ in range(n):
        phi = np.random.uniform(0, 2 * np.pi, size=I.shape)

        u = np.cos(phi)
        v = np.sin(phi)
        I = advect(I, tau * u, tau * v)
    return I


