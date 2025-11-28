import cv2
import numpy as np

from LAF import advect
def gradient_magnitude(I):
    Ix = cv2.Sobel(I, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(I, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(Ix, Iy)

    return Ix, Iy, magnitude

def NAF(I, n, tau_0, k):
    I = I.astype(np.float32)

    for _ in range(n):
        phi = np.random.uniform(0, 2 * np.pi, size=I.shape)

        u = np.cos(phi)
        v = np.sin(phi)

        _, _, grad = gradient_magnitude(I)
        tau = tau_0  / (1 + (grad / k) ** 2)
        I = advect(I, tau * u, tau * v)

    return I





