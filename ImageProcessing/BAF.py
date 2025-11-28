import numpy as np

from LAF import advect
from NAF import gradient_magnitude

def external_field(I, k):
    Ix, Iy, grad = gradient_magnitude(I)
    decrease_function = 1 / (1 + (grad / k) ** 2)
    u_f = - decrease_function * Ix
    v_f = - decrease_function * Iy
    return u_f, v_f

def BAF(I, n, tau_0, k):
    I = I.astype(np.float32)
    for _ in range(n):
        phi = np.random.uniform(0, 2 * np.pi, size=I.shape)
        _, _, grad = gradient_magnitude(I)

        u_f, v_f = external_field(I, k)
        tau = tau_0 / (1 + (grad / k) ** 2)
        u = np.cos(phi) + u_f
        v = np.sin(phi) + v_f

        I = advect(I, tau * u, tau * v)

    return I