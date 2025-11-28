import matplotlib.pyplot as plt
import cv2
import numpy as np

from LAF import LAF
from NAF import NAF
from BAF import BAF

def normalize(image):
    mean = np.mean(image)
    std = np.std(image)
    return (image - mean) / (std + 1e-6)
    # return image / np.max(image)

def add_gaussian_noise(I, std=30):
    noise = np.random.normal(0, std, I.shape)
    noisy = I.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

def degradation_model(I, std=30, p=2):
    noise = np.random.normal(0, std, I.shape)
    blur = cv2.GaussianBlur(I, (0, 0), p)
    
    I_degraded = blur.astype(np.float32) + noise

    I_degraded = np.clip(I_degraded, 0, 255)
    return I_degraded.astype(np.uint8)

def plot_image(image, method):

    if method == 'LAF':
        filter_image_1 = LAF(image, 100, 0.05)
        filter_image_2 = LAF(image, 200, 0.05)
        filter_image_3 = LAF(image, 500, 0.05)
        title = ['Original', 'LAF 100', 'LAF 200', 'LAF 500']

    elif method == 'NAF':
        filter_image_1 = NAF(image, 100, 0.05, 0.3)
        filter_image_2 = NAF(image, 100, 0.05, 1)
        filter_image_3 = NAF(image, 100, 0.05, 3)
        title = ['Original', 'k = 0.3', 'k = 1', 'k = 3']

    elif method == 'BAF':
        filter_image_1 = BAF(image, 250, 0.05, 0.5)
        filter_image_2 = BAF(image, 250, 0.05, 1)
        filter_image_3 = BAF(image, 250, 0.05, 1.25)
        title = ['Original', 'k = 0.5', 'k = 1', 'k = 1.25']


    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title(title[0])
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(filter_image_1, cmap='gray')
    plt.title(title[1])
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(filter_image_2, cmap='gray')
    plt.title(title[2])
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(filter_image_3, cmap='gray')
    plt.title(title[3])
    plt.axis('off')

    plt.tight_layout()
    plt.show()

image = cv2.imread("camera_man.png")
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
noise_image = add_gaussian_noise(grayscale_image)
# noise_image = degradation_model(grayscale_image)
noise_image = normalize(noise_image)
# cv2.imshow("Degradation Model", grayscale_image)
# cv2.waitKey(0)
# cv2.destroyWindow("q")
plot_image(noise_image, 'NAF')