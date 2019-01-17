import numpy as np
import cv2
import matplotlib.pyplot as plt
import math as m


# absolute Laplacian value
def contrast(imgs):
    n = len(imgs)
    h, w, c = imgs[0].shape
    W = np.zeros((h, w, n))
    for i in range(n):
        gray_img = cv2.cvtColor(np.uint8(imgs[i]), cv2.COLOR_BGR2GRAY)
        lap_img = cv2.Laplacian(gray_img, cv2.CV_64F, ksize=1)
        abs_img = np.abs(lap_img)
        W[:, :, i] = np.uint8(abs_img)

    return W


# standard deviation of color (R,G,B)
def saturation(imgs):
    n = len(imgs)
    h, w ,c= imgs[0].shape
    W = np.zeros((h, w, n))
    for i in range(n):
        imgs[i] = imgs[i]/255
        mu = (imgs[i][:, :, 0] + imgs[i][:, :, 1] + imgs[i][:, :, 2]) / 3
        S = np.sqrt((np.power(imgs[i][:, :, 0]-mu, 2) + np.power(imgs[i][:, :, 1]-mu, 2) + np.power(imgs[i][:, :, 2]-mu, 2))/3)
        W[:, :, i] = S

    return W


# middle value is the best of pixel values (gaussian weighting)
def exposure(imgs):
    n = len(imgs)
    h, w = imgs[0].shape[:2]
    W = np.zeros((h, w, n))

    for i in range(n):
        imgs[i] = imgs[i]/255
        sigma = 0.1
        Eb = np.exp(-0.5 * np.power(((imgs[i][:, :, 0])-0.5)/sigma, 2))
        Eg = np.exp(-0.5 * np.power(((imgs[i][:, :, 1])-0.5)/sigma, 2))
        Er = np.exp(-0.5 * np.power(((imgs[i][:, :, 2])-0.5)/sigma, 2))
        E = Eb * Eg * Er
        W[:, :, i] = E

    return W


def demoFeatures():
    img = cv2.imread('img/house/C.jpg')
    contrast_image = contrast([img])
    saturation_image = saturation([img])
    exposure_image = exposure([img])

    cv2.imshow('exposure', exposure_image)
    cv2.imshow('saturation', saturation_image)
    cv2.imshow('contrast', contrast_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    demoFeatures()
