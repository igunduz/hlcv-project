import math
import numpy as np
from scipy import signal


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def gauss(sigma):
    
    x = np.arange(-3*sigma,3*sigma+1)
    Gx = 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-x**2/(2*sigma**2))
    
    return Gx, x


def gaussianfilter(img, sigma):

    #kernel = np.outer(gauss(sigma), gauss(sigma))
    kernel = gauss(sigma)
    outimage = signal.convolve2d(img, kernel, boundary='symm', mode='same')
    
    return outimage


def gaussdx(sigma):
    
    x = np.arange(-3*sigma,3*sigma+1)
    D = -1/np.sqrt(2*np.pi*sigma**3)*x*np.exp(-x**2/(2*sigma**2))
    
    return D, x


def gaussderiv(img, sigma):

    G, _ = gauss(sigma)
    D, _ = gaussdx(sigma)
    Dx = signal.convolve(img, D[np.newaxis, :], mode='same')
    imgDx = signal.convolve(Dx, G[:, np.newaxis], mode='same')
    Dy = signal.convolve(img, D[:, np.newaxis], mode='same')
    imgDy = signal.convolve(Dy, G[np.newaxis, :], mode='same')
    
    return imgDx, imgDy