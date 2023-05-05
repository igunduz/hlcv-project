import math
import numpy as np
from scipy import signal


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def gauss(sigma):
    
    Gx = 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-np.arange(-3*sigma,3*sigma+1)**2/(2*sigma**2))
    x = np.arange(-3*sigma,3*sigma+1)
    
    return Gx, x


def gaussianfilter(img, sigma):

    #kernel = np.outer(gauss(sigma), gauss(sigma))
    kernel = gauss(sigma)
    outimage = signal.convolve2d(img, kernel, boundary='symm', mode='same')
    
    return outimage


def gaussdx(sigma):
    
    ### Your code here
    
    raise NotImplementedError
    
    return D, x


def gaussderiv(img, sigma):
    
    ### Your code here
    
    raise NotImplementedError
    
    return imgDx, imgDy