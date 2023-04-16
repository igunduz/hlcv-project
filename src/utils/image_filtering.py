import math
import numpy as np
from scipy import signal


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def gauss(sigma):
    
    ### Your code here
    
    raise NotImplementedError
    
    return Gx, x


def gaussianfilter(img, sigma):
    
    ### Your code here
    
    raise NotImplementedError
    
    return outimage


def gaussdx(sigma):
    
    ### Your code here
    
    raise NotImplementedError
    
    return D, x


def gaussderiv(img, sigma):
    
    ### Your code here
    
    raise NotImplementedError
    
    return imgDx, imgDy