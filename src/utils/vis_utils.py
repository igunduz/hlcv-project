from builtins import range

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

from math import sqrt, ceil
import numpy as np

background = [200, 222, 250]  # Light Sky Blue
c1 = [0, 0, 205]  # ok
c2 = [34, 139, 34]  # ok
c3 = [192, 192, 128]  # 3
c4 = [165, 42, 42]  # ok
c5 = [128, 64, 128]  # 5
c6 = [204, 102, 0]  # 6
c7 = [184, 134, 11]  # ok
c8 = [0, 153, 153]  # ok
c9 = [0, 134, 141]  # ok
c10 = [184, 0, 141]  # ok
c11 = [184, 134, 0]  # ok
c12 = [184, 134, 223]  # ok
c13 = [43, 134, 141]  # ok
c14 = [11, 23, 141]  # ok
c15 = [14, 34, 141]  # ok
c16 = [14, 134, 41]  # ok
c17 = [233, 14, 241]  # ok
c18 = [182, 24, 241]  # ok
c19 = [123, 13, 141]  # ok
c20 = [13, 164, 141]  # ok
c21 = [84, 174, 141]  # ok
c22 = [184, 14, 41]  # ok
c23 = [184, 34, 231]  # ok

label_colours = np.array(
    [background, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23])
# label_mono = np.array([0, 1, 2 , 3, 4, 5, 6 ,7, 8, 9])


classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
           "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def visualize_grid(Xs, ubound=255.0, padding=1):
    """
    Reshape a 4D tensor of image data to a grid for easy visualization.

    Inputs:
    - Xs: Data of shape (N, H, W, C)
    - ubound: Output grid will have values scaled to the range [0, ubound]
    - padding: The number of blank pixels between elements of the grid
    """
    (N, H, W, C) = Xs.shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                # grid[y0:y1, x0:x1] = Xs[next_idx]
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    # grid_max = np.max(grid)
    # grid_min = np.min(grid)
    # grid = ubound * (grid - grid_min) / (grid_max - grid_min)
    return grid

def vis_grid(Xs):
    """ visualize a grid of images """
    (N, H, W, C) = Xs.shape
    A = int(ceil(sqrt(N)))
    G = np.ones((A*H+A, A*W+A, C), Xs.dtype)
    G *= np.min(Xs)
    n = 0
    for y in range(A):
        for x in range(A):
            if n < N:
                G[y*H+y:(y+1)*H+y, x*W+x:(x+1)*W+x, :] = Xs[n,:,:,:]
                n += 1
    # normalize to [0,1]
    maxg = G.max()
    ming = G.min()
    G = (G - ming)/(maxg-ming)
    return G

def vis_nn(rows):
    """ visualize array of arrays of images """
    N = len(rows)
    D = len(rows[0])
    H,W,C = rows[0][0].shape
    Xs = rows[0][0]
    G = np.ones((N*H+N, D*W+D, C), Xs.dtype)
    for y in range(N):
        for x in range(D):
            G[y*H+y:(y+1)*H+y, x*W+x:(x+1)*W+x, :] = rows[y][x]
    # normalize to [0,1]
    maxg = G.max()
    ming = G.min()
    G = (G - ming)/(maxg-ming)
    return G
