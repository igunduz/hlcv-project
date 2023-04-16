from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from .image_filtering import rgb2gray
from .image_histograms import is_grayvalue_hist, get_dist_by_name, get_hist_by_name


def find_best_match(model_images, query_images, dist_type, hist_type, num_bins):

    hist_isgray = is_grayvalue_hist(hist_type)

    model_hists = compute_histograms(model_images, hist_type, hist_isgray, num_bins)
    query_hists = compute_histograms(query_images, hist_type, hist_isgray, num_bins)

    D = np.zeros((len(model_images), len(query_images)))

    # Your code here
    
    raise NotImplementedError

    return best_match, D


def compute_histograms(image_list, hist_type, hist_isgray, num_bins):

    image_hist = []

    # Compute hisgoram for each image and add it at the bottom of image_hist
    # Your code here
    
    raise NotImplementedError

    return np.array(image_hist)


def show_neighbors(model_images, query_images, dist_type, hist_type, num_bins):

    plt.figure()
    num_nearest = 5  # Show the top-5 neighbors
    
    # Your code here
    
    [_, D] = find_best_match(model_images, query_images, dist_type, hist_type, num_bins)
    
    raise NotImplementedError


