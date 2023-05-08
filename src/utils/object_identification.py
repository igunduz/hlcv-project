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
    best_match = -1
    for i, query_hist in enumerate(query_hists):
        for j, model_hist in enumerate(model_hists):
            D[j, i] = get_dist_by_name(model_hist, query_hist, dist_type)
    best_match = np.argmin(D, axis=0)
    return best_match, D


def compute_histograms(image_list, hist_type, hist_isgray, num_bins):

    image_hist = []

    # Compute hisgoram for each image and add it at the bottom of image_hist
    for image in image_list:
        image_array = np.array(Image.open(image))
        if hist_isgray:
            image_array = rgb2gray(image_array)
        hist = get_hist_by_name(image_array.astype('double'), num_bins, hist_type)
        image_hist.append(hist)
    
    return np.array(image_hist)


def show_neighbors(model_images, query_images, dist_type, hist_type, num_bins):

    plt.figure()
    num_nearest = 5  # Show the top-5 neighbors
    
    [_, D] = find_best_match(model_images, query_images, dist_type, hist_type, num_bins)
    indices = np.apply_along_axis(lambda x : np.argpartition(x, 5), axis=0, arr=D)

    f, ax = plt.subplots(5,len(query_images))
    for i, query_image in enumerate(query_images):
        ax[0, i].imshow(Image.open(query_image))
        ax[0, i].axis('off')
        for j in range(1, num_nearest):
            ax[j, i].imshow(Image.open(model_images[indices[j, i]]))
            ax[j, i].axis('off')

    plt.show()

