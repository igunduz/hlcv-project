import numpy as np
import matplotlib.pyplot as plt

from .object_identification import find_best_match


def plot_rpc(D, plot_color):
    
    """
    Compute and plot the recall/precision curve
    D - square matrix, D(i, j) = distance between model image i, and query image j
    Note: assume that query and model images are in the same order, i.e. correct answer for i-th query image is the i-th model image
    """

    recall = []
    precision = []
    total_imgs = D.shape[1]

    num_images = D.shape[0]
    assert(D.shape[0] == D.shape[1])

    labels = np.diag([1]*num_images)

    d = D.reshape(D.size)
    l = labels.reshape(labels.size)

    sortidx = d.argsort()
    d = d[sortidx]
    l = l[sortidx]

    # Compute precision and recall values and append them to "recall" and "precision" vectors
    tp = 0
    fp = 0
    for idx in range(len(d)):
        tp = tp + l[idx]
        fp = fp + (1 - l[idx])
        precision.append(tp / (tp + fp))
        recall.append(tp / (tp + (total_imgs - tp)))

    plt.plot([1-precision[i] for i in range(len(precision))], recall, plot_color+'-')


def compare_dist_rpc(model_images, query_images, dist_types, hist_type, num_bins, plot_colors):

    assert len(plot_colors) == len(dist_types)
    for idx in range(len(dist_types)):#
        [_, D] = find_best_match(model_images, query_images, dist_types[idx], hist_type, num_bins)
        plot_rpc(D, plot_colors[idx])
        plt.axis([0, 1, 0, 1]);
        plt.xlabel('1 - precision');
        plt.ylabel('recall');
        # legend(dist_types, 'Location', 'Best')
        plt.legend(dist_types, loc='best')
