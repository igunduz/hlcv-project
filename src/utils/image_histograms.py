import numpy as np

from .image_filtering import gaussderiv


def normalized_histogram(img_gray, num_bins):
  assert len(img_gray.shape) == 2, 'image dimension mismatch'
  assert img_gray.dtype == 'float', 'incorrect image type'

  # Compute histogram
  hist, bin_edges = np.histogram(img_gray, bins=num_bins)
  
  # Normalize hist
  hist_norm = hist / (img_gray.size * np.diff(bin_edges))

  return hist_norm, bin_edges


def rgb_hist(img_color, num_bins):
  assert len(img_color.shape) == 3, 'image dimension mismatch'
  assert img_color.dtype == 'float', 'incorrect image type'

  # define a 3D histogram  with "num_bins^3" number of entries
  hists = np.zeros((num_bins, num_bins, num_bins))

  # execute the loop for each pixel in the image 
  for i in range(img_color.shape[0]):
      for j in range(img_color.shape[1]):
          # increment a histogram bin which corresponds to the value of pixel i,j; h(R,G,B)
          b, g, r = np.clip(img_color[i,j], 0, 255).astype(int)
          bin_r = int(r / (255 / num_bins))
          bin_g = int(g / (255 / num_bins))
          bin_b = int(b / (255 / num_bins))
          hists[bin_b,bin_g,bin_r] += 1
  # normalize the histogram such that its integral (sum) is equal 1
  hists = hists / np.sum(hists)
  hists = hists.reshape(hists.size)
  return hists


def rg_hist(img_color, num_bins):

  """
  compute joint histogram for r/g values
  note that r/g values should be in the range [0, 1];
  histogram should be normalized so that sum of all values equals 1

  img_color - input color image
  num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2
  """

  assert len(img_color.shape) == 3, 'image dimension mismatch'
  assert img_color.dtype == 'float', 'incorrect image type'

  # define a 2D histogram  with "num_bins^2" number of entries
  hists = np.zeros((num_bins, num_bins))
  rg = img_color[:, :, 0] / (img_color[:, :, 0] + img_color[:, :, 1])
  rg_range = np.linspace(0, 1, num_bins + 1)

  for i in range(num_bins):
    for j in range(num_bins):
      r_inds = np.where((rg >= rg_range[i]) & (rg < rg_range[i+1]))
      g_inds = np.where((rg>= rg_range[j]) & (rg < rg_range[j+1]))
      hists[i, j] = len(np.intersect1d(r_inds, g_inds))
  hists = hists / np.sum(hists)
  hists = hists.reshape(hists.size)
  return hists



def dxdy_hist(img_gray, num_bins):

  """
  compute joint histogram of Gaussian partial derivatives of the image in x and y direction
  for sigma = 7.0, the range of derivatives is approximately [-30, 30]
  histogram should be normalized so that sum of all values equals 1
  
  img_gray - input grayvalue image
  num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2
  
  note: you can use the function gaussderiv from the filter exercise.
  """

  assert len(img_gray.shape) == 2, 'image dimension mismatch'
  assert img_gray.dtype == 'float', 'incorrect image type'

  # compute the first derivatives
  Gx, Gy = gaussderiv(img_gray, 7.0)

  # quantize derivatives to "num_bins" number of values
  min_val = np.minimum(np.min(Gx), np.min(Gy))
  max_val = np.maximum(np.max(Gx), np.max(Gy))
  Gx = np.clip(np.round((Gx - min_val) / (max_val - min_val) * (num_bins - 1)), 0, num_bins - 1).astype(np.int32)
  Gy = np.clip(np.round((Gy - min_val) / (max_val - min_val) * (num_bins - 1)), 0, num_bins - 1).astype(np.int32)

  # define a 2D histogram  with "num_bins^2" number of entries
  hists = np.zeros((num_bins, num_bins))
  for i in range(img_gray.shape[0]):
    for j in range(img_gray.shape[1]):
      hists[Gx[i, j], Gy[i, j]] += 1
  
  hists = hists / np.sum(hists)
  hists = hists.reshape(hists.size)
  return hists


def dist_chi2(x,y):
  """ Compute chi2 distance between x and y """
  assert len(x) == len(y)
  return np.sum((x - y)**2 / y)


def dist_l2(x,y):
  """Compute l2 distance between x and y"""
  assert len(x) == len(y)    
  return np.sqrt(np.sum((x - y)**2))


def dist_intersect(x,y):

  """Compute intersection distance between x and y. Return 1 - intersection, so that smaller values also correspond to more similart histograms"""
  assert len(x) == len(y)
  intsec = np.sum(np.minimum(x, y))
  return (1 - (intsec / np.sum(np.maximum(x, y))))


def get_dist_by_name(x, y, dist_name):
  
  if dist_name == 'chi2':
    return dist_chi2(x,y)
  elif dist_name == 'intersect':
    return dist_intersect(x,y)
  elif dist_name == 'l2':
    return dist_l2(x,y)
  else:
    assert 'unknown distance: %s'%dist_name
    

def is_grayvalue_hist(hist_name):
  
  if hist_name == 'grayvalue' or hist_name == 'dxdy':
    return True
  elif hist_name == 'rgb' or hist_name == 'rg':
    return False
  else:
    assert False, 'unknown histogram type'
    

def get_hist_by_name(img1_gray, num_bins_gray, dist_name):
  
  if dist_name == 'grayvalue':
    return normalized_histogram(img1_gray, num_bins_gray)
  elif dist_name == 'rgb':
    return rgb_hist(img1_gray, num_bins_gray)
  elif dist_name == 'rg':
    return rg_hist(img1_gray, num_bins_gray)
  elif dist_name == 'dxdy':
    return dxdy_hist(img1_gray, num_bins_gray)
  else:
    assert 'unknown distance: %s'%dist_name