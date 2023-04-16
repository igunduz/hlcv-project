import numpy as np

from .image_filtering import gaussderiv


def normalized_histogram(img_gray, num_bins):
  assert len(img_gray.shape) == 2, 'image dimension mismatch'
  assert img_gray.dtype == 'float', 'incorrect image type'

  ### Your code here

  raise NotImplementedError

  return hists, bins


def rgb_hist(img_color, num_bins):
  assert len(img_color.shape) == 3, 'image dimension mismatch'
  assert img_color.dtype == 'float', 'incorrect image type'

  # define a 3D histogram  with "num_bins^3" number of entries
  hists = np.zeros((num_bins, num_bins, num_bins))

  # execute the loop for each pixel in the image 
  for i in range(img_color.shape[0]):
      for j in range(img_color.shape[1]):
          # increment a histogram bin which corresponds to the value of pixel i,j; h(R,G,B)
          ### Your code here
          pass

  # normalize the histogram such that its integral (sum) is equal 1
  ### Your code here
  raise NotImplementedError

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
  
  # Your code here
  
  raise NotImplementedError

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
  

  # quantize derivatives to "num_bins" number of values
  # define a 2D histogram  with "num_bins^2" number of entries
  hists = np.zeros((num_bins, num_bins))

  raise NotImplementedError
  
  hists = hists.reshape(hists.size)
  return hists


def dist_chi2(x,y):
  """ Compute chi2 distance between x and y """
  
  # your code here    
  raise NotImplementedError


def dist_l2(x,y):
  """Compute l2 distance between x and y"""
      
  # your code here    
  raise NotImplementedError


def dist_intersect(x,y):

  """Compute intersection distance between x and y. Return 1 - intersection, so that smaller values also correspond to more similart histograms"""
  
  # your code here    
  raise NotImplementedError


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