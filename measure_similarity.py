from __future__ import division, print_function, absolute_import

import os
import glob
import argparse

from PIL import Image
from matplotlib import pyplot as plt

import numpy as np
from scipy.stats import ks_2samp
from scipy.stats import norm

def parse_args():
  parser = argparse.ArgumentParser(description="Similarity measure")
  parser.add_argument(
    '--dir_path_source', help = 'Path to directory of source frames',
    required = True
  )
  parser.add_argument(
    '--dir_path_target', help = 'Path to directory of target frames',
    required = True
  )
  parser.add_argument(
    '--display_imhist', help = 'Flag used to enable visual comparisons between frames',
    action = 'store_true'
  )
  parser.set_defaults(display_imhist = False)
  return parser.parse_args()

def display_comparison(im1, im2):
  ''' Utility function used to compare two images and the underlying pdfs'''

  # Display images on one row
  plt.subplot(2, 2, 1)
  plt.imshow(im1)
  plt.axis('off')

  plt.subplot(2, 2, 2)
  plt.imshow(im2)
  plt.axis('off')

  # Display pdfs on second row
  plt.subplot(2, 2, 3)
  plt.hist(im1.ravel(), bins = 256)

  plt.subplot(2, 2, 4)
  plt.hist(im2.ravel(), bins = 256)

  plt.show()

def main(source_path, target_path, display_imhist):
  
  # Extract all labes from one of our images dir
  source_labels = sorted([int(os.path.splitext(os.path.basename(file_path))[0]) for file_path in glob.glob(source_path + '*.jpg')])
  source_frames = []
  
  # Add all frames with label to list
  for file_id in source_labels:
    source_frames.append(Image.open(source_path + str(file_id) + '.jpg'))

  # Extract all labels from phd guy's action tubes
  target_labels = os.listdir(target_path)
  # Sort them by frame number
  target_labels = sorted(target_labels, key = lambda f: int(os.path.splitext(os.path.basename(f))[0].split('.')[3].split('_')[1]))
  
  # Load corresponding frames
  target_frames = []
  for file_id in target_labels:
    target_frames.append(Image.open(target_path + str(file_id)))

  # Remove video name and extension from labels of action tubes
  target_labels = list(map(lambda f: int(os.path.splitext(os.path.basename(f))[0].split('.')[3].split('_')[1]), target_labels))
  
  # Computer average score
  statistic = 0
  pvalue = 0
  count = 0
  for tLabel in target_labels:
    # Get image indices
    if tLabel not in source_labels:
      continue
    
    sFrame_idx = source_labels.index(tLabel)
    tFrame_idx = target_labels.index(tLabel)
    print(sFrame_idx, tFrame_idx)

    # Extract image and coonvert
    sFrame = np.asarray(source_frames[sFrame_idx])
    tFrame = np.asarray(target_frames[tFrame_idx])

    # Compute the image histograms
    sHist, _ = np.histogram(sFrame.ravel(), 256, [0,256])
    tHist, _ = np.histogram(tFrame.ravel(), 256, [0,256])

    sCdf = norm.cdf(sHist)
    tCdf = norm.cdf(tHist)

    # Perform the Two-sample Kolmogorov–Smirnov test
    ks_test = ks_2samp(sCdf, tCdf)
    statistic += ks_test.statistic
    pvalue += ks_test.pvalue

    count += 1

    # Display the images and the pdfs if option enabled
    if display_imhist:
      display_comparison(sFrame, tFrame)

  print('statistic: ', statistic / count)
  print('pvalue: ', pvalue / count)

  

if __name__ == '__main__':
  # Parse user provided arguments
  args = parse_args()
  main(args.dir_path_source, args.dir_path_target, args.display_imhist)
