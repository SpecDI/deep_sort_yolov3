from __future__ import division, print_function, absolute_import

import os
import glob
import argparse
import numpy as np
from PIL import Image
import glob
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
import cv2
from skimage.color import rgb2gray

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
  return parser.parse_args()

def resize_images(im1, im2):
  if(im1.shape[0] > im2.shape[0]):
    im1 = im1[:-(im1.shape[0] - im2.shape[0]), :]
  else:
    im2 = im2[:-(im2.shape[0] - im1.shape[0]), :]

  if(im1.shape[1] > im2.shape[1]):
    im1 = im1[:, :-(im1.shape[1] - im2.shape[1])]
  else:
    im2 = im2[:, :-(im2.shape[1] - im1.shape[1])]

  return im1, im2

def display_img_comparrison(im1, im2):
  plt.subplot(1, 2, 1)
  plt.imshow(im1, 'gray')
  plt.axis('off')

  plt.subplot(1, 2, 2)
  plt.imshow(im2, 'gray')
  plt.axis('off')

  plt.show()

def main(source_path, target_path):
  
  source_labels = sorted([int(os.path.splitext(os.path.basename(file_path))[0]) for file_path in glob.glob(source_path + '*.jpg')])
  source_frames = []
  
  for file_id in source_labels:
    # Add the frame to our frame array
    source_frames.append(Image.open(source_path + str(file_id) + '.jpg'))

  target_labels = os.listdir(target_path)
  target_labels = sorted(target_labels, key = lambda f: int(os.path.splitext(os.path.basename(f))[0].split('.')[3].split('_')[1]))
  
  target_frames = []
  for file_id in target_labels:
    target_frames.append(Image.open(target_path + str(file_id)))

  target_labels = list(map(lambda f: int(os.path.splitext(os.path.basename(f))[0].split('.')[3].split('_')[1]), target_labels))
  
  # Computer average score
  score = 0
  for tLabel in target_labels:
    sFrame_idx = (source_labels == tLabel)
    tFrame_idx = (target_frames == tLabel)

    sFrame = rgb2gray(np.asarray(source_frames[sFrame_idx]))
    tFrame = rgb2gray(np.asarray(target_frames[tFrame_idx]))

    sFrame, tFrame = resize_images(sFrame, tFrame)
    score += ssim(sFrame, tFrame)

  print(score / len(target_labels))
  
  

if __name__ == '__main__':
  # Parse user provided arguments
  args = parse_args()
  main(args.dir_path_source, args.dir_path_target)
