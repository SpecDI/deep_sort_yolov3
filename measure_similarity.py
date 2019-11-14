from __future__ import division, print_function, absolute_import

import os
import glob
import argparse
import numpy as np
from PIL import Image
import glob
from matplotlib import pyplot as plt
from SSIM_PIL import compare_ssim
import cv2

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

def load_frames(dir_path):
  frame_list = []
  for file_path in glob.glob(dir_path + '*.jpg'):
    frame_list.append(np.asarray(Image.open(file_path)))

  return frame_list

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
  
  for tLabel in target_labels:
    print(tLabel)
    sFrame_idx = (source_labels == tLabel)
    tFrame_idx = (target_frames == tLabel)

    sFrame = np.asarray(source_frames[sFrame_idx])
    tFrame = np.asarray(target_frames[tFrame_idx])
    
    if(sFrame.shape[0] < tFrame.shape[0]):
      sFrame = cv2.copyMakeBorder(sFrame, tFrame.shape[0] - sFrame.shape[0], 0, 0, 0, cv2.BORDER_CONSTANT)
    else:
      tFrame = cv2.copyMakeBorder(tFrame, sFrame.shape[0] - tFrame.shape[0], 0, 0, 0, cv2.BORDER_CONSTANT)

    if(sFrame.shape[1] < tFrame.shape[1]):
      sFrame = cv2.copyMakeBorder(sFrame, tFrame.shape[1] - sFrame.shape[1], 0, 0, 0, cv2.BORDER_CONSTANT)
    else:
      tFrame = cv2.copyMakeBorder(tFrame, 0, 0, sFrame.shape[1] - tFrame.shape[1], 0, cv2.BORDER_CONSTANT)

    print(compare_ssim(Image.fromarray(sFrame, 'RGB'), Image.fromarray(tFrame, 'RGB')))


  
  

if __name__ == '__main__':
  # Parse user provided arguments
  args = parse_args()
  main(args.dir_path_source, args.dir_path_target)
