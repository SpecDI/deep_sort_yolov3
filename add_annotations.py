from __future__ import division, print_function, absolute_import

import os
import glob

import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

import argparse

def parse_args():
  """ Parse command line arguments."""

  parser = argparse.ArgumentParser(description="Add annotations to tracked videos")
  parser.add_argument(
      "--sequence_file", help="Path to input sequence",
      required = True)
  parser.add_argument(
      "--labels_file", help="Path to file containing action labels",
      required = True)

  return parser.parse_args()

def main(sequence_file, labels_file):
  action_map = dict()
  with open(labels_file, 'r') as lb_handler:
    line = lb_handler.readline()
    while line:
      # Ignore first line as it's header
      line = lb_handler.readline()

      try:
        track_id = line.split(' ')[0].split('/')[1]
      except:
        continue
      
      action_id = line.split(' ')[1]
      action_map[track_id] = action_id

    print(action_map)

if __name__ == '__main__':
  # Parse user provided arguments
  args = parse_args()
  main(args.sequence_file, args.labels_file)