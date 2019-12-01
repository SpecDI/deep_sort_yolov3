from __future__ import division, print_function, absolute_import

import os
import glob
import json
import time

import cv2
from PIL import Image

import argparse

def parse_args():
  """ Parse command line arguments.
  """
  parser = argparse.ArgumentParser(description="Annotate videos using okutama labels")
  parser.add_argument(
    "--sequence_file", help = "Path to input sequence",
    required = True)
  parser.add_argument(
    "--okutama_labels", help = "Path to file containing action labels",
    required = True)
  return parser.parse_args()

def load_labels(okutama_labels):
  frame_map = dict()
  with open(okutama_labels, 'r') as fp:
    line = fp.readline()

    while line:
      line_split = line.split(' ')
      #print(line_split)
      frame_id = int(line_split[5])
      val = (int(line_split[0]), list(map(int, line_split[1:5])), list(map(int, line_split[6:8])))

      if frame_id not in frame_map:
        frame_map[frame_id] = [val]
      else:
        frame_map[frame_id].append(val)

      line = fp.readline()

  return frame_map

def main(sequence_file, okutama_labels):
  frame_map = load_labels(okutama_labels)
  
  sequence_name = os.path.splitext(os.path.basename(sequence_file))[0]
  output_seq = './okutama_output/' + sequence_name + '.avi'

  writeVideo_flag = True
  video_capture = cv2.VideoCapture(sequence_file)
  if writeVideo_flag:
    # Define the codec and create VideoWriter object
    w = int(video_capture.get(3))
    h = int(video_capture.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # Build video output handler only if we are not cropping
    out = cv2.VideoWriter(output_seq, fourcc, 11, (w, h))

  frame_number = 0
  while video_capture.isOpened():
    ret, frame = video_capture.read()  # frame shape 640*480*3
    if ret != True:
      break

    tracks = frame_map[frame_number]
    for track in tracks:
      if 1 in track[2]:
        continue

      bbox = track[1]
      cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
      cv2.putText(frame, str(track[0]),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)
    
    cv2.imshow('', cv2.resize(frame, (1200, 675)))

    if writeVideo_flag:
      out.write(frame)

    frame_number += 1

    # Press Q to stop!
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  video_capture.release()
  if writeVideo_flag:
    out.release()
  cv2.destroyAllWindows()


if __name__ == '__main__':
  # Parse user provided arguments
  args = parse_args()
  main(args.sequence_file, args.okutama_labels)