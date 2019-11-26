from __future__ import division, print_function, absolute_import

import os
import glob
import json

import argparse

def parse_args():
  """ Parse command line arguments.
  """
  parser = argparse.ArgumentParser(description="mAP Estimate")
  parser.add_argument(
    "--gen_coords", help = "Path to input sequence",
    required = True)
  parser.add_argument(
    "--ref_coords", help = "Path to file containing action labels",
    required = True)
  parser.add_argument(
    "--map_file", help = "Path to map file between gen coords to ref coords",
    default = "./okutama_labels/okutama_map_labels.json"
  )
  return parser.parse_args()

def load_labelMapping(map_file):
  with open(map_file, 'r') as handle:
    dictdump = json.loads(handle.read())
  
  return {int(k):int(v) for k,v in dictdump.items()}

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def mean_averagePrecision(gen_track_id, ref_track_id, track_map, ref_map):
  average_precision = 0.0
  count = 0
  for frame_number in track_map[gen_track_id]['frame']:
    if frame_number in list(map(int, (track_map[gen_track_id]['frame']))):
      idx = track_map[gen_track_id]['frame'].index(frame_number)
    else:
      continue

    current_gen = track_map[gen_track_id]['coords'][idx]
    
    if frame_number in ref_map[ref_track_id]['frame']:
      idx = ref_map[ref_track_id]['frame'].index(frame_number)
    else:
      continue

    current_ref = ref_map[ref_track_id]['coords'][idx]
    
    iou = bb_intersection_over_union(current_gen, current_ref)
    #print('Frame ' + str(frame_number) +' IoU: ', iou)
    average_precision += iou
    count += 1

  return average_precision / count

def main(gen_coords, ref_coords, map_file):
  # Load dict of gen coords
  with open(gen_coords, 'r') as fp:
    track_map = json.load(fp)
  track_map = {int(k):v for k,v in track_map.items()}
  
  # Init reference coordinates map
  ref_map = dict()

  # Load mapping between ids
  # gen id -> ref id
  id_map = load_labelMapping(map_file)

  # Open ref file
  with open(ref_coords, 'r') as fp:
    line = fp.readline()
    while line:
      lineSplit = line.split(' ')
      lineSplit = list(map(int, lineSplit[:6]))
      currentId = lineSplit[0]
      corners = lineSplit[1:5]
      frame_number = lineSplit[5]

      if currentId not in ref_map:
        ref_map[currentId] = dict()
        ref_map[currentId]['coords'] = [corners]
        ref_map[currentId]['frame'] = [frame_number]
      else:
        ref_map[currentId]['coords'].append(corners)
        ref_map[currentId]['frame'].append(frame_number)

      line = fp.readline()

  for gen_track_id in id_map:
    mAP_score = mean_averagePrecision(gen_track_id, id_map[gen_track_id], track_map, ref_map)
    print(f'Id {gen_track_id} score: {mAP_score}')

if __name__ == '__main__':
    # Parse user provided arguments
    args = parse_args()
    main(args.gen_coords, args.ref_coords, args.map_file)