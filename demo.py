from __future__ import division, print_function, absolute_import

import os
import glob
import json

from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')

import argparse

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_file", help="Path to input sequence",
        required = True)
    parser.add_argument(
        "--fps", help="Frames per second.",
        type = int,
        default = 11)
    parser.add_argument(
        "--enable_cropping", help="Flag used to enable cropping of frames."
        "When active the tracker no longer draws the bounding boxes",
        action = 'store_true',
        default = False
    )
    parser.add_argument(
        "--labels_file", help="Path to file containing action labels",
        default = None)
    return parser.parse_args()

def parse_labels_file(labels_file):
  action_map = dict()
  with open(labels_file, 'r') as lb_handler:
    line = lb_handler.readline()
    
    while line:
      track_id = int(line.split(',')[0])
      action_id = line.split(',')[1]

      action_map[track_id] = action_id
      line = lb_handler.readline()

  return action_map

def main(yolo, sequence_file, fps_render_rate, enable_cropping, labels_file):
    # Compute output file
    file_name = os.path.splitext(os.path.basename(sequence_file))[0] if sequence_file != '0' else '0'
    if sequence_file == '0':
        sequence_file = 0

    # Compute the action map if labels provided
    action_map = dict()
    if labels_file != None:
        action_map = parse_labels_file(labels_file)
    print(action_map)

    # Create results/file_name dir
    if not os.path.exists('results/' + file_name):
        os.mkdir('results/' + file_name)

    # Create coords dir for movie
    coords_path = 'coords/' + file_name + '.json'

    output_seq = './output/' + file_name + '.avi'

    # Dict of coordinates for each tracked individual
    track_map = dict()

   # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    
   # deep_sort 
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True 

    video_capture = cv2.VideoCapture(sequence_file)

    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'XVID') #*'MJPG'
        # Build video output handler only if we are not cropping
        out = cv2.VideoWriter(output_seq, fourcc, fps_render_rate, (w, h)) if not enable_cropping else None
        list_file = open('detection.txt', 'w')
        frame_index = -1 
        
    fps = 0.0
    frame_number = 0
    while video_capture.isOpened():
        frame_number+=1
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        t1 = time.time()

       # image = Image.fromarray(frame)
        image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        boxs = yolo.detect_image(image)
       # print("box_num",len(boxs))
        features = encoder(frame,boxs)
        
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            crop_img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])].copy()

            # Append coordinates for individual to track map
            if track.track_id not in track_map:
                track_map[track.track_id] = [(frame_number, [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])]
            else:
                track_map[track.track_id].append((frame_number, [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]))

            # Build directory path
            frames_dir_path = "results/" + file_name + '/' + str(track.track_id)
            if not os.path.exists(frames_dir_path):
                os.mkdir(frames_dir_path)
            # Write frame or annotate frame
            if enable_cropping:
                cv2.imwrite(frames_dir_path + "/" + str(frame_number) + ".jpg", crop_img)
            else:
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                
                append_str = str(track.track_id) + ": Person"
                if track.track_id in action_map:
                    append_str += ' ' + action_map[track.track_id]
                cv2.putText(frame, append_str,(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

        with open(coords_path, 'w+') as fp:
            json.dump(track_map, fp)

        for det in detections:
            bbox = det.to_tlbr()
            if not enable_cropping:
                cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
            
        cv2.imshow('', cv2.resize(frame, (1200, 675)))
        
        if writeVideo_flag:
            # save a frame
            if not enable_cropping:
                out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')
            
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %f"%(fps))
        
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Parse user provided arguments
    args = parse_args()
    main(YOLO(), args.sequence_file, args.fps, args.enable_cropping, args.labels_file)
