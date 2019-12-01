import os
import cv2
import argparse
from timeit import time

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Crop creator")
    parser.add_argument(
        "--sequence_file", help="Path to input sequence",
        default = 0)
    parser.add_argument(
        "--crop_file", help="Path to bounding box text file",
        default = 0)
    parser.add_argument(
        "--mode", help="1 = output video overlay | 0 = produce image crops"
    )
    return parser.parse_args()

def process_crops(crops_file):
    crops = dict() #crops[frame_number] = (track_id, xMin, yMin, xMax, yMax, action)
    with open(crops_file) as fp:
        line = fp.readline()
        while line:
            data = line.split(' ')
            if int(data[6]) != 1 and int(data[7]) != 1:
                if not int(data[5]) in crops.keys():
                    crops[int(data[5])] = []
                action = "unknown"
                if len(data) > 10:
                    action = data[10].strip().replace('"', '').replace('/', 'or')
                #crops[frame_number] = (track_id, xMin, yMin, xMax, yMax, action)
                crops[int(data[5])].append((int(data[0]), int(data[1]), int(data[2]), int(data[3]), int(data[4]), action))
            line = fp.readline()
    return crops

def create_crops(sequence_file, crops):
    if not os.path.exists('crops'):
        os.mkdir('crops')

    path = 'crops/' + sequence_file
    if not os.path.exists(path):
        os.mkdir(path)
    video_capture = cv2.VideoCapture(sequence_file)
    frame_number = -1
    while video_capture.isOpened():
        frame_number+=1
        ret, frame = video_capture.read()

        if ret != True:
            break
        if not frame_number in crops:
            continue


        for crop in crops[frame_number]:
            crop_img = frame[crop[2]:crop[4], crop[1]:crop[3]]
            if not os.path.exists(path + "/" + crop[5]):
                os.mkdir(path + "/" + crop[5])
            if not os.path.exists(path + "/" + crop[5] +"/" + str(crop[0])):
                os.mkdir(path + "/" +crop[5] +"/"+ str(crop[0]))
            cv2.imwrite(path + "/" + crop[5] +"/" + str(crop[0]) + "/" + str(frame_number) + ".jpg", crop_img)
        
    video_capture.release()
    cv2.destroyAllWindows()

def overlay_video(sequence_file, crops):
    if not os.path.exists('okutama_output'):
        os.mkdir('okutama_output')

    sequence_name = os.path.splitext(os.path.basename(sequence_file))[0]
    output_seq = './okutama_output/' + sequence_name + '.avi'

    writeVideo_flag = True
    video_capture = cv2.VideoCapture(sequence_file)

    fps = 0.0

    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # Build video output handler only if we are not cropping
        out = cv2.VideoWriter(output_seq, fourcc, 10, (w, h))
        list_file = open('detection.txt', 'w')

    frame_number = -1

    while video_capture.isOpened():
        frame_number+=1
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        if not frame_number in crops:
            if writeVideo_flag:
                out.write(frame)
            continue

        t1 = time.time()

        for crop in crops[frame_number]:
            cv2.rectangle(frame, (int(crop[1]), int(crop[2])), (int(crop[3]), int(crop[4])),(255,255,255), 2)
            cv2.putText(frame, str(crop[5]),(int(crop[1]), int(crop[2])),0, 5e-3 * 200, (0,255,0),2)
        if writeVideo_flag:
            out.write(frame)

        # Press Q to stop!
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %f"%(fps))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()


def main(sequence_file, crops_file, mode):
    crops = process_crops(crops_file) #crops[frame_number] = (track_id, xMin, yMin, xMax, yMax, action)

    if int(mode) == 0:
        create_crops(sequence_file, crops)
    elif int(mode) == 1:
        overlay_video(sequence_file, crops)
        

if __name__ == '__main__':
    # Parse user provided arguments
    print("Processing crop file...")
    args = parse_args()
    main(args.sequence_file, args.crop_file, args.mode)
