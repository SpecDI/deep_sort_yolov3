import os
import cv2
import argparse

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
    return parser.parse_args()

def process_crops(crops_file):
    crops = dict() #crops[frame_number] = (track_id, xMin, yMin, xMax, yMax, action)
    with open(crops_file) as fp:
        line = fp.readline()
        while line:
            data = line.split(' ')
            if not int(data[5]) in crops.keys():
                crops[int(data[5])] = []
            action = "unknown"
            if len(data) > 10:
                action = data[10].strip().replace('"', '').replace('/', 'or')
            #crops[frame_number] = (track_id, xMin, yMin, xMax, yMax, action)
            crops[int(data[5])].append((int(data[0]), int(data[1]), int(data[2]), int(data[3]), int(data[4]), action))
            line = fp.readline()
    return crops

def main(sequence_file, crops_file):
    crops = process_crops(crops_file) #crops[frame_number] = (track_id, xMin, yMin, xMax, yMax, action)

    if not os.path.exists('crops'):
        os.mkdir('crops')

    path = 'crops/' + sequence_file
    if not os.path.exists(path):
        os.mkdir(path)

    video_capture = cv2.VideoCapture(sequence_file)

    frame_number = 0
    while video_capture.isOpened():
        frame_number+=1
        ret, frame = video_capture.read()

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

        

if __name__ == '__main__':
    # Parse user provided arguments
    print("Processing crop file...")
    args = parse_args()
    print("Creating actiontubes...")
    main(args.sequence_file, args.crop_file)
