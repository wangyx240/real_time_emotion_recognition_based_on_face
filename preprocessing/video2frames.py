import os
import cv2
import matplotlib.pyplot as plt
import pickle
import numpy as np

def convert2frames(videos_folder, frames_folder):
    ''' Converts videos into frames

    Args:
        videos_folder (string): folder with input videos
        frames_folder (string): folder with output frames
    '''
    
    # working directory '../Emotion_classification_Ravdess'
    cwd = os.getcwd()

    videos_fn = os.path.join(cwd, videos_folder)
    frames_fn = os.path.join(cwd, frames_folder)
    
    # make a list of all videos in 'videos_folder' directory
    videos_list = os.listdir(videos_fn)
    
    # convert all videos
    for i in range(0, len(videos_list)):
        print("Progress: {}/{}".format(i+1, len(videos_list)))
        utterance_fn = os.path.join(*(videos_fn, videos_list[i]))        
        emotion = int(videos_list[i].split("-")[2]) - 1

        # Create directory for frames
        res_fn = os.path.join(*(frames_fn, str(i)))
        if not os.path.exists(res_fn):
            os.makedirs(res_fn)

        vidcap = cv2.VideoCapture(utterance_fn)
        success,image = vidcap.read()
        count = 0
        success = True
        while success:
            idx = 'frame_' + str(count) + '.png'
            frame_fn = os.path.join(*(res_fn, idx))

            cv2.imwrite(frame_fn, image)   
            success,image = vidcap.read()
            count += 1

        annotations = {"emotion": emotion}
        with open(res_fn + "\\annotation.pickle", 'wb') as f:
            pickle.dump(annotations, f, protocol = 2)

def convert2pickle(frames_folder, pickle_folder):

    # frame index processing will start from
    start_index = 0 

    # skip <skip> frames after reading 1 frame (to simulate lower fps) (higher = more data)
    skip = 0

    # max instances count for each emotion in the result dataset (lower = possibly less data)
    max_count = inf 

    # frames count in one result images array
    frames_count = 30 

    # max frames count to get from each folder with frames (lower = less data)
    max_from_folder = inf 


    conf_threshold = 0.8 # threshold for face detector
    emotions_count = 8 # number of emotion classes
    emotions = np.zeros(emotions_count)

    # resolution of result frames that will be saved to the .pickle file
    frames_resolution = [64, 64, 1]

    # starting frame index to process in each folder with frames
    image_index = 0

    # max array count from one folder with frames (each array has 'frames_count' images)
    max_array_count = 1000

    # saves result data every 'save_index' time
    save_index = 500

    # result data
    data = []
    # with open("dataset//resolution_{}x{}x{}_index_{}_frames_count_{}_skip_{}_train.pickle".format(frames_resolution[0], frames_resolution[1], frames_resolution[2], i, frames_count, skip), 'rb') as handle:
    #     data = pickle.load(handle)



if __name__ == "__main__":
    convert2frames('dataset\\TestVideos\\', 'dataset\\TestFrames\\')
    convert2frames('dataset\\TrainingVideos\\', 'dataset\\TrainingFrames\\')


