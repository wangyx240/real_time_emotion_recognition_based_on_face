#     N      EN    
#     0      neutral     
#     1      calm        
#     2      happy       
#     3      sad         
#     4      angry       
#     5      fearful     
#     6      disgust    
#     7      surprised  

import cv2
import time
import argparse
import os
import numpy as np
import pickle
from imutils.face_utils import FaceAligner
import dlib
import sys

project_directory = os.getcwd()
print("Project directory: ", project_directory)

# ___ IMAGE FILTER DATA ___

# folder index processing will start from (lower = more instances)
folder_start_index = 0

# skip <skip> frames after reading 1 frame (to simulate lower fps) (lower = more instances)
skip = 0

# instances max count for each emotion in the result dataset (lower = less instances)
max_count = 1000

# frames count in one instance in the result dataset (lower = more instances but they are smaller)
frames_count = 30

# number of emotion classes
emotions_count = 8
emotions = np.zeros(emotions_count)

# frames max count to get from each folder with frames (lower = less instances)
max_from_folder = 1000

# resolution of the result frames that will be in the result dataset
frames_resolution = [64, 64, 1]

# frame index in each folder processing will start from (lower = more instances)
image_start_index = 0

# instances max count from one folder with frames (each instance has <frames_count> images) (lower = less instances)
max_array_count = 1000

# saves result pickle data every <save_index> processed folder
save_index = 500

# ___ FACE DETECTOR ___

# threshold for face detector
conf_threshold = 0.8

modelFile = os.path.join(project_directory, "models\\opencv_face_detector_uint8.pb")
configFile = os.path.join(project_directory, "models\\opencv_face_detector.pbtxt")
face_detector = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

# ___ FACE ALIGNER ___ (uses emotion recognition model input shape)
predictor = dlib.shape_predictor(os.path.join(project_directory, "models\\shape_predictor_68_face_landmarks.dat"))
fa = FaceAligner(predictor, desiredFaceWidth=frames_resolution[0], desiredFaceHeight=frames_resolution[1])

# ___ FOLDERS ___
training_frames_folder = os.path.join(project_directory, "dataset\\TrainingFrames")
test_frames_folder = os.path.join(project_directory, "dataset\\TestFrames")

training_save_folder = os.path.join(project_directory, "dataset\\resolution_{}x{}x{}_train.pickle".format(frames_resolution[0], frames_resolution[1], frames_resolution[2]))
test_save_folder = os.path.join(project_directory, "dataset\\resolution_{}x{}x{}_test.pickle".format(frames_resolution[0], frames_resolution[1], frames_resolution[2]))

def toOneHot(number, emotions_count):
    arr = np.zeros((1, emotions_count), dtype=int)
    arr[0][number] = 1
    return arr[0]

def get_data(frames_folder, save_folder):
    data = []

    frames_folders = sorted(os.listdir(frames_folder), key=int)
    
    for i, folder_name in enumerate(frames_folders):
        if i < folder_start_index:
            continue

        if(i%save_index==0 and i!=0):
            with open(os.path.join(project_directory, save_folder), 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        folder_path = os.path.join(*(frames_folder, folder_name))  

        # load emotion number for the annotation file in the folder
        with open(os.path.join(*(folder_path, "annotation.pickle")), 'rb') as handle:
            annotation = pickle.load(handle)

        # if arrays count in the result data for this emotion is less than max_count
        emotion = annotation["emotion"]

        # if directory exist and contain enough frames for 1 array 
        if (os.path.exists(folder_path)) and (len(os.listdir(folder_path)) - 1 >= frames_count):

            if(emotions[emotion] < max_count):

                # loop through all frames
                length = len(os.listdir(folder_path)) - 1
                array_count = 0
                index = image_start_index
                images = []
                while (index < length) and (max_from_folder > array_count) :
                    image_path = os.path.join(folder_path, "frame_{}.png".format(index))
                    image = cv2.imread(image_path, 1)

                    # blob for face detector
                    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
                    face_detector.setInput(blob)
                    faces = face_detector.forward()

                    # loop through all found faces
                    for f in range(faces.shape[2]):
                        confidence = faces[0, 0, f, 2]
                        if confidence > conf_threshold:
                            x1 = int(faces[0, 0, f, 3] * image.shape[1])
                            y1 = int(faces[0, 0, f, 4] * image.shape[0])
                            x2 = int(faces[0, 0, f, 5] * image.shape[1])
                            y2 = int(faces[0, 0, f, 6] * image.shape[0])

                            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                            detected_face = fa.align(image, gray, dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2))
                            if detected_face.size != 0:

                                # resize, normalize and save the frame (convert to grayscale if frames_resolution[2] == 1)
                                if(frames_resolution[2] == 1):
                                    detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)

                                # detected_face = cv2.resize(detected_face, (frames_resolution[0], frames_resolution[1]))
                                detected_face = cv2.normalize(detected_face, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                                images.append(detected_face)

                                # skip 'skip' next frames
                                index += skip

                    index += 1

                    # if True, saves array of 'frames_count' images
                    if len(images) == frames_count:
                        item = {"emotion": np.asarray(toOneHot(emotion, emotions_count)), "images": np.asarray(images)}
                        images = []
                        array_count += 1
                        emotions[emotion] += 1
                        data.append(item)

                    if (array_count >= max_array_count):
                        break
                        
                print("Processed: {}/{} \tAdded: \tEmotion: {} \t{} arrays of {} images".format(i, len(frames_folders), emotion, array_count, frames_count))
        
            else:
                print("Processed: {}/{} \tEmotion: {} \tAlready enough data of this emotion".format(i, len(frames_folders), emotion)) 

        else:
            print("Directory doesn't exist or contain not enough frames") 

    return data

# Visualize emotion distribution
def data_distibution(data):
    emotions_data = np.array([x["emotion"] for x in data])
    emotions = np.zeros(emotions_count)

    for i, x in enumerate(emotions_data):
        index = np.where(x == 1)[0][0]
        emotions[index] += 1

    return emotions

# Append flipped images for emotion number 'emotion_index'
def append_flipped(data, emotion_index):
    for item in list(data):
        em_index = np.where(item["emotion"] == 1)[0][0]

        if em_index == emotion_index:
            arr = []     
            for image in item["images"]:
                arr.append(cv2.flip(image, 0))

            arr = np.asarray(arr)
            new_item = {"emotion": item["emotion"], "images": arr}
            data.append(new_item)

def prepare_data_4_save(data, max_instances_count):  
    shuffled_data = np.asarray(data)
    np.random.shuffle(shuffled_data)
    
    result = []
    emotions = np.zeros(emotions_count)
    emotions_data = np.array([x["emotion"] for x in shuffled_data])

    for j in range(0, len(shuffled_data)):  
        index = np.where(emotions_data[j] == 1)[0][0]

        if emotions[index] < max_instances_count:
            emotion = np.zeros(emotions_count)
            emotion[index] = 1
            sample = shuffled_data[j]
            sample["emotion"] = emotion
            result.append(sample)
            emotions[index] += 1

    print("data distribution: ", emotions)
    return result

def save_data(data, file_name):
    file_name = os.path.join(project_directory,"dataset//{}".format(file_name))
    with open(file_name, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


test_data = get_data(test_frames_folder, test_save_folder)
train_data = get_data(training_frames_folder, training_save_folder)

print("Data distribution:")
print("train data:")
print(data_distibution(train_data))

print("test data:")
print(data_distibution(test_data))

print("Flipping frames to get more data for emotion number '0':")
append_flipped(train_data, 0)
append_flipped(test_data, 0)

print("Data distribution after flipping:")
print("train data:")
print(data_distibution(train_data))

print("test data:")
print(data_distibution(test_data))

test_data_count = 24
train_data_count = 500

# form
test_data_4_save = prepare_data_4_save(test_data, test_data_count)
train_data_4_save = prepare_data_4_save(train_data, train_data_count)

# save
save_data(test_data_4_save, "resolution_{}x{}x{}_count_{}_test_data.pickle".format(frames_resolution[0], frames_resolution[1], frames_resolution[2], test_data_count))
save_data(train_data_4_save, "resolution_{}x{}x{}_count_{}_train_data.pickle".format(frames_resolution[0], frames_resolution[1], frames_resolution[2], train_data_count))
