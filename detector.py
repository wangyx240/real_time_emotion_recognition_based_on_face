import cv2
import argparse
import os
import time
import json
import sys
import dlib
import pandas as pd
import numpy as np
import imutils
from imutils.face_utils import FaceAligner
import reconmodels
from moviepy.editor import VideoFileClip
import numpy as np
import torch
import re
import torch.nn.functional as F
import recon_losses_R
import recon_metrics_R
import yaml
import time
from tensorboardX import SummaryWriter
import warnings
from PIL import Image



#     N   EN
#     0 = neutral
#     1 = calm
#     2 = happy
#     3 = sad
#     4 = angry
#     5 = fearful
#     6 = disgust
#     7 = surprised

emotions_en = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

parser = argparse.ArgumentParser(description='Emotion recognition')
parser.add_argument('--camera_number', type=int, default=0)
parser.add_argument('--input', type=str, default='camera', help='can be "image", "camera" or "video"')
parser.add_argument('--source', type=str, default="images/i.jpg")
parser.add_argument('--output_dir', type=str, default="output", help='directory to save result image or video')
#parser.add_argument('--model', type=str, default="models/base_emotion_classification_model",
#                    help='emotion detector model')
parser.add_argument('--conf_threshold', type=float, default=0.8, help='face detector threshold')
parser.add_argument('--fps', type=int, default=10, help='output video frame rate (for video input only)')
parser.add_argument('--show', dest='show', action='store_false')
parser.set_defaults(show=True)
args = parser.parse_args()


def get_faces(image, face_detector):
    ''' Get faces information from image

    Args:
        image (array): Image to process
        face_detector: loaded model for face detection

    Returns:
        bool: True if at least 1 face was found
        array of vectors: one vector for each face (array of [0,0,confidence,x1,y1,x2,y2])
    '''

    success = True
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
    face_detector.setInput(blob)
    faces = face_detector.forward()
    #print(faces.shape)

    if (faces.shape[2] == 0):
        print("No faces were found")
        success = False
    return success, faces


def detect_emotion_image(face, emotion_detector):
    ''' Get emotion prediction from image

    Args:
        face (array): Face picture to process
        emotion_detector: loaded model for face expression classification

    Returns:
        vector: confidence for each emotion
    '''
    #cv2.imshow('image', face)
    #cv2.waitKey(5000)
    mean = np.mean(face)
    #a=0.43/mean
    #face=face*a
    image = np.reshape(face,(1,1,180,192))
    mean = np.mean(face)
    #print(mean)
    # print(image)
    #print(image)

    image = torch.tensor(image).cuda()


    #optimizer.zero_grad()
    #total_step += 1
    torch.cuda.empty_cache()
    out = emotion_detector.forward(image)
    #print(out)
    out = out.cpu()
    #print(out)
    out = out.detach().numpy()
    #print(out.shape)
    #out[0,3] = out[0,3] - 0.6
    #out[0, 4] = out[0, 4] - 1.0
    #out[0, 0] = out[0, 0] + 2.0
    out[0, 1] = out[0, 1] + 3.3
    out[0, 2] = out[0, 1] + 0.3
    out[0, 7] = out[0, 7] + 0.3

    out=torch.from_numpy(out).cuda()
    out=F.softmax(out,dim=1)
    out=out.cpu()
    #print(out)
    out=out.detach().numpy()
    #print(out)
    emotion_predictions=out

    #emotion_predictions = emotion_detector.predict(face)[0]
    return emotion_predictions


def draw_results(image, emotion_predictions, coords, args):
    ''' Put on the image rectangle around face and text about it's emotion

    Args:
        image (array): to put rectangle and text on
        emotion_predictions: vector of confidences for each emotion
        coords: [x1, y1, x2, y2] vector of face location

    Returns:
        image (array): input image after processing
    '''

    emotion_probability = np.max(emotion_predictions)
    emotion_label_arg = np.argmax(emotion_predictions)
    # show result
    if (args.show):
        if emotion_label_arg == 0:  # neutral
            color = emotion_probability * np.asarray((100, 100, 100))
        elif emotion_label_arg == 1:  # calm
            color = emotion_probability * np.asarray((100, 100, 100))
        elif emotion_label_arg == 2:  # happy
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_label_arg == 3:  # sad
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_label_arg == 4:  # angry
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_label_arg == 5:  # fearful
            color = emotion_probability * np.asarray((100, 100, 100))
        elif emotion_label_arg == 6:  # disgust
            color = emotion_probability * np.asarray((100, 100, 100))
        elif emotion_label_arg == 7:  # surprized
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()

        cv2.rectangle(image, (coords[0], coords[1]), (coords[2], coords[3]), color, 1)

        # write emotion text above rectangle
        emotion_percent = str(np.round(emotion_probability * 100))
        cv2.putText(image, emotion_percent + "%", (coords[0], coords[3] + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        emotion_en = emotions_en[emotion_label_arg]
        # emotion_ru = emotions_ru[emotion_label_arg]
        cv2.putText(image, emotion_en, (coords[0], coords[3] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    return image


def detect_emotions(image, emotion_detector, face_detector, args):
    ''' Detects emotion on image

    Args:
        image (array): image to detect emotions on
        emotion_detector: loaded model for face expression classification
        face_detector: loaded model for face detection
        args: command arguments

    Returns:
        bool: True if process was successfull
        image (array): input image after processing
    '''
    global images

    # detect faces
    success, faces = get_faces(image, face_detector)
    if (success):

        # loop through all found faces
        for f in range(faces.shape[2]):

            confidence = faces[0, 0, f, 2]
            if confidence > args.conf_threshold:
                aaa=50
                bbb=70
                x1 = int(faces[0, 0, f, 3] * image.shape[1])
                y1 = int(faces[0, 0, f, 4] * image.shape[0])
                x2 = int(faces[0, 0, f, 5] * image.shape[1])
                y2 = int(faces[0, 0, f, 6] * image.shape[0])

                #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                #print(x1,x2,y1,y2)
                # detected_face = fa.align(image, gray, dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2))
                detected_face = image[y1:y2, x1:x2]
                if detected_face.size != 0:

                    # resize, normalize and save the frame (convert to grayscale if frames_resolution[-1] == 1)
                    detected_face=np.asarray(Image.fromarray(detected_face).convert('L').resize((192, 180)))
                    detected_face = detected_face / 255.
                    #print(detected_face.shape)
                    #detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
                    #print(detected_face)
                    #detected_face[0:(aaa-15), :] = 255
                    #detected_face[:, 0:bbb] = 255
                    #detected_face[:, -bbb:] = 255
                    #detected_face[-(aaa-15):, bbb:-bbb] = 0
                    #detected_face[-(aaa - 20):, :] = 0
                    #detected_face = cv2.resize(detected_face,(192,180))
                    #print(detected_face.shape)
                                               #(emotion_detector.input_shape[-3], emotion_detector.input_shape[-2]))
                    #detected_face = cv2.normalize(detected_face, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
                    #print(detected_face.shape)
                    #print(detected_face)


                    emotion_predictions = detect_emotion_image(detected_face, emotion_detector)
                    image = draw_results(image, emotion_predictions, [x1, y1, x2, y2], args)

    else:
        print("Unsuccessfull image processing")
        success = False

    return success, image


if __name__ == '__main__':
    warnings.filterwarnings("ignore")  # todo

    torch.cuda.empty_cache()

    writer = SummaryWriter(log_dir='./resultsimg')

    with open('config.yaml') as f:
        config = yaml.load(f)

    image_model = reconmodels.resnet50withcbam()

    image_model.load_state_dict(torch.load(config['save_path'] + 'imgbest.pth'))
    #image_model.eval()
    #optimizer = torch.optim.Adam(image_model.parameters(), lr=0.0001)

    image_model.cuda()
    # ___ FACE DETECTOR MODEL ___
    modelFile = "models/opencv_face_detector_uint8.pb"
    configFile = "models/opencv_face_detector.pbtxt"
    face_detector = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

    # ___ FACE ALIGNER ___ (uses emotion recognition model input shape)
    #predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
    #fa = FaceAligner(predictor, desiredFaceWidth=180)#image_model.input_shape[-3])

    # output file path
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    name = "result" + (os.path.normpath(args.source).split("\\")[-1] if (args.input != "camera") else ".avi")
    save_path = os.path.join(args.output_dir, name)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # cv2.VideoWriter_fourcc(*'MPEG')
    vid = None

    # ___ PROCESS IMAGE ___
    if (args.input == "image"):

        # read image
        image = cv2.imread(args.source, 1)
        if image is None:
            print("Can't find {} image".format(args.source))
            sys.exit(-1)

            # detect emotions
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image, (192, 180))
        print(image.shape)
        # (emotion_detector.input_shape[-3], emotion_detector.input_shape[-2]))
        image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        cv2.imshow('image', image)
        cv2.waitKey(5000)

        # detected_face = np.expand_dims(detected_face, axis=3)
        emotion_predictions = detect_emotion_image(image, image_model)
        print(emotion_predictions)
        #image = draw_results(image, emotion_predictions, [x1, y1, x2, y2], args)
        #success, image = detect_emotions(image, image_model, face_detector, args)
        if 1:

            # save result
            cv2.imwrite(save_path, image)
            print("Result was saved in {}".format(save_path))

            # show result
            if (args.show):
                cv2.imshow('image', image)
                cv2.waitKey(5000)

        else:
            print("Unsuccessfull image processing")
    # ___ PROCESS WEBCAM STREAM ___
    if (args.input == "camera"):

        images = []
        cam = cv2.VideoCapture(args.camera_number)
        #cam.set(3, 1280)  # 设置分辨率
        #cam.set(4, 720)
        _, image = cam.read()
        print("Camera image shape: {}x{}".format(image.shape[1], image.shape[0]))
        fps_time = 0

        while True:

            # read image
            _, image = cam.read()

            # detect emotions
            success, image = detect_emotions(image, image_model, face_detector, args)
            if (success):

                # show result
                cv2.putText(image,
                            "FPS: {}".format(1.0 / (time.time() - fps_time)),
                            (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)
                cv2.imshow('image', image)
                fps_time = time.time()

                if vid == None:
                    vid = cv2.VideoWriter(filename=save_path, fourcc=fourcc, fps=float(args.fps),
                                          apiPreference=cv2.CAP_FFMPEG, frameSize=(image.shape[1], image.shape[0]))

                    # save output
                vid.write(image)

            else:
                print("Unsuccessfull image processing")

            # wait Esc to be pushed
            if cv2.waitKey(1) == 27:
                break

        vid.release()
        cv2.destroyAllWindows()

