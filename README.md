# Real Time Human Emotion Detector Demo


It's a emotion recognition demo based on an image-only model. You can simply detect your own emotion through your computer camera. 
It supports detecting emotion for multiple people, but it may have delay for more than 3 people. For only 1 face, the demo can reach about 20fps in cpu-only mode.

The algorithm first uses OpenCV to extract faces from each frame, then uses an attention Resnet network to detect emotion from faces.

# How to run:
1. Using detector.py to run.
2. If you don't have a cuda GPU with your machine, please modify the detect_emotion_image function in detector.py
3. To make the real-time detector work well, please make sure in good illumination

# To train your own model:
The model is trained based the Ravdess Dataset and I use OpenCV-tensorflow module to extract all the face area from dataset as training samples.
If you want to train your own dataset, please use traintest.py for training.
