import cv2
import numpy as np
from PIL import Image  # pillow package
import os

path = 'engine\\auth\\samples'  # Path for samples already taken

recognizer = cv2.face.LBPHFaceRecognizer_create()  # Local Binary Patterns Histograms
detector = cv2.CascadeClassifier("engine\\auth\\haarcascade_frontalface_default.xml")
# Haar Cascade classifier is an effective object detection approach

def Images_And_Labels(path):  # function to fetch the images and labels
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path does not exist: {path}")
    
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:  # to iterate particular image path
        try:
            gray_img = Image.open(imagePath).convert('L')  # convert it to grayscale
        except Exception as e:
            print(f"Warning: Could not read image {imagePath}. Error: {e}")
            continue
        
        img_arr = np.array(gray_img, 'uint8')  # creating an array

        try:
            id = int(os.path.split(imagePath)[-1].split(".")[1])
        except ValueError:
            print(f"Warning: Could not extract ID from image {imagePath}")
            continue
        
        faces = detector.detectMultiScale(img_arr)

        for (x, y, w, h) in faces:
            faceSamples.append(img_arr[y:y+h, x:x+w])
            ids.append(id)
    
    return faceSamples, ids

print("Training faces. It will take a few seconds. Wait ...")

faces, ids = Images_And_Labels(path)

if len(faces) > 1 and len(ids) > 1:
    recognizer.train(faces, np.array(ids))
    recognizer.write('engine\\auth\\trainer\\trainer.yml')  # Save the trained model as trainer.yml
    print("Model trained, Now we can recognize your face.")
else:
    print("Error: Not enough data to train the model. Please provide more samples.")