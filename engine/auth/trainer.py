import cv2
import numpy as np
from PIL import Image
import os

def train_model():
    path = 'engine\\auth\\samples'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("engine\\auth\\haarcascade_frontalface_default.xml")

    def get_images_and_labels(path):
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        face_samples = []
        ids = []

        for image_path in image_paths:
            gray_img = Image.open(image_path).convert('L')
            img_arr = np.array(gray_img, 'uint8')
            id = int(os.path.split(image_path)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_arr)

            for (x, y, w, h) in faces:
                face_samples.append(img_arr[y:y+h, x:x+w])
                ids.append(id)

        return face_samples, ids

    print("Training faces. It will take a few seconds. Wait ...")
    faces, ids = get_images_and_labels(path)
    recognizer.train(faces, np.array(ids))
    recognizer.write('engine\\auth\\trainer\\trainer.yml')
    print("Model trained. Now we can recognize your face.")

# Example usage:
train_model()