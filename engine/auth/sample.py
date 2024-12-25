import cv2
import os

def capture_face_samples(user_id, user_name):
    # Create directory for user if it doesn't exist
    sample_dir = f'engine\\auth\\samples\\user_{user_id}'
    os.makedirs(sample_dir, exist_ok=True)

    # Initialize webcam
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        print("Error: Could not open camera.")
        return -1

    face_detector = cv2.CascadeClassifier('engine\\auth\\haarcascade_frontalface_default.xml')
    sample_count = 0

    print("Capturing face samples. Look at the camera and wait...")

    while True:
        ret, img = cam.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            sample_count += 1
            # Save the captured image into the samples folder
            cv2.imwrite(f"{sample_dir}\\{user_name}.{user_id}.{sample_count}.jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('image', img)

        # Break the loop after 30 samples
        if sample_count >= 30:
            break

        if cv2.waitKey(100) & 0xff == 27:  # Press 'ESC' to exit
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"Captured {sample_count} face samples for user {user_name}.")

# Example usage:
capture_face_samples(1, "Digambar")