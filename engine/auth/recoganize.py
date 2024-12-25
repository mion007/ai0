import cv2

def AuthenticateFace():
    flag = ""
    # Local Binary Patterns Histograms recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Load trained model
    try:
        recognizer.read('engine\\auth\\trainer\\trainer.yml')
        print("Model loaded successfully.")
    except cv2.error as e:
        print("Error: Could not read the trained model. Ensure the model is trained and the path is correct.")
        return -1  # Indicate failure to load model

    cascadePath = "engine\\auth\\haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)

    # Font for displaying text
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Names corresponding to id (index 0 is empty as id starts from 1)
    names = ['', 'Digambar']  # Add any other users here

    # Initialize webcam
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        print("Error: Could not open camera.")
        return -1  # Indicate failure to open camera

    cam.set(3, 640)  # Set video frame width
    cam.set(4, 480)  # Set video frame height

    # Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    print("Camera initialized. Please look at the camera.")

    while True:
        ret, img = cam.read()  # Read frames from the camera
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Convert to grayscale
        converted_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            converted_image,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, accuracy = recognizer.predict(converted_image[y:y + h, x:x + w])

            # Check if accuracy is greater than or equal to 55%
            if accuracy < 45:  # 100 - 55 = 45
                person_name = names[id]
                accuracy_text = "  {0}%".format(round(100 - accuracy))
                flag = 1
                print(f"Recognized {person_name} with accuracy {accuracy_text}")
            else:
                person_name = "unknown"
                accuracy_text = "  {0}%".format(round(100 - accuracy))
                flag = 0
                print(f"Unrecognized face with accuracy {accuracy_text}")

            cv2.putText(img, str(person_name), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(accuracy_text), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        cv2.imshow('camera', img)
        if cv2.waitKey(10) & 0xff == 27 or flag == 1:  # Press 'ESC' to exit
            break

    cam.release()
    cv2.destroyAllWindows()

    return flag

# Example usage:
result = AuthenticateFace()
if result == 1:
    print("Authentication successful.")
elif result == 0:
    print("Authentication failed.")
else:
    print("An error occurred during the authentication process.")