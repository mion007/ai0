import time
import cv2
import pyautogui as p

def AuthenticateFace():
    print("Starting face authentication...")

    # Local Binary Patterns Histograms
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('engine\\auth\\trainer\\trainer.yml')  # load trained model
        print("Loaded trained model.")
    except cv2.error as e:
        print("Error loading recognizer or model:", e)
        return 0

    cascadePath = "engine\\auth\\haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    print("Loaded cascade classifier.")

    font = cv2.FONT_HERSHEY_SIMPLEX  # denotes the font type
    names = ['', 'Digambar']  # names, leave first empty bcz counter starts from 0

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # cv2.CAP_DSHOW to remove warning
    if not cam.isOpened():
        print("Error: Could not open camera.")
        return 0

    cam.set(3, 640)  # set video FrameWidth
    cam.set(4, 480)  # set video FrameHeight

    # Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)
    flag = 0  # Initialize flag

    while True:
        ret, img = cam.read()  # read the frames using the above created object
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        # The function converts an input image from one color space to another
        converted_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            converted_image,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:
            # used to draw a rectangle on any image
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # to predict on every single image
            id, accuracy = recognizer.predict(converted_image[y:y + h, x:x + w])

            # Check if accuracy is less than 100 ==> "0" is a perfect match
            if accuracy < 100:
                id = names[id]
                accuracy = "  {0}%".format(round(100 - accuracy))
                flag = 1
            else:
                id = "unknown"
                accuracy = "  {0}%".format(round(100 - accuracy))
                flag = 0

            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(accuracy), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        cv2.imshow('camera', img)

        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27 or flag == 1:
            break

    # Do a bit of cleanup
    cam.release()
    cv2.destroyAllWindows()
    print("Face authentication completed with flag:", flag)
    return flag

# Run the function to test
if __name__ == "__main__":
    AuthenticateFace()