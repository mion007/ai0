import cv2

def AuthenticateFace():
    # Initialize the LBPH face recognizer and load the trained model
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('engine\\auth\\trainer\\trainer.yml')
    
    # Load the Haar Cascade classifier for face detection
    cascadePath = "engine\\auth\\haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    
    # Set font for displaying text on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Define the list of names corresponding to the IDs used in training
    names = ['', 'Digambar']  # Add more names as needed
    
    # Create a video capture object to capture video from the webcam
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(3, 640)  # Set video frame width
    cam.set(4, 480)  # Set video frame height
    
    # Define minimum window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)
    
    recognized_flag = False  # Flag to indicate if a face is recognized
    
    while True:
        # Read frames from the webcam
        ret, img = cam.read()
        if not ret:
            print("Failed to capture image")
            break
        
        # Convert the captured frame to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )
        
        for (x, y, w, h) in faces:
            # Draw a rectangle around the detected face
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Predict the identity of the face
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            
            # Check if confidence is less than 100 ==> "0" is a perfect match
            if confidence < 100:
                name = names[id]
                confidence_text = "  {0}%".format(round(100 - confidence))
                recognized_flag = True
            else:
                name = "unknown"
                confidence_text = "  {0}%".format(round(100 - confidence))
                recognized_flag = False
            
            # Display the name and confidence on the image
            cv2.putText(img, str(name), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence_text), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
        
        # Display the frame with the detected face
        cv2.imshow('camera', img)
        
        # Wait for a key press
        k = cv2.waitKey(10) & 0xff
        if k == 27:  # Press 'ESC' to exit
            break
        if recognized_flag:
            break
    
    # Do a bit of cleanup
    cam.release()
    cv2.destroyAllWindows()
    return recognized_flag

# Call the function to authenticate face
if AuthenticateFace():
    print("Face recognized successfully.")
else:
    print("Face not recognized.")