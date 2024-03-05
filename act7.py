import os 
import cv2 as cv
import numpy as np

names = ['abo', 'alferos','aquino', 'baconawa', 'spencer']

p = []
path = r'C:\Users\manol\OneDrive\Desktop\compvis\fqaces'
haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

for i in os.listdir(path):
    p.append(i)

features = []
labels = []
def trainData():
    for idx, person in enumerate(names):
        pic = os.path.join(path, person)
        label = idx

        for img in os.listdir(pic):
            img_path = os.path.join(pic, img)
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            
            face_rect = haar_cascade.detectMultiScale(gray, 1.3, 3)

            for (x,y,w,h) in face_rect:
                face_roi = gray[y:y+h, x:x+w]
                resize = cv.resize(face_roi, (200, 200))
                features.append(resize)
                labels.append(label)
trainData()
print(f'Lenght of Features: {len(features)}, Lenght of Labels: {len(labels)}')

face_recog = cv.face.LBPHFaceRecognizer_create()
labels = np.array(labels)
#Train the features
face_recog.train(features, labels)
print(labels)


confidence_threshold = 90

# Open a connection to the webcam (0 represents the default camera)
cap = cv.VideoCapture(1)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)

    for (x, y, w, h) in faces:
        # Crop the face region from the frame
        face_roi = gray[y:y + h, x:x + w]

        # Recognize the face using the LBPH recognizer
        resize = cv.resize(face_roi, (200, 200))
        label, confidence = face_recog.predict(resize)

        # Draw a rectangle around the face and display the label and confidence
        if confidence < confidence_threshold:
            label_text = f'Label: {names[label]}'
        else:
            label_text = 'Unknown'

        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(frame, label_text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv.putText(frame, f'Confidence: {confidence}', (x, y + h + 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the resulting frame
    cv.imshow('Face Recognition', frame)

    # Break the loop when 'q' key is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv.destroyAllWindows()
