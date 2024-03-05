# Make sure that for this activity, you have downloaded the
# file indicated below from the resource linked in the instructional materials
# in the module.

import cv2


picPath = r'C:\Users\manol\OneDrive\Desktop\compvis\faces\abo\abo (10).jpg'
haarPath = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

def faceDetect(picPath, name):
    face_cascade = cv2.CascadeClassifier(haarPath)

    img = cv2.imread(picPath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 3)

    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)

    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


for i in range(1, 55):
    path = r'C:\Users\manol\OneDrive\Desktop\compvis\faces\test\a (' + str(i) + ').jpg'
    faceDetect(path, str(i))