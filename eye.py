import cv2
import time

face_cascade_path = "haarcascade_frontalface_alt.xml"
eye_cascade_path = 'haarcascade_eye.xml'

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

src = cv2.imread('test14.jpg')
print("src",src)
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
equ_gray = cv2.equalizeHist(src_gray)
faces = face_cascade.detectMultiScale(equ_gray,scaleFactor=1.13,minNeighbors=2)

for x, y, w, h in faces:
    cv2.rectangle(src, (x, y), (x + w, y + h), (255, 0, 255), 2)
    face = src[y: y + h, x: x + w]
    face_gray = src_gray[y: y + h, x: x + w]
    eyes = eye_cascade.detectMultiScale(face_gray)
    # for (ex, ey, ew, eh) in eyes:
    #     cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
cv2.imwrite('./opencv_face_detect_rectangle.jpg', src)
# print(f"faces : {faces}")
# print(f"eyes : {eyes}")
# print("faces : ",faces)
# print("eyes : ",eyes)
for x, y, w, h in faces:
    face = src[y: y + h, x: x + w]
    for (ex, ey, ew, eh) in eyes:
        cut_img = face[ey : ey + eh, ex:ex + ew]
        cv2.imwrite(str(time.time())+".jpg", cut_img)