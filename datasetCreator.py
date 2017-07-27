import numpy as np
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

face_cascade = cv2.CascadeClassifier('/home/pi/Softwares/opencv-3.0.0/data/haarcascades/haarcascade_frontalface_alt.xml')
camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(320,240))
time.sleep(0.1)

id = input("Enter user ID : ")
sampleNum = 0

cv2.namedWindow("face", cv2.WINDOW_AUTOSIZE)

while (True):
    for imageFrame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        img = imageFrame.array
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img, 1.2, 5)
        for (x,y,w,h) in faces:
            sampleNum += 1
            cv2.imwrite("dataSet/user."+str(id)+"."+str(sampleNum)+".jpg", gray[y:y+h, x:x+w])
            img = cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.waitKey(100)
            cv2.imshow("face", img)
            cv2.waitKey(1)
        if (sampleNum == 40):
            break

        # show the frame
        cv2.imshow("Frame", img)
        key = cv2.waitKey(1) & 0xFF
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
#cam.release();  
cv2.destroyAllWindows();


