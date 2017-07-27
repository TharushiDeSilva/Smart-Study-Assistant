import numpy as np
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

#import winsound, platform, thread
face_cascade = cv2.CascadeClassifier('/home/pi/Softwares/opencv-3.0.0/data/haarcascades/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('/home/pi/Softwares/opencv-3.0.0/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
lefteye_cascade = cv2.CascadeClassifier('/home/pi/Softwares/opencv-3.0.0/data/haarcascades/haarcascade_lefteye_2splits.xml')
 
camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(320,240))
 
# allow the camera to warmup
time.sleep(0.1)

#video_capture = cv2.VideoCapture(1)
count = 0
iters = 0


while True:
    #ret, frame = video_capture.read()
    for imageFrame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        frame = imageFrame.array
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);
        faces = face_cascade.detectMultiScale(grayFrame, 1.3, 5)
        for(x,y,w,h) in faces:
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = grayFrame[y:y+h,x:x+w]
            roi_color = frame[y:y+h,x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) == 0:
                print ("Eyes closed")
            else:
                print ("Eyes open")
            count += len(eyes)
            iters += 1
            if iters == 2:
                iters = 0
                if count == 0:
                    print ("Drowsiness Detected!!!")
                    #thread.start_new_thread(beep,())
                count = 0
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh), (0,255,0),2)



        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
