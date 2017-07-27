import numpy as np
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import time


face_cascade = cv2.CascadeClassifier('/home/pi/Softwares/opencv-3.0.0/data/haarcascades/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('/home/pi/Softwares/opencv-3.0.0/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer.load("recognizer/trainingData.yml")

camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(320,240))
time.sleep(0.1)

statusString = ""

def detectSleep(statusString):
        
        if (len(statusString) >= 10 and statusString[-1:-11:-1]=="1111111111"):
                return "sleeping"
        else:
                return "not sleeping"

while True:
        for imageFrame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
                img = imageFrame.array

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(img, 1.2, 5)
                
                for (x,y,w,h) in faces:
                        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                        face_detected = gray[y:y+h, x:x+w]
                        ID= recognizer.predict(face_detected)
                
                        if (ID[0]==1):
                                cv2.putText(img, "Tharushi", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
                                print("Tharushi",)
                        elif(ID[0]==2):
                                cv2.putText(img, "Darshana", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
                                print("Darshana",)
                        elif(ID[0]==3):
                                cv2.putText(img, "Chamath", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
                                print("Chamath",)
                        elif(ID[0]==4):
                                cv2.putText(img, "Arosha", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
                                print("Arosha",)
                        elif (ID[0]==5):
                                cv2.putText(img, "Malaka", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
                                print("Malaka",)
                        elif (ID[0]==6):
                                cv2.putText(img, "Chanuka", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
                                print("Chanuka",)
                        elif (ID[0]==7):
                                cv2.putText(img, "Tharushi", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
                                print("Tharushi",)
                        elif (ID[0]==12):
                                cv2.putText(img, "Tharushi", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
                                print("Tharushi",)
                        elif (ID[0]==10):
                                cv2.putText(img, "Chandana Sir", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
                                print("Chandana Sir",)
                        print("\tAccuracy Level :", ID[1],"%",)
                        
                        roi_gray = gray[y:y+h, x:x+w]
                        roi_color = img[y:y+h, x:x+w]
                        eyes = eye_cascade.detectMultiScale(roi_gray)	
                        count = 0
                        for (ex,ey,ew,eh) in eyes:
                                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,255,255),2)
                                count+=1
                        if(count>1):
                                cv2.putText(img, "eyes Open", (x,y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
                                statusString+="0"
                        else:
                                cv2.putText(img, "eyes Closed", (x,y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
                                statusString += "1"
                        
                        print (detectSleep(statusString))
                cv2.imshow('img',img)
                rawCapture.truncate(0)
                if cv2.waitKey(1) & 0xFF == ord('q'): # escape with the keyboard character q
                        break
#cap.release()
cv2.destroyAllWindows()

