import cv2
import numpy as np


face_cascade = cv2.CascadeClassifier("/home/tharushi/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml")
recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer.load("recognizer/trainingData.yml")
cam = cv2.VideoCapture(0)
while True:
	ret, img = cam.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(img, 1.2, 5)
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
		#cv2.putText(img, "FaceRecognized", (x,y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
		face_detected = gray[y:y+h, x:x+w]
		ID= recognizer.predict(face_detected)
		if (ID==1):
			cv2.putText(img, "Tharushi", (x,y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
			print("Tharushi")
		elif(ID==2):
			cv2.putText(img, "Darshana", (x,y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
			print("Darshana")
		elif(ID==3):
			cv2.putText(img, "Chamath", (x,y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
			print("Chamath")
		elif(ID==4):
			cv2.putText(img, "Arosha", (x,y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
			print("Arosha")
		else:
			print("Not detected")
	cv2.imshow("Live Video", img)
	if cv2.waitKey(1) & 0xFF == ord('q'): # escape with the keyboard character q
        	break
	
	
cam.release()
cv2.destroyAllWindows()
		
#cv2.putText(frame, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)

