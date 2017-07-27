import cv2
import numpy as np

recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer.load('recognizer/trainingData.yml')

cascadePath = "/home/tharushi/opencv-3.1.0/data/haarcascades/haarcascade_frontalcatface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
cam = cv2.VideoCapture(0)
#font = cv2.InitFont(cv2.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
	ret, img = cam.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray, 1.2, 5)
	for (x, y, w, h) in faces:
		cv2.rectangle(img, (x-50, y-50), (x+w+50, y+h+50), (255, 0, 0), 2)
		Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
		if(conf<50):
			if(Id ==1):
				Id = "Tharushi"
			else:
				Id = "Unknown"
		cv2.PutText(cv2.cv.fromarray(img), str(Id), (x, y+h), font, 255)
	cv2.imshow('image', img)
	if(cv2.waitKey(10) & 0xFF == ord('q')):
		break

cam.release()
cv2.destroyAllWindows()	
waitKey()
#cv2.putText(frame, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)

