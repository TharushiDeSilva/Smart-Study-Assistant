import numpy as np
import cv2

# Eye Detection

face_cascade = cv2.CascadeClassifier("/home/tharushi/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("/home/tharushi/opencv-3.1.0/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml")

cap = cv2.VideoCapture(0)

while True:
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(img, 1.2, 5)

	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray)	
		count = 0
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,255,255),2)
			count+=1
		if(count>1):
			cv2.putText(img, "eyes Open", (x,y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
		else:
			cv2.putText(img, "eyes Closed", (x,y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))		
	cv2.imshow('img',img)
    
	if cv2.waitKey(1) & 0xFF == ord('q'): # escape with the keyboard character q
		break
cap.release()
cv2.destroyAllWindows()

