import numpy as np
import cv2
#import winsound, platform, thread
face_cascade = cv2.CascadeClassifier('/home/arosha/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('/home/arosha/opencv-3.1.0/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
lefteye_cascade = cv2.CascadeClassifier('/home/arosha/opencv-3.1.0/data/haarcascades/haarcascade_lefteye_2splits.xml')

video_capture = cv2.VideoCapture(1)
count = 0
iters = 0

#def beep():
#  for i in range(4):
#    winsound.Beep(1500, 250)


while True:
    ret, frame = video_capture.read()
    if(ret==True):
      grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);
      faces = face_cascade.detectMultiScale(grayFrame, 1.3, 5)
      for(x,y,w,h) in faces:
          frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
          roi_gray = grayFrame[y:y+h,x:x+w]
          roi_color = frame[y:y+h,x:x+w]
          eyes = eye_cascade.detectMultiScale(roi_gray)
          for (ex, ey, ew, eh) in eyes:
              eye_frame = cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 255), 2)
              eye_gray_frame = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY);

              circles = cv2.HoughCircles(eye_gray_frame, cv2.HOUGH_GRADIENT, 0.1, 20)

              # ensure at least some circles were found
              if circles is not None:
                  # convert the (x, y) coordinates and radius of the circles to integers
                  circles = np.round(circles[0, :]).astype("int")
                  cv2.imshow("Eye", eye_gray_frame)

                  # loop over the (x, y) coordinates and radius of the circles
                  for (x, y, r) in circles:
                      # draw the circle in the output image, then draw a rectangle
                      # corresponding to the center of the circle
                      cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                      cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)



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




      cv2.imshow("Image",frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()