import os

import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict

# Initializing the Model
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2)

cascPath1=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
cascPath2=os.path.dirname(cv2.__file__)+"/data/haarcascade_eye.xml"
cascPath3=os.path.dirname(cv2.__file__)+"/data/haarcascade_smile.xml"

faceCascade = cv2.CascadeClassifier(cascPath1)
eyeCascade = cv2.CascadeClassifier(cascPath2)
smileCascade = cv2.CascadeClassifier(cascPath3)

cap = cv2.VideoCapture(0)
cap.set(3,640) 
cap.set(4,480) 

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
  
        # Both Hands are present in image(frame)
        if len(results.multi_handedness) == 2:
                # Display 'Both Hands' on the image
            cv2.putText(img, 'Both Hands', (250, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 0.9,
                        (0, 255, 0), 2)
  
        # If any hand present
        else:
            for i in results.multi_handedness:
                
                # Return whether it is Right or Left Hand
                label = MessageToDict(i)[
                    'classification'][0]['label']
  
                if label == 'Left':
                    
                    # Display 'Left Hand' on left side of window
                    cv2.putText(img, label+' Hand', (20, 50),
                                cv2.FONT_HERSHEY_COMPLEX, 0.9,
                                (0, 255, 0), 2)
  
                if label == 'Right':
                    
                    # Display 'Left Hand' on left side of window
                    cv2.putText(img, label+' Hand', (460, 50),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.9, (0, 255, 0), 2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5, 
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eyeCascade.detectMultiScale(
            roi_gray,
            scaleFactor= 1.5,
            minNeighbors=5,
            minSize=(5, 5),
            flags=cv2.CASCADE_SCALE_IMAGE
            )
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)             
        
        smile = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor= 1.5,
            minNeighbors=15,
            minSize=(25, 25),
            flags=cv2.CASCADE_SCALE_IMAGE
            )
        
        for (xx, yy, ww, hh) in smile:
            cv2.rectangle(roi_color, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2)


        
        cv2.imshow('video', img)

#    k = cv2.waitKey(30) & 0xff
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

cap.release()
cv2.destroyAllWindows()




