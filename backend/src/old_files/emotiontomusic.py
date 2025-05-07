import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


def detect_and_annotate(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    out = frame.copy()
    emotion = None

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        crop = cv2.resize(roi, (48,48))
        crop = crop.astype('float32')/255.0
        crop = np.expand_dims(crop, axis=(0,-1))  # shape (1,48,48,1)
        preds = model.predict(crop)
        idx = int(np.argmax(preds))
        emotion = emotion_dict[idx]

        # draw
        cv2.rectangle(out, (x, y-50), (x+w, y+h+10), (255,0,0), 2)
        cv2.putText(out, emotion, (x+20, y-60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    return out, emotion

def on_mouse(event, x, y, flags, param):
    global capture_flag
    if event == cv2.EVENT_LBUTTONDOWN:
        capture_flag = True




# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
  
        
model.load_weights('model.h5')

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# start the webcam feed
cap = cv2.VideoCapture(0)

cv2.namedWindow('Live')
capture_flag = False
cv2.setMouseCallback('Live', on_mouse)

video_stream = True

while video_stream:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow('Live', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if capture_flag:
        # 4) On click: process that frame
        processed, emotion = detect_and_annotate(frame)
        cv2.imshow('Captured → Emotion', processed)
        cv2.imwrite('captured_emotion.jpg', processed)  # optional save
        capture_flag = False
        
        if emotion is None:
            print("No emotion detected")
        else:
            print(emotion)
        

cv2.release = cap.release
cap.release()
cv2.destroyAllWindows()