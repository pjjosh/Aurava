import cv2
import detection_utils
from emotion_model import create_model, load_model_weights

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

model = create_model()
model = load_model_weights(model)

cv2.ocl.setUseOpenCL(False)

# Start the webcam feed
cap = cv2.VideoCapture(0)
capture_flag = False

def on_mouse(event, x, y, flags, param):
    global capture_flag
    if event == cv2.EVENT_LBUTTONDOWN:
        capture_flag = True

cv2.namedWindow('Live')
cv2.setMouseCallback('Live', on_mouse)


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # detect & draw live boxes
    live_annotated, last_emotion = detection_utils.detect_and_annotate(frame, model, emotion_dict)

    cv2.imshow('Live', live_annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    # on click: do full emotion detect & exit
    if capture_flag:
        cv2.imwrite('captured_emotion.jpg', live_annotated)
        print("Captured emotion:", last_emotion or "None")
        capture_flag = False
        break

if last_emotion is not None:
    print(last_emotion)
else:
    print("No face detected.")


        
cv2.release = cap.release
cap.release()
cv2.destroyAllWindows()
    
