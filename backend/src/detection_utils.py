# backend/src/detection_utils.py
import cv2
import numpy as np
import os
import uuid
from .emotion_model import create_model, load_model_weights

# load your Keras model once
_model = create_model()
_model = load_model_weights(_model)

# load your cascade as before
cascade_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'models',
    'haarcascade_frontalface_default.xml'
)
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    raise IOError(f"Cannot load cascade classifier from {cascade_path}")

def detect_from_bytes(img_bytes: bytes) -> str:
    # decode incoming JPEG to BGR array
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # ==== DEBUG: dump what we got ====
    fn = f"debug_{uuid.uuid4().hex[:8]}.jpg"
    cv2.imwrite(fn, img)
    print(f"[DEBUG] wrote incoming frame to {fn}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # try more lenient detection on small crops
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    print(f"[DEBUG] faces found: {faces}")

    if len(faces) == 0:
        return "no_face"

    # pick first face
    x, y, w, h = faces[0]
    roi = gray[y:y+h, x:x+w]
    crop = cv2.resize(roi, (48, 48)).astype('float32') / 255.0
    crop = np.expand_dims(crop, axis=(0, -1))

    preds = _model.predict(crop)
    idx = int(np.argmax(preds))
    emotion_dict = {
        0: "Angry", 1: "Disgusted", 2: "Fearful",
        3: "Happy",  4: "Neutral",   5: "Sad",
        6: "Surprised"
    }
    return emotion_dict.get(idx, "unknown")
