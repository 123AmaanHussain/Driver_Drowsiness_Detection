import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os
import urllib.request

model = tf.keras.models.load_model("models/drowsiness_model.keras")

MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("[INFO] Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = FaceLandmarker.create_from_options(options)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def get_eye_bbox(landmarks, indices, w, h, pad=0.3):
    coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    pw = int((max(xs) - min(xs)) * pad)
    ph = int((max(ys) - min(ys)) * pad)
    return (max(0, min(xs) - pw), max(0, min(ys) - ph), min(w, max(xs) + pw), min(h, max(ys) + ph))

cap = cv2.VideoCapture(0)
frame_timestamp = 0

print("=" * 50)
print("CALIBRATION MODE")
print("Keep eyes OPEN and note the prediction values")
print("Then close eyes and note those values")
print("Press 'q' to quit")
print("=" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    frame_timestamp += 33
    results = landmarker.detect_for_video(mp_image, frame_timestamp)

    if results.face_landmarks:
        for face_landmarks in results.face_landmarks:
            for eye_points, label, color in [(LEFT_EYE, "LEFT", (255, 0, 0)), (RIGHT_EYE, "RIGHT", (0, 255, 0))]:
                x1, y1, x2, y2 = get_eye_bbox(face_landmarks, eye_points, w, h)
                if x2 > x1 and y2 > y1:
                    eye_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y1:y2, x1:x2]
                    eye_resized = cv2.resize(eye_gray, (24, 24))
                    eye_norm = eye_resized / 255.0
                    eye_input = eye_norm.reshape(1, 24, 24, 1)

                    prediction = model.predict(eye_input, verbose=0)
                    pred_val = float(prediction[0][0])

                    # Draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    text = f"{label}: {pred_val:.4f}"
                    cv2.putText(frame, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    print(f"[{label}] {pred_val:.4f}")

    cv2.putText(frame, "OPEN = note value, CLOSE = note value", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("Calibration - Raw Predictions", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
print("[INFO] Calibration ended.")
