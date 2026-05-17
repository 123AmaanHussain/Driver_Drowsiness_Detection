import os
import urllib.request
import time
import cv2
import numpy as np
import tensorflow as tf
import pygame
import mediapipe as mp
from flask import Flask, Response, jsonify
from flask_cors import CORS
import threading

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# -------------------------
# Load Model
# -------------------------
model = tf.keras.models.load_model("models/drowsiness_model.keras")

# -------------------------
# Initialize Sound
# -------------------------
pygame.mixer.init()
alarm = pygame.mixer.Sound("alarm.wav")

# -------------------------
# MediaPipe Tasks API (FaceLandmarker)
# -------------------------
MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("[INFO] Downloading face landmarker model (~3MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("[INFO] Download complete.")

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

# Eye landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Global metrics dictionary to share status with API endpoint
system_status = {
    "status": "safe",  # 'safe', 'warning', 'danger'
    "ear_avg": 0.0,
    "counter": 0,
    "eyes_found": 0,
    "closed_eyes": 0,
    "alarm_on": False
}

status_lock = threading.Lock()

def eye_aspect_ratio(landmarks, indices, w, h):
    p = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    v1 = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
    v2 = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
    h_dist = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
    return (v1 + v2) / (2.0 * h_dist) if h_dist > 0 else 0

def get_eye_bbox(landmarks, indices, w, h, pad=0.3):
    coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    pw = int((max(xs) - min(xs)) * pad)
    ph = int((max(ys) - min(ys)) * pad)
    return (max(0, min(xs) - pw), max(0, min(ys) - ph), min(w, max(xs) + pw), min(h, max(ys) + ph))

def generate_frames():
    global system_status
    counter = 0
    alarm_on = False
    DROWSY_THRESHOLD = 10

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        frame_timestamp = int(time.time() * 1000)  # real wall-clock ms — always monotonically increasing
        results = landmarker.detect_for_video(mp_image, frame_timestamp)

        eyes_found = 0
        closed_eyes = 0
        ear_avg = 0.0

        if results.face_landmarks:
            for face_landmarks in results.face_landmarks:
                for idx in LEFT_EYE + RIGHT_EYE:
                    x = int(face_landmarks[idx].x * w)
                    y = int(face_landmarks[idx].y * h)
                    cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)

                for eye_points in [LEFT_EYE, RIGHT_EYE]:
                    ear = eye_aspect_ratio(face_landmarks, eye_points, w, h)
                    ear_avg += ear

                    x1, y1, x2, y2 = get_eye_bbox(face_landmarks, eye_points, w, h)

                    if x2 > x1 and y2 > y1:
                        eyes_found += 1
                        eye_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y1:y2, x1:x2]
                        eye_resized = cv2.resize(eye_gray, (24, 24))
                        eye_norm = eye_resized / 255.0
                        eye_input = eye_norm.reshape(1, 24, 24, 1)

                        prediction = model.predict(eye_input, verbose=0)

                        is_closed = ear < 0.20
                        if is_closed:
                            closed_eyes += 1

                        color = (0, 0, 255) if is_closed else (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"EAR:{ear:.2f}", (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                if eyes_found > 0:
                    ear_avg /= eyes_found

                if eyes_found > 0 and closed_eyes == eyes_found:
                    counter += 1
                else:
                    counter = max(0, counter - 2)
        else:
            counter = max(0, counter - 1)

        # Status logic
        if counter > DROWSY_THRESHOLD:
            status_text = "danger"
            if not alarm_on:
                alarm.play(-1)
                alarm_on = True
        elif counter > 3:
            status_text = "warning"
            if alarm_on:
                alarm.stop()
                alarm_on = False
        else:
            status_text = "safe"
            if alarm_on:
                alarm.stop()
                alarm_on = False

        # Update global status thread-safely
        with status_lock:
            system_status = {
                "status": status_text,
                "ear_avg": float(f"{ear_avg:.3f}"),
                "counter": counter,
                "eyes_found": eyes_found,
                "closed_eyes": closed_eyes,
                "alarm_on": alarm_on
            }

        # Visual overlays on the stream for the UI feed
        hud_color = (0, 0, 255) if status_text == "danger" else (0, 165, 255) if status_text == "warning" else (0, 255, 0)
        cv2.putText(frame, f"STATUS: {status_text.upper()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, hud_color, 2)
        cv2.putText(frame, f"Avg EAR: {ear_avg:.3f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if status_text == "danger":
            cv2.putText(frame, "WAKE UP!", (w // 2 - 80, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    with status_lock:
        return jsonify(system_status)

if __name__ == '__main__':
    # Run the Flask app on host 0.0.0.0 and port 5000
    app.run(host='0.0.0.0', port=5000, threaded=True)
