import cv2
import numpy as np
import tensorflow as tf
import pygame
import time

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
# Load Detectors
# -------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

# -------------------------
# Variables
# -------------------------
closed_counter = 0
alarm_on = False
DROWSY_THRESHOLD = 15

print("[INFO] Improved detector starting...")
print("[INFO] Make sure your face is visible and well-lit.")
print("-" * 50)

cap = cv2.VideoCapture(0)

# Warm-up camera
for _ in range(5):
    cap.read()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces first
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

    eyes_detected = 0
    closed_eyes = 0

    for (fx, fy, fw, fh) in faces:
        # Draw face rectangle
        cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 2)

        # Region of interest: upper half of face (where eyes are)
        # Leave some margin to avoid eyebrows being detected as eyes
        roi_y1 = fy + int(fh * 0.15)
        roi_y2 = fy + int(fh * 0.60)
        roi_x1 = fx + int(fw * 0.05)
        roi_x2 = fx + int(fw * 0.95)

        face_roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
        if face_roi.size == 0:
            continue

        # Detect eyes within face ROI
        eyes = eye_cascade.detectMultiScale(
            face_roi,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(fw // 12, fw // 12),
            maxSize=(fw // 4, fw // 4)
        )

        # Process up to 2 eyes
        for (ex, ey, ew, eh) in eyes[:2]:
            # Convert to absolute frame coordinates
            x = roi_x1 + ex
            y = roi_y1 + ey
            w = ew
            h = eh

            # Sanity check: aspect ratio should be roughly eye-like (wider than tall)
            aspect = w / max(h, 1)
            if aspect < 1.5 or aspect > 4.0:
                continue

            eyes_detected += 1

            # Extract and preprocess eye
            eye_gray = gray[y:y+h, x:x+w]
            eye_resized = cv2.resize(eye_gray, (24, 24))
            eye_norm = eye_resized / 255.0
            eye_input = eye_norm.reshape(1, 24, 24, 1)

            prediction = model.predict(eye_input, verbose=0)
            pred_val = float(prediction[0][0])

            # Prediction < 0.5 means closed eye in this model
            is_closed = pred_val < 0.5
            if is_closed:
                closed_eyes += 1

            # Color: red = closed, green = open
            color = (0, 0, 255) if is_closed else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(
                frame,
                f"{pred_val:.2f}",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )

    # Drowsiness logic: require BOTH detected eyes to be closed, or a single eye closed
    if eyes_detected > 0 and closed_eyes >= max(1, eyes_detected // 2):
        closed_counter += 1
    elif eyes_detected > 0:
        closed_counter = max(0, closed_counter - 2)
    else:
        # No eyes detected — don't penalize, but don't reward either
        closed_counter = max(0, closed_counter - 1)

    # Trigger alarm
    if closed_counter > DROWSY_THRESHOLD:
        cv2.putText(
            frame,
            "DROWSY! WAKE UP!",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )
        if not alarm_on:
            alarm.play(-1)
            alarm_on = True
    else:
        if alarm_on:
            alarm.stop()
            alarm_on = False

    # HUD
    status = "WAKE UP!" if closed_counter > DROWSY_THRESHOLD else "AWAKE"
    status_color = (0, 0, 255) if closed_counter > DROWSY_THRESHOLD else (0, 255, 0)

    cv2.putText(frame, f"Faces: {len(faces)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Eyes: {eyes_detected}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Counter: {closed_counter}/{DROWSY_THRESHOLD}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, status, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Detection ended.")
