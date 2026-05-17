import cv2
import numpy as np
import tensorflow as tf
import pygame

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
# Load Eye Detector
# -------------------------
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# -------------------------
# Variables
# -------------------------
counter = 0
alarm_on = False

# -------------------------
# Start Camera
# -------------------------
cap = cv2.VideoCapture(0)

print("[INFO] Starting debug detection. Press 'q' to quit.")
print("[INFO] If no eyes are detected, try better lighting or move closer to camera.")
print("-" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

    # Show number of eyes detected on frame
    cv2.putText(frame, f"Eyes found: {len(eyes)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Counter: {counter}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    for i, (x, y, w, h) in enumerate(eyes[:2]):
        eye = gray[y:y+h, x:x+w]
        eye = cv2.resize(eye, (24, 24))
        eye = eye / 255.0
        eye = eye.reshape(1, 24, 24, 1)

        prediction = model.predict(eye, verbose=0)
        pred_val = float(prediction[0][0])

        # Label on frame
        label = f"Eye {i+1}: {pred_val:.3f}"
        color = (0, 0, 255) if pred_val < 0.5 else (0, 255, 0)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        print(f"[DEBUG] Eye {i+1} prediction: {pred_val:.4f} | Counter: {counter}")

        if pred_val < 0.5:
            counter += 1
            if counter > 15:
                cv2.putText(frame, "DROWSY!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                if not alarm_on:
                    alarm.play(-1)
                    alarm_on = True
        else:
            counter = max(0, counter - 1)
            if counter == 0 and alarm_on:
                alarm.stop()
                alarm_on = False

    if len(eyes) == 0:
        print("[DEBUG] No eyes detected in this frame")

    cv2.imshow("Driver Drowsiness Detection - DEBUG", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Debug session ended.")
