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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in eyes[:2]:  # take first 2 eyes
        eye = gray[y:y+h, x:x+w]
        eye = cv2.resize(eye, (24, 24))
        eye = eye / 255.0
        eye = eye.reshape(1, 24, 24, 1)

        prediction = model.predict(eye, verbose=0)

        if prediction < 0.5:  # closed eye
            counter += 1

            if counter > 15:
                cv2.putText(frame, "DROWSY!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                if not alarm_on:
                    alarm.play(-1)
                    alarm_on = True
        else:
            counter = 0
            if alarm_on:
                alarm.stop()
                alarm_on = False

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()