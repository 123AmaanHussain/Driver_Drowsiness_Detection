<div align="center">

# 🚗 Driver Drowsiness Detection System

### Real-time drowsiness monitoring powered by MediaPipe, TensorFlow & React

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.35-0097A7?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org)
[![Flask](https://img.shields.io/badge/Flask-3.1-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Vite](https://img.shields.io/badge/Vite-8-646CFF?style=for-the-badge&logo=vite&logoColor=white)](https://vitejs.dev)

</div>

---

## 📋 Overview

The **Driver Drowsiness Detection System** is a real-time computer vision application that monitors a driver's alertness using a standard webcam. It uses **Google MediaPipe Face Mesh** to extract 468 facial landmarks per frame and computes the **Eye Aspect Ratio (EAR)** to determine whether eyes are open or closed. A custom-trained **TensorFlow/Keras** CNN model provides a secondary classification layer for improved accuracy.

When drowsiness is detected, the system:
- Triggers an **audio alarm** (`alarm.wav`) to alert the driver
- Updates the **live React dashboard** with warning/danger state
- Logs the event with timestamp and EAR value in the **Alerts** panel

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎯 **MediaPipe Face Mesh** | Tracks 468 facial landmarks at 30fps for precise eye region extraction |
| 📐 **EAR Algorithm** | Eye Aspect Ratio computed from 6 key landmark points per eye |
| 🤖 **TF Keras Model** | Custom CNN model classifying 24×24 eye patches as open/closed |
| 📡 **MJPEG Stream** | Annotated live webcam stream served directly to the browser |
| 📊 **Live Charts** | Real-time EAR history and drowsy-frame counter charts via Recharts |
| 🔔 **Alert Logging** | Full session log of all status transitions with timestamps |
| ⚙️ **Settings Panel** | Configurable EAR threshold, frame threshold, alarm toggle, and more |
| 🌐 **REST API** | Flask API exposing `/status` (JSON) and `/video_feed` (MJPEG) endpoints |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        React Frontend                           │
│   ┌────────────┐  ┌─────────────┐  ┌──────────┐  ┌──────────┐ │
│   │ Dashboard  │  │ Camera Feed │  │  Alerts  │  │ Settings │ │
│   └─────┬──────┘  └──────┬──────┘  └────┬─────┘  └──────────┘ │
│         │  GET /status   │              │                       │
│         │◄───────────────┘              │                       │
│         │  GET /video_feed              │ alert event log       │
│         │◄──────────────────────────────┘                       │
└─────────┼───────────────────────────────────────────────────────┘
          │ HTTP (localhost:5000)
┌─────────▼───────────────────────────────────────────────────────┐
│                     Flask Backend (server.py)                   │
│                                                                 │
│   ┌───────────┐    ┌────────────────┐    ┌──────────────────┐  │
│   │  Webcam   │───►│ MediaPipe Face │───►│  EAR Calculation │  │
│   │ OpenCV    │    │    Mesh (468   │    │  (6 points/eye)  │  │
│   └───────────┘    │   landmarks)  │    └────────┬─────────┘  │
│                    └────────────────┘             │            │
│                                        ┌──────────▼─────────┐  │
│                                        │  Keras CNN Model   │  │
│                                        │  (24×24 eye patch) │  │
│                                        └──────────┬─────────┘  │
│                                                   │            │
│                              ┌────────────────────▼──────────┐ │
│                              │   Drowsiness Logic            │ │
│                              │   EAR < 0.20 for N frames     │ │
│                              │   → alarm + status update     │ │
│                              └───────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔬 How the Detection Works

### 1. Face Landmark Detection
MediaPipe Face Landmarker processes each webcam frame and returns a set of 468 3D facial landmarks. From these, we extract 6 specific points around each eye.

### 2. Eye Aspect Ratio (EAR)
The EAR is computed using the Soukupova & Cech (2016) formula:

```
        ||p2 - p6|| + ||p3 - p5||
EAR  =  ──────────────────────────
               2 × ||p1 - p4||
```

Where `p1–p6` are the six eye landmark points. A fully open eye has an EAR ≈ 0.30. A closed eye drops below **0.20**.

### 3. Drowsiness Trigger
```
if EAR < 0.20:
    counter += 1          # increment drowsy frame counter
else:
    counter -= 2          # decay when eyes reopen

if counter >= 10:         # ~330ms at 30fps
    trigger_alarm()
    status = "DANGER"
elif counter >= 3:
    status = "WARNING"
else:
    status = "SAFE"
```

### 4. Secondary ML Model
A custom-trained Keras CNN takes a 24×24 grayscale eye patch as input and outputs a probability of the eye being closed. This serves as a reference signal alongside the primary EAR logic.

---

## 📁 Project Structure

```
Driver_Drowsiness/
│
├── server.py                  # Flask backend — video stream + status API
├── detect_mediapipe.py        # Standalone MediaPipe detection script
├── detect.py                  # Legacy Haar cascade script (deprecated)
├── detect_improved.py         # Haar + face detection improvement
├── detect_debug.py            # Debug script with raw model output
├── calibrate.py               # Model calibration / raw EAR viewer
│
├── models/
│   └── drowsiness_model.keras # Trained TensorFlow/Keras CNN model
│
├── face_landmarker.task       # MediaPipe face landmark model (auto-downloaded)
├── alarm.wav                  # Alarm audio file
│
├── frontend/                  # React + Vite frontend
│   ├── src/
│   │   ├── App.jsx            # Main app — routing, pages, live data hooks
│   │   └── index.css          # Global styles (dark mode, CSS variables)
│   ├── package.json
│   └── vite.config.js
│
├── README.md
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11** (TensorFlow does not support 3.13)
- **Node.js 18+** and npm
- A working **webcam**

### 1. Clone the Repository

```bash
git clone https://github.com/123AmaanHussain/Driver_Drowsiness_Detection.git
cd Driver_Drowsiness_Detection
```

### 2. Set Up Python Virtual Environment

```bash
# Windows (PowerShell)
python3.11 -m venv venv
.\venv\Scripts\Activate.ps1

# Windows (CMD)
python3.11 -m venv venv
venv\Scripts\activate.bat

# macOS / Linux
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install opencv-python tensorflow pygame mediapipe==0.10.35 protobuf==7.34.1 flask flask-cors
```

> **Note:** `mediapipe==0.10.35` and `protobuf==7.34.1` are pinned for compatibility with TensorFlow 2.x.

### 4. Add Required Files (Not in Repo)

The following files are excluded from the repo due to size. You must provide them:

| File | Description |
|---|---|
| `models/drowsiness_model.keras` | Your trained Keras model |
| `alarm.wav` | Alarm audio file |
| `face_landmarker.task` | Auto-downloads on first run |

### 5. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

---

## ▶️ Running the System

You need **two terminals** running simultaneously.

### Terminal 1 — Start the Flask Backend

```bash
# Activate venv first
.\venv\Scripts\Activate.ps1     # Windows PowerShell
# or
source venv/Scripts/activate    # Windows Git Bash

python server.py
```

The backend will start at `http://localhost:5000`.  
On first run, `face_landmarker.task` (~3MB) will be auto-downloaded.

### Terminal 2 — Start the React Frontend

```bash
cd frontend
npm run dev
```

The frontend will start at `http://localhost:5173`.

Open **http://localhost:5173** in your browser.

---

## 🌐 API Reference

The Flask backend exposes two endpoints:

### `GET /video_feed`
Returns a **multipart MJPEG stream** of the webcam feed with MediaPipe face mesh annotations, eye bounding boxes (green = open, red = closed), and EAR overlays.

```
Content-Type: multipart/x-mixed-replace; boundary=frame
```

### `GET /status`
Returns the current detection state as JSON.

```json
{
  "status":      "safe",   // "safe" | "warning" | "danger"
  "ear_avg":     0.312,    // Average Eye Aspect Ratio (both eyes)
  "counter":     0,        // Consecutive drowsy frames
  "eyes_found":  2,        // Number of eyes detected (0–2)
  "closed_eyes": 0,        // Eyes currently flagged as closed
  "alarm_on":    false     // Whether the audio alarm is active
}
```

---

## 🖥️ Dashboard Pages

| Page | Description |
|---|---|
| **Dashboard** | Live camera feed + EAR area chart + drowsy counter bar chart + status panel + 4 stat cards |
| **Camera Feed** | Full-resolution annotated stream with side panel showing all raw metrics and detection pipeline explanation |
| **Alerts** | Filterable event log of all status transitions (Danger / Warning / Recovery) with timestamps and EAR values |
| **Settings** | Configurable EAR threshold, frame threshold, alarm toggle, warning stage toggle, and system info |

---

## ⚙️ Configuration

Key parameters in `server.py` and `detect_mediapipe.py`:

| Parameter | Default | Description |
|---|---|---|
| `EAR_THRESHOLD` | `0.20` | Eye considered closed below this value |
| `DROWSY_THRESHOLD` | `10` | Frames before alarm triggers |
| `Counter decay` | `-2` | Frames subtracted per open-eye frame |
| `num_faces` | `1` | Max faces tracked by MediaPipe |
| `Flask port` | `5000` | Backend API port |

---

## 🛠️ Troubleshooting

| Issue | Fix |
|---|---|
| `ImportError: DLL load failed` | Install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) |
| `ValueError: Input timestamp must be monotonically increasing` | Fixed — server now uses `int(time.time() * 1000)` |
| `No module named mediapipe` | Run `pip install mediapipe==0.10.35` inside your `venv` |
| False alarms with eyes open | Lower EAR threshold: change `0.20` → `0.16` in `server.py:125` |
| `Python 3.13` not working | TensorFlow requires **Python 3.11** — install from [python.org](https://python.org) |
| Camera not found | Change `VideoCapture(0)` → `VideoCapture(1)` if you have multiple cameras |

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| **Face Detection** | MediaPipe Face Landmarker (Tasks API) |
| **Eye Classification** | Custom TensorFlow/Keras CNN |
| **EAR Algorithm** | Soukupova & Cech (2016) |
| **Video Processing** | OpenCV |
| **Backend API** | Flask + Flask-CORS |
| **Audio Alarm** | Pygame |
| **Frontend** | React 18 + Vite |
| **Charts** | Recharts |
| **Styling** | Vanilla CSS with CSS Variables |

---

## 📚 References

- Soukupova, T. & Cech, J. (2016). *Real-Time Eye Blink Detection using Facial Landmarks*. CVWW.
- [MediaPipe Face Landmarker — Google](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)
- [TensorFlow / Keras Documentation](https://www.tensorflow.org/)
- [OpenCV Documentation](https://docs.opencv.org/)

---

## 📄 License

This project is intended for **educational and academic use only**.  
Ensure safe testing — **do not use while operating a vehicle**.

---

<div align="center">
Made with ❤️ by <a href="https://github.com/123AmaanHussain">Amaan Hussain</a>
</div>
