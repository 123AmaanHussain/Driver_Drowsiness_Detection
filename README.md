# Driver Drowsiness Detection System

Real-time drowsiness detection using webcam, MediaPipe face landmarks, and a custom TensorFlow model.

---

## Features

- **Real-time webcam-based eye detection** — no additional hardware required
- **MediaPipe Face Mesh** — robust face and eye tracking even at sitting distance
- **EAR (Eye Aspect Ratio) + Custom ML Model** — hybrid detection for reliability
- **Audio alarm** — plays `alarm.wav` when drowsiness is detected
- **Visual feedback** — green (open) / red (closed) eye boxes with live metrics

---

## Project Structure

```
Driver_drowsiness/
├── alarm.wav                    # Alarm sound file
├── detect.py                    # Original script (Haar cascade — deprecated)
├── detect_improved.py           # Face-first Haar approach
├── detect_mediapipe.py          # **Recommended** — MediaPipe + EAR detection
├── detect_debug.py              # Debug script with raw predictions
├── calibrate.py                 # Model calibration / raw value viewer
├── models/
│   └── drowsiness_model.keras   # Your custom TensorFlow model
├── face_landmarker.task         # Auto-downloaded MediaPipe model
├── venv/                        # Python virtual environment
└── README.md                    # This file
```

---

## Requirements

- **Python 3.11** (Python 3.13 is NOT supported by TensorFlow)
- Webcam
- Windows, Linux, or macOS

---

## Setup

### 1. Install Python 3.11

Download and install Python 3.11 from [python.org](https://www.python.org/downloads/release/python-31111/).

**Important:** Check **"Add Python to PATH"** during installation.

### 2. Create Virtual Environment

```bash
python3.11 -m venv venv
```

On Windows (Git Bash):
```bash
python3.11 -m venv venv
```

### 3. Activate Virtual Environment

**Windows (Git Bash):**
```bash
source venv/Scripts/activate
```

**Windows (CMD):**
```cmd
venv\Scripts\activate
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install opencv-python tensorflow pygame mediapipe==0.10.35 protobuf==7.34.1
```

**Note:** `mediapipe==0.10.35` and `protobuf==7.34.1` are pinned for compatibility with TensorFlow 2.21.

---

## Running the Project

### Recommended: MediaPipe Version (Best Accuracy)

```bash
python detect_mediapipe.py
```

This uses MediaPipe Face Landmarker for precise eye detection and EAR-based drowsiness logic.

### Legacy: Original Version (Basic)

```bash
python detect.py
```

Uses Haar cascade eye detection. Less reliable at distance, prone to false positives.

### Debug / Calibration

```bash
python detect_debug.py      # Shows raw model predictions per frame
python calibrate.py        # Shows model values for open vs closed eyes
```

---

## How It Works

1. **Face Detection** — MediaPipe detects your face and 468 facial landmarks
2. **Eye Extraction** — Eye regions are cropped from the face mesh
3. **EAR Calculation** — Eye Aspect Ratio measures how open/closed each eye is
4. **Drowsiness Logic** — If **both** eyes have EAR < 0.20 for ~15-20 consecutive frames, the alarm triggers
5. **Alarm** — `alarm.wav` plays in a loop until eyes reopen

---

## On-Screen Display

| Element | Meaning |
|---|---|
| **Green eye boxes** | Eyes are open (safe) |
| **Red eye boxes** | Eyes are closed / narrowed (counting up) |
| **E: 0.xx** | Eye Aspect Ratio value |
| **Counter** | Frames with closed eyes (alarm at threshold) |
| **Avg EAR** | Average eye openness across both eyes |

---

## Known Issues & Fixes

### "ImportError: DLL load failed" (TensorFlow)
**Fix:** Install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)

### "No attribute 'solutions'" (MediaPipe)
**Fix:** Use `mediapipe==0.10.35` with the Tasks API (already done in `detect_mediapipe.py`)

### "HTTP Error 404" (model download)
**Fix:** The `face_landmarker.task` auto-downloads on first run. If it fails, delete it and rerun.

### False alarms with eyes open
**Fix:** The EAR threshold is set to 0.20. If you have naturally small eyes, you may need to adjust this in `detect_mediapipe.py` line 125:
```python
is_closed = ear < 0.20  # Try 0.15 or 0.18
```

### Python 3.13 not working
TensorFlow does not support Python 3.13. You **must** use Python 3.11 or 3.12.

---

## Customizing

| Parameter | Location | Default | Effect |
|---|---|---|---|
| Drowsy threshold | `detect_mediapipe.py:73` | 10 | Frames of closed eyes before alarm |
| EAR threshold | `detect_mediapipe.py:125` | 0.20 | Eye openness cutoff |
| Counter decay | `detect_mediapipe.py:142` | -2 | How fast counter drops when eyes open |
| Webcam index | Any file line ~80 | 0 | `VideoCapture(0)` — change if multiple cameras |

---

## Quitting

Press **`q`** in the camera window to stop the program.

---

## Credits

- Eye Aspect Ratio (EAR) algorithm: Soukupova & Cech, 2016
- MediaPipe by Google
- TensorFlow by Google
- OpenCV

---

## License

For educational use. Ensure safe testing — do not operate vehicles while testing this system.
