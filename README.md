# NHAI Drishti 👁️
### Offline AI-Powered Face Authentication & Attendance System

> Hackathon Submission — Smart India Hackathon / NHAI  
> **Fully offline. No cloud required to capture attendance.**

---

## Overview

NHAI Drishti is an edge-deployed biometric attendance system that:

- **Detects** faces in real time using MediaPipe Face Mesh (468 landmarks)
- **Verifies** liveness with randomised challenges (blink, smile, head-turn left/right)
- **Recognises** enrolled employees via MobileFaceNet 192-D cosine similarity
- **Logs** every attendance event to a local SQLite database instantly
- **Syncs** records to AWS when connectivity returns (optional)

All inference runs **100% on-device** — no internet connection is required during operation.

---

## System Requirements

| Component | Version |
|-----------|---------|
| Python | 3.11 |
| OS | Windows 11 (also Linux-compatible) |
| TensorFlow | 2.15.0 |
| MediaPipe | 0.10.14 |
| OpenCV | 4.11.0.86 |
| Camera | Any USB/built-in webcam |

---

## Installation

```bash
# 1. Clone / extract the project
cd submission/

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the System

### Start face authentication + attendance
```bash
python main.py
```

The system will:
1. Load MobileFaceNet from `ai/models/mobilefacenet.tflite`
2. Open the default webcam (index 0)
3. Begin real-time face detection + liveness verification
4. Log attendance to `data/drishti.db`

Press `Ctrl+C` to stop gracefully.

---

## Enrollment

To enroll a new employee, use the `EnrollmentManager` programmatically:

```python
from ai.storage.database_manager import DatabaseManager
from ai.detector.face_detector import FaceDetector
from ai.embedding.mobilefacenet import MobileFaceNet
from ai.enrollment.enrollment_manager import EnrollmentManager
import cv2

db = DatabaseManager("data/drishti.db")
db.initialize()

detector = FaceDetector()
model = MobileFaceNet(model_dir="ai/models")
manager = EnrollmentManager(db_manager=db, face_detector=detector, model=model)

camera = cv2.VideoCapture(0)
report = manager.enroll(
    camera=camera,
    employee_code="NHAI-001",
    name="Rajesh Kumar",
    department="Operations",
)
camera.release()

print(f"Enrolled: {report.success}")
print(f"Frames used: {report.frames_used}/{report.frames_captured}")
```

---

## Architecture

```
main.py
  ├── DrishtiApplication         (orchestrator, DI container)
  ├── RecognitionPipeline        (detect → liveness → embed → recognise)
  ├── AttendancePipeline         (cooldown → dedup → mark → sync)
  ├── CameraProcessor            (frame loop, display overlay)
  └── SyncWorker                 (background AWS upload thread)

ai/
  ├── detector/face_detector.py      MediaPipe Face Mesh + alignment
  ├── embedding/mobilefacenet.py     TFLite inference wrapper (192-D)
  ├── liveness/
  │   ├── blink.py                   EAR-based blink detection
  │   ├── smile.py                   MAR-based smile detection
  │   ├── head_turn.py               solvePnP yaw-based head turn
  │   └── challenge_manager.py       Random challenge orchestrator
  ├── storage/database_manager.py    SQLite (users + attendance + sync)
  ├── enrollment/enrollment_manager.py  Multi-frame enrollment pipeline
  ├── attendance/attendance_service.py  Attendance logic + dedup
  ├── recognition/similarity.py     Cosine similarity utilities
  └── sync/sync_queue.py            Offline-first sync queue
```

---

## Configuration

Edit `SystemConfig` in `main.py` or pass values at startup:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `camera_index` | `0` | Webcam index |
| `recognition_threshold` | `0.65` | Minimum cosine similarity to accept |
| `liveness_enabled` | `True` | Enable/disable challenge verification |
| `attendance_cooldown_seconds` | `300` | Re-mark interval per employee |
| `model_dir` | `ai/models` | Path to .tflite model |
| `db_path` | `data/drishti.db` | SQLite database path |

---

## Model

| Property | Value |
|----------|-------|
| Architecture | MobileFaceNet |
| Input | 112 × 112 × 3 (float32, normalised to [-1, 1]) |
| Output | 192-dimensional L2-normalised embedding |
| File size | ~5 MB |
| Inference backend | TensorFlow Lite |
| Similarity metric | Cosine similarity |
| Match threshold | 0.65 (configurable) |

---

## Liveness Detection

Four randomised challenges prevent photo/video spoofing:

| Challenge | Method | Metric |
|-----------|--------|--------|
| Blink | Eye Aspect Ratio (EAR) | EAR < 0.20 for 2–15 frames |
| Smile | Mouth Aspect Ratio (MAR) | MAR > 1.4× neutral baseline |
| Head Left | solvePnP yaw | Yaw < −20° sustained 8 frames |
| Head Right | solvePnP yaw | Yaw > +20° sustained 8 frames |

Each challenge has an 8-second timeout. A new challenge is randomly selected per session.

---

## Database Schema

```sql
-- Enrolled users
CREATE TABLE users (
    user_id       TEXT PRIMARY KEY,
    employee_code TEXT UNIQUE NOT NULL,
    name          TEXT NOT NULL,
    department    TEXT NOT NULL,
    embedding     TEXT NOT NULL,   -- JSON float array
    last_seen     TEXT,
    created_at    TEXT NOT NULL
);

-- Attendance log
CREATE TABLE attendance_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT    NOT NULL,
    confidence      REAL    NOT NULL,
    challenge_type  TEXT    NOT NULL,
    liveness_passed INTEGER NOT NULL,
    timestamp       TEXT    NOT NULL,
    synced          INTEGER NOT NULL DEFAULT 0
);

-- AWS sync queue
CREATE TABLE sync_queue (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    retry_count INTEGER NOT NULL DEFAULT 0
);
```

---

## AWS Sync (Optional)

Set environment variables to enable cloud upload:

```bash
set AWS_REGION=ap-south-1
set AWS_API_ENDPOINT=https://<id>.execute-api.ap-south-1.amazonaws.com/prod
set AWS_ACCESS_KEY_ID=...
set AWS_SECRET_ACCESS_KEY=...
```

Without these variables, attendance is stored locally in SQLite and synced when connectivity is next available.

---

## Testing Checklist

### Startup validation
```bash
python main.py
# Expected: "NHAI Drishti — System LIVE" in logs
# Expected: no ImportError, no TypeError, no AttributeError
```

### Enrollment test
```python
# Run enrollment script above with a real camera
# Expected: EnrollmentReport.success == True
# Expected: frames_used >= 3
```

### Recognition test
```bash
# After enrolling, run main.py
# Stand in front of webcam
# Expected: "Face recognised — subject=<id> confidence=X.XX"
# Expected: "Attendance marked" in logs
```

### Liveness test
```bash
# Run main.py with liveness_enabled=True (default)
# Follow on-screen challenge (blink / smile / turn head)
# Expected: Challenge passes → recognition proceeds
# Expected: Holding a photo → challenge never passes
```

---

## Project Team

| Role | Name |
|------|------|
| Lead Engineer | — |
| AI/ML Engineer | — |
| Backend Engineer | — |

---

## License

Apache 2.0 — see model metadata for model-specific license.
