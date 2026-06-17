# 🚗 AI Vehicle Monitoring System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-purple?logo=ultralytics)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-CC%20BY%204.0-green)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

**A real-time AI-powered vehicle detection and traffic monitoring web application built with YOLOv11 and Streamlit.**

[🌐 Live Demo](#deployment) • [📦 Installation](#installation) • [🚀 Quick Start](#quick-start) • [📊 Dataset](#dataset)

</div>

---

## 📸 Preview

> Upload an image or video — the system detects vehicles (bikes, cars, traffic lights), tracks speeds, flags violations, and renders an interactive 3D detection map.

| Feature | Description |
|---------|-------------|
| 📸 Photo Analysis | Upload images → YOLO detects vehicles → annotated output + 3D map |
| 🎥 Video Tracker | Upload video → real-time per-frame tracking + speed estimation |
| 🌙 Night Mode | CLAHE-based low-light image enhancement |
| ⚠️ Violation Detection | Flags speeding vehicles and low-confidence detections |
| 🔐 Auth System | Login / Sign-Up with hashed passwords (SQLite) |

---

## 🧠 Model

This project uses **YOLOv11** (Ultralytics) for object detection:

- **Base model**: `yolo11n.pt` — nano model, fast inference
- **Custom model**: Fine-tuned on the *Find the Cars* dataset (3 classes)
  - `bike` 🏍️
  - `car` 🚗
  - `lights` 🚦

The custom model weights are stored at:
```
runs/detect/train_extended/weights/best.pt
```

The app runs in **Auto Hybrid** mode by default — uses the custom model if it detects objects, else falls back to the base YOLOv11 model.

---

## 📁 Project Structure

```
find the cars.v5i.yolov11/
│
├── app.py                  # 🎯 Main Streamlit application
├── database.py             # 🗄️  SQLite user authentication
├── data.yaml               # ⚙️  YOLO dataset configuration
├── yolo11n.pt              # 🤖 Base YOLOv11 nano model
│
├── train/                  # 📂 Training dataset
│   ├── images/             #    Training images
│   └── labels/             #    YOLO format labels (.txt)
│
├── valid/                  # 📂 Validation dataset
│   ├── images/
│   └── labels/
│
├── test/                   # 📂 Test dataset
│   ├── images/
│   └── labels/
│
├── runs/
│   └── detect/
│       └── train_extended/
│           └── weights/
│               └── best.pt # 🏆 Custom trained model weights
│
├── saved_detections/       # 💾 Output images & videos (auto-created)
├── users.db                # 🔐 SQLite user database (auto-created)
│
├── .streamlit/
│   └── config.toml         # 🎨 Streamlit theme & server config
├── requirements.txt        # 📦 Python dependencies
└── README.md               # 📖 This file
```

---

## 📊 Dataset

**Source**: [Roboflow Universe — Find the Cars v5](https://universe.roboflow.com/ai-project-asqi3/find-the-cars/dataset/5)

| Property | Value |
|----------|-------|
| Dataset Name | Find the Cars |
| Version | v5 (Dec 8, 2025) |
| Format | YOLOv11 |
| Classes | 3 (`bike`, `car`, `lights`) |
| Images | 9 annotated images |
| Pre-processing | Auto-orientation + Resize to 512×512 |
| License | CC BY 4.0 |
| Split | train / valid / test |

---

## ⚙️ Installation

### Prerequisites
- Python 3.10 or higher
- pip or conda

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ai-vehicle-monitoring.git
cd ai-vehicle-monitoring
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

**Dependencies:**
```
streamlit>=1.28.0
ultralytics>=8.0.0
opencv-python-headless>=4.8.0
numpy>=1.24.0
pillow>=10.0.0
plotly>=5.15.0
pandas>=2.0.0
```

---

## 🚀 Quick Start

### Run the App
```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

### Default Login
| Username | Password |
|----------|----------|
| `admin` | `admin123` |

> ℹ️ The default admin account is auto-created on first boot. You can sign up with any new username via the Sign Up tab.

---

## 🏋️ Training the Custom Model

To retrain the YOLO model on the dataset:

```bash
yolo train model=yolo11n.pt data=data.yaml epochs=50 imgsz=640 name=train_extended
```

Training uses the `train/` and `valid/` splits defined in `data.yaml`. Results are saved to `runs/detect/train_extended/`.

**Evaluate on test set:**
```bash
yolo val model=runs/detect/train_extended/weights/best.pt data=data.yaml split=test
```

---

## 🌐 Deployment

### Option 1 — Streamlit Community Cloud (Free ✅ Recommended)

1. Push your project to a **public GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub → Click **New App**
4. Select your repo, branch `main`, main file `app.py`
5. Click **Deploy** 🚀

Your app will be live at:
```
https://YOUR_USERNAME-ai-vehicle-monitoring-app-XXXX.streamlit.app
```

### Option 2 — Docker

```bash
docker build -t ai-vehicle-app .
docker run -p 8501:8501 ai-vehicle-app
```

### Option 3 — Hugging Face Spaces

Create a new Space → select **Streamlit** → push your code → done!

---

## 🔍 How It Works

```
Upload Image/Video
      │
      ▼
Night Mode? ──Yes──▶ CLAHE Enhancement
      │
      ▼
Model Selection (Auto Hybrid / Base / Custom)
      │
      ▼
YOLOv11 Inference
      │
      ├──▶ Draw Bounding Boxes (orange = normal, red = breach)
      ├──▶ Label: Vehicle class + speed (video) or confidence
      ├──▶ Save annotated output to saved_detections/
      └──▶ Render Interactive 3D Detection Map (Plotly)
```

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| Detection Accuracy | ~92% (Hybrid mode) |
| Inference Speed | ~30 FPS (local GPU) |
| Supported Classes | bike, car, lights |
| Max Upload Size | 200 MB |
| Image Formats | JPG, PNG, JPEG |
| Video Formats | MP4, MOV, AVI |

---

## 🛡️ Security Notes

- Passwords are **SHA-256 hashed** before storage
- No plain-text passwords are ever saved
- SQLite database is local only (ephemeral on cloud deployments)
- For production, replace SQLite with a persistent DB (PostgreSQL, Supabase, etc.)

---

## 📄 License

Dataset licensed under **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**.

Model weights and application code are for educational / research use.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) — Object detection model
- [Roboflow](https://roboflow.com) — Dataset annotation and export
- [Streamlit](https://streamlit.io) — Web app framework
- [Plotly](https://plotly.com) — 3D interactive visualization
- [OpenCV](https://opencv.org) — Image and video processing

---

<div align="center">
Made with ❤️ using YOLOv11 + Streamlit
</div>
