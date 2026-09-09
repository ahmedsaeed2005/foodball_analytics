# ⚽ Football Analytics

A Computer Vision system for **football player detection, multi-object tracking, and player speed estimation** using a custom-trained YOLO model.

The model used in this project was **trained by me using a football dataset from Roboflow**, then integrated with YOLO tracking and OpenCV to analyze football videos.

---

## 🚀 Project Overview

The goal of this project is to build a practical Football Analytics pipeline that can detect and track players across video frames while estimating their movement speed.

The system processes a football video and provides:

* ⚽ Player Detection
* 🆔 Unique Player IDs
* 🎯 Multi-Object Tracking
* 📍 Player Position Tracking
* 🏃 Player Speed Estimation
* 📉 Speed Smoothing using Exponential Moving Average (EMA)
* 🎥 Real-Time Video Visualization

---

## 🧠 System Pipeline

```text
Roboflow Dataset
       │
       ▼
Model Training
       │
       ▼
Custom YOLO Model
       │
       ▼
Player Detection
       │
       ▼
BoT-SORT / ByteTrack
       │
       ▼
Player IDs
       │
       ▼
Position Tracking
       │
       ▼
Speed Calculation
       │
       ▼
EMA Smoothing
       │
       ▼
Football Analytics
```

---

## 🤖 Custom YOLO Model

The YOLO model used in this project is **not a pretrained football model**.

I trained the model using a football dataset prepared with **Roboflow**.

The trained weights are:

```text
runs_players_model-5_weights_best.pt
```

The trained model is then loaded using Ultralytics YOLO and used for player detection and tracking.

---

## 🎯 Object Tracking

For tracking detected players across consecutive frames, the project supports:

### BoT-SORT

```text
trackers/botsort.yaml
```

### ByteTrack

```text
trackers/bytetrack.yaml
```

The tracker maintains a unique ID for each detected player, allowing the system to follow the same player throughout the video.

Example:

```text
Player → ID: 1
Player → ID: 2
Player → ID: 3
```

---

## 🏃 Player Speed Estimation

The project estimates player movement speed based on the change in the player's position between consecutive frames.

Instead of using the center of the bounding box, the system uses the **bottom-center point** of the bounding box as an approximation of the player's foot position.

```text
Bounding Box
┌───────────────┐
│               │
│    Player     │
│               │
└───────●───────┘
        ↑
   Bottom Center
```

The distance between the current and previous positions is calculated using Euclidean distance:

```text
distance = √((x₂ - x₁)² + (y₂ - y₁)²)
```

Then the speed is estimated using:

```text
speed = distance / time
```

The current implementation reports the estimated speed in:

```text
pixels / second
```

---

## 📉 Speed Smoothing

Raw frame-by-frame speed can fluctuate because of detection and tracking noise.

To make the displayed speed more stable, the project uses **Exponential Moving Average (EMA)** smoothing.

The smoothing factor is:

```python
ALPHA = 0.3
```

The smoothed speed is calculated as:

```text
Smoothed Speed =
α × Current Speed
+
(1 - α) × Previous Smoothed Speed
```

This helps reduce sudden changes and produces a more stable visualization.

---

## 💻 Technologies

* Python
* OpenCV
* PyTorch
* Ultralytics YOLO
* BoT-SORT
* ByteTrack
* Roboflow
* Computer Vision
* Multi-Object Tracking

---

## 📂 Project Structure

```text
Football-analytics/
│
├── demo/
│   └── football_analytics_demo.mp4
│
├── trackers/
│   ├── botsort.yaml
│   └── bytetrack.yaml
│
├── runs_players_model-5_weights_best.pt
│
├── sped.py
├── track.py
├── tracker.py
│
├── .gitignore
└── README.md
```

---

## 📜 Files Description

### `tracker.py`

Contains the `FootballTracker` class responsible for loading the YOLO model and performing object tracking.

### `track.py`

Runs the football player tracking pipeline and displays the tracking results.

### `sped.py`

Extends the tracking pipeline by calculating and displaying player movement speed.

### `trackers/`

Contains the tracking configurations for:

* BoT-SORT
* ByteTrack

### `runs_players_model-5_weights_best.pt`

Custom-trained YOLO model weights trained on the football dataset.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/ahmedsaeed2005/foodball_analytics.git
```

Move into the project directory:

```bash
cd foodball_analytics
```

Install the required dependencies:

```bash
pip install ultralytics opencv-python torch
```

---

## ▶️ Running the Project

### Player Tracking

Run:

```bash
python track.py
```

The system will:

1. Load the custom YOLO model.
2. Open the football video.
3. Detect players.
4. Track players across frames.
5. Assign unique IDs.
6. Display the tracking results.

---

### Player Speed Estimation

Run:

```bash
python sped.py
```

The system will additionally:

1. Track each player.
2. Calculate the player's movement between frames.
3. Estimate speed in pixels/second.
4. Apply EMA smoothing.
5. Display the player ID and estimated speed.

Example:

```text
ID: 7
124.5 px/s
```

---

## 🎥 Demo

A demonstration video is included in the repository showing the football analytics system running on a football video.

The demo demonstrates:

* Player detection
* Player tracking
* Unique player IDs
* Player position tracking
* Speed estimation

---

## 📊 Current Capabilities

| Feature                 | Status |
| ----------------------- | ------ |
| Player Detection        | ✅      |
| Custom YOLO Training    | ✅      |
| Multi-Object Tracking   | ✅      |
| Player IDs              | ✅      |
| BoT-SORT                | ✅      |
| ByteTrack               | ✅      |
| Position Tracking       | ✅      |
| Speed Estimation        | ✅      |
| EMA Smoothing           | ✅      |
| Real-Time Visualization | ✅      |

---

## 🔬 Future Improvements

The project can be extended to include more advanced football analytics such as:

* ⚽ Ball Detection & Tracking
* 🏃 Real-World Speed in km/h
* 📊 Player Statistics
* 🔵 Team Classification
* 🟢 Possession Analysis
* 🎯 Shots Detection
* 🥅 Goal Detection
* 📍 Heatmaps
* 🗺️ Football Pitch Mapping
* 📈 Player Performance Analytics
* 📐 Homography-based field coordinates

---

## 🎓 Learning Outcomes

Through this project, I gained practical experience with:

* Training custom Computer Vision models.
* Preparing and using datasets with Roboflow.
* Object detection using YOLO.
* Multi-object tracking.
* Player identity tracking across frames.
* Coordinate-based motion analysis.
* Speed estimation from video.
* Noise reduction using EMA.
* Building practical Computer Vision applications.

---

## 👨‍💻 Author

**Ahmed Saeed**

Computer Science Student | AI & Computer Vision Enthusiast

---

## ⭐ If you find this project useful

Feel free to explore the repository, experiment with the tracking configurations, and build your own football analytics applications.

**Built with Python, YOLO, OpenCV & Computer Vision.** ⚽🤖
