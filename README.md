# Sign Language Tutor Game 🤟

An Intelligent Adaptive Agent for Learning American Sign Language (ASL)

**Course:** CEN 352 – Artificial Intelligence  
**Authors:** Dionis Beçi, Durim Çeka  
**Date:** December 2024

---

## 🎯 Project Overview

This is an **Intelligent Tutoring System (ITS)** that teaches ASL alphabets using:
1. **Deep Learning** - Neural network trained on 87,000 Kaggle images
2. **Computer Vision** - Real-time hand tracking via MediaPipe
3. **Adaptive Learning** - Tracks mistakes and generates targeted practice words

---

## 🚀 Quick Start (For Professor)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Game
```bash
python game_robust.py
```

### 3. Controls
- **Click a letter** to practice that specific letter
- **Click "RANDOM MODE"** for continuous practice
- **Press H** to toggle Hybrid Mode (Math + AI)
- **Press Q** to quit

---

## 📦 Requirements

```
opencv-python
mediapipe
torch
torchvision
numpy
```

**Python Version:** 3.10+  
**Webcam:** Required

---

## 🧠 AI Architecture

### Technique 1: Deep Learning (Neural Network)
- **Model:** Custom PyTorch network (`SignLanguageModel`)
- **Training Data:** Kaggle ASL Dataset (87,000 images)
- **Input:** 21 hand landmarks (42 features: x, y per point)
- **Output:** Probability distribution over 27 classes (A-Z + Space)

### Technique 2: Rule-Based Reasoning (ITS)
- **Mistake Matrix:** Tracks user error rates per letter
- **Spaced Repetition:** Prioritizes weak letters in word generation
- **Constraint Satisfaction Search:** Finds words containing problem letters

---

## 📁 Project Structure

```
SignLanguageProject/
├── game_robust.py          # Main game application
├── its_engine.py           # Intelligent Tutoring System
├── train_from_images.py    # Training script (optional)
├── robust_model_gpu.pth    # Trained neural network weights
├── robust_classes.pkl      # Class labels (A-Z, Space)
├── canonical_landmarks.pkl # Visual guide data
├── requirements.txt        # Dependencies
└── README.md               # This file
```

---

## 🎮 Features

| Feature | Description |
|---------|-------------|
| **Real-time Recognition** | Webcam hand tracking with MediaPipe |
| **Visual Guide** | Ghost hand shows correct pose |
| **Hold Timer** | 2-second hold to confirm letter |
| **Score System** | Gamified learning experience |
| **Adaptive Words** | ITS generates words with problem letters |
| **Debug Mode** | Shows top-3 predictions with confidence |

---

## 📊 PEAS Framework

- **Performance:** Classification accuracy, user retention rate, game score
- **Environment:** Partially observable (webcam FOV), dynamic, continuous
- **Actuators:** GUI displaying targets, feedback, adaptive challenges
- **Sensors:** Webcam

---

## 📚 References

1. Pigou, L., et al. (2018). "Beyond Temporal Pooling: Recurrence and Temporal Convolutions for Gesture Recognition."
2. LeCun, Y., Bengio, Y., & Hinton, G. (2015). "Deep learning." Nature, 521.
3. Kaggle ASL Dataset: https://www.kaggle.com/datasets/grassknoted/asl-alphabet

---

## ⚠️ Troubleshooting

**"Model not found" error:**
- Ensure `robust_model_gpu.pth` is in the project folder

**Webcam not detected:**
- Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`

**Low accuracy:**
- Press **H** to enable Hybrid Mode for better R/U/V detection
- Ensure good lighting and plain background
