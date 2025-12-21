# 🎮 Universal ASL Sign Language Tutor

> An intelligent, "Dual-Engine" AI Tutor that gamifies learning American Sign Language (ASL). 
> Works on **any computer**, with **any webcam**, and **any background**.

---

## 🌟 Features

*   **Dual-Engine AI**: Combines a 3-Million-Sample Random Forest (for general robustness) with a Geometric Math Engine (for perfect precision on tricky signs like 'U', 'V', 'C', 'O').
*   **Universal Compatibility**: Normalized "Invariant Features" mean it works regardless of how close/far you are or your camera angle.
*   **Gamified Learning**: 
    *   **Interactive Menu**: Select letters or play Random Mode.
    *   **No-Repeat Logic**: Smartly rotates through letters so you don't get bored.
    *   **Scoring System**: Earn points and get "Correct!" flash feedback.
*   **Neon UI**: A polished, sci-fi glassmorphism interface.

---

## 🚀 Quick Start for Friends

### 1. Prerequisites
You need Python installed (Version 3.8 to 3.11 recommended).

### 2. Install
Clone this repository or download the folder. Then open a terminal (Command Prompt) inside the folder and run:

```bash
pip install -r requirements.txt
```

*(This installs the necessary AI brains: MediaPipe, OpenCV, Scikit-Learn)*

### 3. Setup Model
The AI Brain (`universal_model.pkl`) is **300MB+** and cannot be hosted on GitHub directly.

**Option A: Train it yourself (Recommended)**
Run the training script (Takes ~5-10 mins):
```bash
python train_robust.py
```

**Option B: Download it**
*(Link to Google Drive/Dropbox/Release if hosted externally)*

### 4. Play!
Once the model is ready, run:
```bash
python game_universal.py
```

---

## 🎮 How to Play

1.  **Menu Dashboard**: Use your **Mouse** to select a letter you want to practice, or click **RANDOM MODE**.
2.  **Make the Sign**: Raises your hand to the camera.
3.  **Hold It**: When you get it right, a **Green Circle** will fill up. Hold the sign steady!
4.  **Score**: Get 100% loop to score points and move to the next letter.

---

## 🧠 Under the Hood

This isn't just a basic image classifier. It uses **Landmark Extraction**:
1.  **MediaPipe** extracts 21 skeleton points on your hand.
2.  **Normalization** removes the Z-depth and centers the hand, making it "Invariant".
3.  **Random Forest Model** (trained on 3,000,000 variations) predicts the letter.
4.  **Geometric Engine** acts as a referee—if your fingers are in a *perfect* geometric shape (like a 'V' angle), it overrides the AI logic to guarantee correctness.

---

## 🛠 Troubleshooting

*   **"q" to Quit**: If you want to exit, click the window once and press `q`.
*   **Lighting**: It works in most light, but try not to be completely in the dark!
*   **E vs O**: 
    *   **E**: Tight fist, fingers touching palm.
    *   **O**: Make an "O" shape, ensure there is an "Air Gap" inside (don't squash it!).

---

*Created by [Your Name]*
