"""
QUICK DATA COLLECTION - Capture YOUR webcam data for problematic letters
This solves the domain gap issue by training on YOUR exact setup!
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import pickle

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

# Letters to collect (focus on problematic ones)
LETTERS_TO_COLLECT = ['M', 'N', 'S', 'T', 'K', 'P', 'Q', 'R', 'X', 'Y', 'Z']
SAMPLES_PER_LETTER = 50  # Quick collection - 50 samples each

print("="*80)
print("WEBCAM DATA COLLECTION")
print("="*80)
print(f"Letters to collect: {LETTERS_TO_COLLECT}")
print(f"Samples per letter: {SAMPLES_PER_LETTER}")
print(f"\nThis will take ~5-10 minutes")
print("="*80)

# Storage
collected_data = {
    'landmarks': [],
    'labels': [],
    'letter_names': []
}

current_letter_idx = 0
current_letter = LETTERS_TO_COLLECT[current_letter_idx]
samples_collected = 0
countdown = 3
countdown_start = None
is_ready = False

print(f"\nGet ready to show letter: {current_letter}")
print("Position your hand and press SPACE when ready")

while True:
    ret, img = cap.read()
    if not ret:
        break
    
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    
    # Process hand
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    # UI Background
    cv2.rectangle(img, (0, 0), (w, 150), (20, 20, 20), -1)
    
    # Instructions
    if not is_ready:
        cv2.putText(img, f"LETTER: {current_letter}", (w//2 - 100, 60), 
                   cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
        cv2.putText(img, "Press SPACE when ready", (w//2 - 200, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        # Countdown
        import time
        if countdown_start is None:
            countdown_start = time.time()
        
        elapsed = time.time() - countdown_start
        remaining = max(0, countdown - int(elapsed))
        
        if remaining > 0:
            cv2.putText(img, f"Starting in {remaining}...", (w//2 - 150, 80),
                       cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)
        else:
            # Collecting
            cv2.putText(img, f"COLLECTING: {current_letter}", (w//2 - 200, 60),
                       cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)
            cv2.putText(img, f"Sample {samples_collected}/{SAMPLES_PER_LETTER}", 
                       (w//2 - 150, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            
            # Auto-collect when hand detected
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Draw hand
                mp_drawing.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),
                )
                
                # Extract landmarks
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                
                # Save sample
                collected_data['landmarks'].append(landmarks)
                collected_data['labels'].append(current_letter)
                collected_data['letter_names'].append(current_letter)
                
                samples_collected += 1
                
                # Visual feedback
                cv2.circle(img, (w//2, h//2), 50, (0, 255, 0), 5)
                
                # Small delay between captures
                cv2.imshow("Data Collection", img)
                cv2.waitKey(100)
                
                # Check if done with this letter
                if samples_collected >= SAMPLES_PER_LETTER:
                    print(f"[OK] Collected {SAMPLES_PER_LETTER} samples for {current_letter}")
                    
                    # Move to next letter
                    current_letter_idx += 1
                    if current_letter_idx >= len(LETTERS_TO_COLLECT):
                        print("\n[OK] COLLECTION COMPLETE!")
                        break
                    
                    current_letter = LETTERS_TO_COLLECT[current_letter_idx]
                    samples_collected = 0
                    is_ready = False
                    countdown_start = None
                    
                    print(f"\nNext letter: {current_letter}")
                    print("Press SPACE when ready")
    
    # Progress bar (bottom)
    total_samples = len(LETTERS_TO_COLLECT) * SAMPLES_PER_LETTER
    current_total = current_letter_idx * SAMPLES_PER_LETTER + samples_collected
    progress = current_total / total_samples
    
    cv2.rectangle(img, (0, h-30), (w, h), (30, 30, 30), -1)
    cv2.rectangle(img, (0, h-30), (int(w * progress), h), (0, 255, 0), -1)
    cv2.putText(img, f"Overall: {current_total}/{total_samples}", (w//2 - 100, h-8),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow("Data Collection", img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' ') and not is_ready:
        is_ready = True
        countdown_start = None
    elif key == ord('q'):
        print("\nCollection cancelled by user")
        break

cap.release()
cv2.destroyAllWindows()

# Save collected data
if len(collected_data['landmarks']) > 0:
    print("\nSaving collected data...")
    
    # Convert to numpy
    X_custom = np.array(collected_data['landmarks'], dtype=np.float32)
    
    # Save
    with open('webcam_custom_data.pkl', 'wb') as f:
        pickle.dump(collected_data, f)
    
    print(f"[OK] Saved {len(X_custom)} samples to webcam_custom_data.pkl")
    print("\nNext step: Run 'python train_custom.py' to train on your data!")
else:
    print("\nNo data collected")
