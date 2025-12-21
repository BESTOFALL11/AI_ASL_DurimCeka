"""
Debug script to show what the model sees vs what MediaPipe sees
This reveals why professional systems use landmarks instead of images
"""

import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import mediapipe as mp

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Load model
model = models.resnet18(pretrained=False)
model.fc = nn.Sequential(
    nn.Dropout(0.6),
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 26)
)
model = model.to(device)

try:
    model.load_state_dict(torch.load("smart_model.pth", weights_only=True))
    model.eval()
    print("Model loaded!")
except:
    print("ERROR: Model not found")
    exit()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

print("\n" + "="*80)
print("DEBUG MODE - See what the model sees")
print("="*80)
print("\nPress 'q' to quit")
print("\nShowing:")
print("  LEFT: What YOU see (webcam)")
print("  MIDDLE: What MODEL sees (preprocessed image)")
print("  RIGHT: What MediaPipe sees (hand landmarks)")
print("="*80 + "\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    # Process with MediaPipe
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    # Create display windows
    display = np.zeros((720, 1920, 3), dtype=np.uint8)
    
    # LEFT: Original webcam
    display[:, :640] = cv2.resize(frame, (640, 720))
    cv2.putText(display, "YOUR WEBCAM", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Draw landmarks on left view
        frame_with_landmarks = frame.copy()
        mp_drawing.draw_landmarks(
            frame_with_landmarks, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),
        )
        
        # Get bounding box
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        for lm in hand_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)
        
        offset = 40
        y1, y2 = max(0, y_min - offset), min(h, y_max + offset)
        x1, x2 = max(0, x_min - offset), min(w, x_max + offset)
        
        try:
            # MIDDLE: What model sees
            img_crop = frame[y1:y2, x1:x2]
            pil_img = Image.fromarray(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))
            
            # Show preprocessed version
            preprocessed = pil_img.resize((224, 224))
            preprocessed_np = np.array(preprocessed)
            preprocessed_display = cv2.resize(preprocessed_np, (640, 720))
            display[:, 640:1280] = cv2.cvtColor(preprocessed_display, cv2.COLOR_RGB2BGR)
            
            # Get prediction
            img_tensor = transform(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, pred_idx = torch.max(probs, 1)
                prediction = CLASSES[pred_idx.item()]
                confidence = conf.item()
            
            cv2.putText(display, "MODEL INPUT (224x224)", (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, f"Prediction: {prediction}", (650, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display, f"Confidence: {confidence*100:.1f}%", (650, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # RIGHT: Landmark visualization
            landmark_canvas = np.zeros((720, 640, 3), dtype=np.uint8)
            
            # Draw landmarks in normalized space (0-1)
            for idx, lm in enumerate(hand_landmarks.landmark):
                x_norm = int(lm.x * 600) + 20
                y_norm = int(lm.y * 700) + 10
                cv2.circle(landmark_canvas, (x_norm, y_norm), 8, (0, 255, 0), -1)
                cv2.putText(landmark_canvas, str(idx), (x_norm+10, y_norm), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Draw connections
            connections = mp_hands.HAND_CONNECTIONS
            for connection in connections:
                start_idx, end_idx = connection
                start = hand_landmarks.landmark[start_idx]
                end = hand_landmarks.landmark[end_idx]
                start_point = (int(start.x * 600) + 20, int(start.y * 700) + 10)
                end_point = (int(end.x * 600) + 20, int(end.y * 700) + 10)
                cv2.line(landmark_canvas, start_point, end_point, (255, 0, 255), 2)
            
            display[:, 1280:] = landmark_canvas
            cv2.putText(display, "MEDIAPIPE LANDMARKS", (1290, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, "21 keypoints (lighting-invariant)", (1290, 680), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            cv2.putText(display, "Normalized coordinates (0-1)", (1290, 710), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            
        except Exception as e:
            cv2.putText(display, f"Error: {str(e)}", (650, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    else:
        cv2.putText(display, "NO HAND DETECTED", (650, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(display, "NO HAND DETECTED", (1450, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("DEBUG: Image vs Landmarks", display)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
