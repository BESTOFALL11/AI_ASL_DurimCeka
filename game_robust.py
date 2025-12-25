"""
ROBUST ASL GAME - GPU Neural Network Version
Uses the ROBUST model trained with Data Augmentation
"""

import cv2
import torch
import torch.nn as nn
import mediapipe as mp
import numpy as np
import pickle
import time
from its_engine import IntelligentTutorSystem

import os
import random

# ==============================================================================
# LOAD ROBUST GPU MODEL
# ==============================================================================

# ... (Previous model loading code is fine, skipping lines for brevity) ...

import copy

# ==============================================================================
# GLOBAL INITIALIZATION
# ==============================================================================
# Robust Import for MediaPipe Solutions
try:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    import mediapipe.python.solutions.hands as mp_hands
    import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.framework.formats.landmark_pb2 as landmark_pb2

# ==============================================================================
# VISUAL REFERENCE SYSTEM (SKELETONS ONLY)
# ==============================================================================
DATASET_ROOT = r"asl_landmark_dataset\ASL_Dataset\Train"
CUSTOM_DATA_PATH = "custom_data"
REFERENCE_LANDMARKS = {} # Populated from canonical_landmarks.pkl

# Load Canonical Landmarks (Real Data)
try:
    with open("canonical_landmarks.pkl", "rb") as f:
        REFERENCE_LANDMARKS = pickle.load(f)
    print(f"[SYSTEM] Loaded {len(REFERENCE_LANDMARKS)} canonical reference skeletons.")
except FileNotFoundError:
    print("[SYSTEM] No canonical reference skeletons found. Ghosts will be empty.")

def get_blank_skeleton_img(w=400, h=400):
    return np.zeros((h, w, 3), dtype=np.uint8)

def convert_array_to_proto(flat_array):
    """Convert (42,) array to NormalizedLandmarkList for mp_drawing"""
    from mediapipe.framework.formats import landmark_pb2
    params = landmark_pb2.NormalizedLandmarkList()
    for i in range(21):
        lm = params.landmark.add()
        # Canonical landmarks are already normalized (wrist-centered, scaled)
        # We need to denormalize them for display: center at (0.5, 0.5) and scale to fit
        lm.x = flat_array[i*2] * 0.3 + 0.5      # Scale down and center horizontally
        lm.y = flat_array[i*2+1] * 0.3 + 0.5    # Scale down and center vertically
        lm.z = 0.0
    return params

def load_reference_skeletons():
    print("[SYSTEM] Loading Skeletons...")
    global REFERENCE_LANDMARKS
    
    letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','Space']
    
    # 1. Try CUSTOM DATA first (Highest Priority)
    for letter in letters:
        npy_path = os.path.join(CUSTOM_DATA_PATH, f"{letter}.npy")
        if os.path.exists(npy_path):
            data = np.load(npy_path)
            if len(data) > 0:
                # Pick random sample
                sample = data[random.randint(0, len(data)-1)]
                # Convert to MP Proto for consistent drawing
                proto = convert_array_to_proto(sample)
                REFERENCE_LANDMARKS[letter] = proto
                continue
                
    # 2. Try GENERIC DATA (Fallback)
    extractor = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    
    for letter in letters:
        if letter in REFERENCE_LANDMARKS: continue # Already loaded custom
        
        path = os.path.join(DATASET_ROOT, letter)
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.endswith('.jpg')]
            if files:
                for _ in range(5):
                    img_path = os.path.join(path, random.choice(files))
                    img = cv2.imread(img_path)
                    if img is None: continue
                    res = extractor.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    if res.multi_hand_landmarks:
                        REFERENCE_LANDMARKS[letter] = res.multi_hand_landmarks[0]
                        break
    extractor.close()
    print(f"[OK] Loaded {len(REFERENCE_LANDMARKS)} skeleton models.")

# Call it once
load_reference_skeletons()

# ==============================================================================
# UI COMPONENTS & STATE MANAGEMENT
# ==============================================================================

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[GPU] DEVICE: {device}")

# Load classes
with open('robust_classes.pkl', 'rb') as f:
    CLASSES = pickle.load(f)
print(f"[OK] Classes Loaded: {CLASSES}")

# Define the model architecture (Must match train_from_images.py)
class SignLanguageModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SignLanguageModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# Create model instance
model = SignLanguageModel(input_size=42, hidden_size=128, num_classes=len(CLASSES)).to(device)

# Load weights
try:
    model.load_state_dict(torch.load('robust_model_gpu.pth', map_location=device, weights_only=True))
    model.eval()
    print("[OK] Robust GPU Model Loaded Successfully!")
except Exception as e:
    print(f"[ERROR] Could not load robust model: {e}")
    print("Please wait for training to finish!")

# ==============================================================================
# UI COMPONENTS & STATE MANAGEMENT
# ==============================================================================

class Button:
    def __init__(self, x, y, w, h, text, color=(255, 100, 0), action_payload=None):
        self.rect = (x, y, w, h)
        self.text = text
        self.color = color # BGR
        self.hover_color = (min(color[0]+50,255), min(color[1]+50,255), min(color[2]+50,255))
        self.action_payload = action_payload
        self.is_hovered = False

    def check_hover(self, mouse_x, mouse_y):
        x, y, w, h = self.rect
        self.is_hovered = (x <= mouse_x <= x+w) and (y <= mouse_y <= y+h)
        return self.is_hovered

    def draw(self, frame):
        x, y, w, h = self.rect
        color = self.hover_color if self.is_hovered else self.color
        
        # Glass effect
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1)
        
        # Blend for transparency
        alpha = 0.6 if self.is_hovered else 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Border (Neon Glow)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(self.text, font, 0.7, 2)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        cv2.putText(frame, self.text, (text_x, text_y), font, 0.7, (255, 255, 255), 2)

# Global Mouse State
mouse_x, mouse_y = 0, 0
mouse_click = False

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y, mouse_click
    mouse_x, mouse_y = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_click = True

# --- FEATURE NORMALIZATION (Same as training) ---
def extract_features_robust(landmarks):
    """Extract normalized features matching train_robust.py"""
    points = np.array([[lm.x, lm.y] for lm in landmarks])
    wrist = points[0]
    points = points - wrist
    max_dist = np.max(np.linalg.norm(points, axis=1))
    if max_dist > 0: 
        points /= max_dist
    return points.flatten().astype(np.float32)

# --- GEOMETRIC CLASSIFIER (Backup/Override) ---
class GeometricClassifier:
    """Mathematical classifier for tricky signs"""
    def predict(self, lm, frame_shape=None):
        # Finger states
        thumb_tip = lm[4]
        index_tip = lm[8]
        middle_tip = lm[12]
        ring_tip = lm[16]
        pinky_tip = lm[20]
        
        index_up = lm[8].y < lm[6].y
        middle_up = lm[12].y < lm[10].y
        ring_up = lm[16].y < lm[14].y
        pinky_up = lm[20].y < lm[18].y
        
        # Distances
        def dist(p1, p2):
            return np.linalg.norm([p1.x-p2.x, p1.y-p2.y])
            
        # Scale by hand size
        hand_size = dist(lm[0], lm[9])
        if hand_size == 0: return None
        
        # --- FIST LOGIC (N, M, S, T, A, E) ---
        # All fingers curled (except maybe thumb)
        are_four_fingers_curled = not (index_up or middle_up or ring_up or pinky_up)
        
        if are_four_fingers_curled:
            # Check Thumb horizontal position relative to Knuckles (MCP joints)
            thumb_x = thumb_tip.x
            index_mcp_x = lm[5].x
            middle_mcp_x = lm[9].x
            ring_mcp_x = lm[13].x
            pinky_mcp_x = lm[17].x
            
            # Determine Hand Orientation (Left/Right)
            # If wrist x < index mcp x -> Right Hand
            is_right_hand = lm[0].x < index_mcp_x 
            
            # Normalize X check based on hand side (assuming palm facing camera)
            # Actually, simpler: just check RELATIVE order of knuckles
            # X coordinates: Index < Middle < Ring < Pinky (Right Hand)
            
            # T: Thumb between Index & Middle
            if min(index_mcp_x, middle_mcp_x) < thumb_x < max(index_mcp_x, middle_mcp_x):
                 return "T"
                 
            # N: Thumb between Middle & Ring
            if min(middle_mcp_x, ring_mcp_x) < thumb_x < max(middle_mcp_x, ring_mcp_x):
                 return "N"
                 
            # M: Thumb between Ring & Pinky OR under Pinky
            if min(ring_mcp_x, pinky_mcp_x) < thumb_x < max(ring_mcp_x, pinky_mcp_x):
                 return "M"
                 
            # S: Thumb Crossed over fingers (thumb tip logic varies)
            # Logic: If it's a fist but NOT T, N, or M, it's likely S (or A)
            # A usually has thumb Vertical. S has thumb Horizontal.
            # Simple heuristic: If we are here, it's a fist.
            # If thumb tip Y is above Index/Middle MCP (meaning thumb is up high), it's A.
            # If thumb tip Y is below (crossed over), it's S.
            # (Note: Y increases downwards in screen coords)
            
            thumb_tip_y = thumb_tip.y
            index_mcp_y = lm[5].y
            
            # If thumb tip is significantly higher (lower Y value) than index knuckle -> A
            # Else -> S
            if thumb_tip_y < index_mcp_y - 0.05: # Threshold for "Thumb Up"
                 pass # Let Neural Net handle 'A'
            else:
                 return "S"
            
        # R vs U vs V distinction (Two fingers up)
        uv_dist_norm = dist(index_tip, middle_tip) / hand_size
        if index_up and middle_up and not ring_up and not pinky_up:
            # R detection: fingers close together but CROSSED (index tip higher/lower than middle tip)
            fingers_close_x = abs(index_tip.x - middle_tip.x) < 0.06  # Widened
            tips_different_y = abs(index_tip.y - middle_tip.y) > 0.03  # Check TIP Y positions
            
            if fingers_close_x and tips_different_y:
                return "R"  # Fingers crossed - tips at different heights
            elif uv_dist_norm < 0.12:  # Very strict - only truly together fingers
                return "U"  # Fingers together
            elif uv_dist_norm > 0.30:  # Wide spread
                return "V"  # Fingers apart
            # else: ambiguous, let neural network decide
        
        # W (3 fingers up)
        if index_up and middle_up and ring_up and not pinky_up:
            return "W"
        
        # Y vs I (thumb detection)
        wrist = lm[0]
        thumb_len = np.linalg.norm([thumb_tip.x-wrist.x, thumb_tip.y-wrist.y])
        thumb_base_len = np.linalg.norm([lm[3].x-wrist.x, lm[3].y-wrist.y])
        thumb_extended = thumb_len > thumb_base_len * 1.1
        
        if pinky_up and not index_up and not middle_up and not ring_up:
            return "Y" if thumb_extended else "I"
        
        # L vs D
        if index_up and not middle_up and not ring_up and not pinky_up:
            return "L" if thumb_extended else "D"
            
        # SPACE vs B (Both have 4 fingers up)
        if index_up and middle_up and ring_up and pinky_up:
            # Space: Thumb Out
            # B: Thumb Tucked significantly
            
            # Check thumb extension relative to palm width
            width = dist(lm[5], lm[17])
            thumb_dist = dist(thumb_tip, lm[2]) # Distance to bottom of index finger
            
            # If thumb is far from palm -> Space
            # If close -> B
            # Heuristic: Thumb tip x is across the palm?
            if thumb_extended:
                return "Space"
            else:
                return "B"
        
        return None

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

geo_classifier = GeometricClassifier()
its = IntelligentTutorSystem()
its.load_session()

# ==============================================================================
# MAIN GAME LOOP
# ==============================================================================

# HINTS for Tutorial
HINTS = {
    "N": "FIST: Tuck thumb between Middle & Ring fingers",
    "M": "FIST: Tuck thumb under Ring/Pinky fingers",
    "T": "FIST: Tuck thumb between Index & Middle",
    "S": "FIST: Thumb crossed OVER fingers",
    "A": "FIST: Thumb pointing UP",
    "Space": "Flat Hand (High Five)",
    "B": "Flat Hand, Thumb Tucked"
}

cap = cv2.VideoCapture(0)
if not cap.isOpened(): cap = cv2.VideoCapture(1)

cv2.namedWindow("ASL Tutor [ROBUST]", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ASL Tutor [ROBUST]", 1280, 720)
cv2.setMouseCallback("ASL Tutor [ROBUST]", mouse_callback)

# SCENE STATES
SCENE_MENU = 0
SCENE_GAME = 1
current_scene = SCENE_MENU

# Initialize Menu Buttons - FIXED TO FULL ALPHABET
menu_buttons = []
# Manual Alphabet list to fix missing N, S, T in dataset
letters_grid = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','Space']

start_x = 100
start_y = 150
btn_w, btn_h = 80, 50
padding = 20
cols = 8

for i, char in enumerate(letters_grid):
    row = i // cols
    col = i % cols
    x = start_x + col * (btn_w + padding)
    y = start_y + row * (btn_h + padding)
    menu_buttons.append(Button(x, y, btn_w, btn_h, char, color=(0, 200, 255), action_payload=char))

# Add "Random Mode" Button
menu_buttons.append(Button(start_x, 400, 300, 60, "RANDOM MODE (Non-Stop)", (0, 255, 0), action_payload="RANDOM"))

target_letter = CLASSES[0]
score = 0
active_message = ""
message_timer = 0
is_holding = False
hold_start = 0.0

# Hybrid Mode Toggle (Press H to toggle)
use_hybrid_mode = False  # Start with Pure AI to test model
hybrid_msg_timer = 0

print("="*80)
print("ASL TUTOR [ROBUST MODE] LAUNCHED")
print("="*80)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # Mirror
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    # Process Hand Landmarks
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    # --------------------------------------------------------------------------
    # SCENE: MENU
    # --------------------------------------------------------------------------
    if current_scene == SCENE_MENU:
        # Dim background
        cv2.addWeighted(frame, 0.3, np.zeros_like(frame), 0.7, 0, frame)
        
        # Title
        cv2.putText(frame, "ROBUST AI TUTOR", (100, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 100), 3)
        cv2.putText(frame, "Powered by Augmented Data", (100, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 200, 150), 1)

        # Draw Buttons
        for btn in menu_buttons:
            btn.check_hover(mouse_x, mouse_y)
            btn.draw(frame)
            
            if btn.is_hovered and mouse_click:
                # ACTION
                choice = btn.action_payload
                if choice == "RANDOM":
                    print("[UI] Selected RANDOM Mode")
                    its.current_mode = 'random'
                    its.reset_random_pool()
                    target_letter = its.get_next_letter()
                else: 
                    print(f"[UI] Selected Letter {choice}")
                    its.set_forced_letter(choice)
                    target_letter = choice
                
                mouse_click = False
                current_scene = SCENE_GAME
                score = 0
                
        # Reset click if not consumed
        if mouse_click: mouse_click = False 

    # --------------------------------------------------------------------------
    # SCENE: GAME
    # --------------------------------------------------------------------------
    elif current_scene == SCENE_GAME:
        
        final_pred = "..."
        confidence = 0.0
        used_geo = False
        debug_pred_1 = ""
        debug_pred_2 = ""
        debug_pred_3 = ""
        
        # Back to Menu Button
        back_btn = Button(20, 20, 100, 40, "MENU", (0, 0, 255))
        back_btn.check_hover(mouse_x, mouse_y)
        back_btn.draw(frame)
        
        if back_btn.is_hovered and mouse_click:
            current_scene = SCENE_MENU
            mouse_click = False
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # --- Extract Features ---
            features = extract_features_robust(hand_landmarks.landmark)
            
            # --- Geometric Check (Override for tricky signs) ---
            geo_pred = geo_classifier.predict(hand_landmarks.landmark, frame.shape)
            
            # --- NEURAL NETWORK PREDICTION (Trained on Kaggle Dataset) ---
            # Uses the trained model to predict letter from landmarks
            with torch.no_grad():
                model.eval()
                features_tensor = torch.tensor(features).unsqueeze(0).to(device)
                outputs = model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Get Top 3
                top3_prob, top3_idx = torch.topk(probabilities, 3)
                
                nn_conf = top3_prob[0][0].item()
                pred_idx = top3_idx[0][0].item()
                nn_pred = CLASSES[pred_idx]
                confidence = nn_conf
                
                # Debug Strings
                debug_pred_1 = f"{CLASSES[top3_idx[0][0]]}: {top3_prob[0][0]*100:.1f}%"
                debug_pred_2 = f"{CLASSES[top3_idx[0][1]]}: {top3_prob[0][1]*100:.1f}%"
                debug_pred_3 = f"{CLASSES[top3_idx[0][2]]}: {top3_prob[0][2]*100:.1f}%"

            # --- POST-PROCESSING: R/U/V Disambiguation ---
            # Only apply geometric check when neural network is uncertain
            # If NN is very confident (>90%), trust it!
            if nn_pred in ['R', 'U', 'V'] and confidence < 0.90:
                # Get finger positions
                lm = hand_landmarks.landmark
                index_tip = lm[8]
                middle_tip = lm[12]
                
                # Horizontal spread between fingertips
                horizontal_spread = abs(index_tip.x - middle_tip.x)
                # Vertical difference between tips (crossing check)
                tips_different_y = abs(index_tip.y - middle_tip.y) > 0.03
                
                # Only override if geometric signal is very clear
                if horizontal_spread < 0.06 and tips_different_y:
                    nn_pred = "R"  # Clear crossed fingers
                elif horizontal_spread < 0.03:  # Very strict for U
                    nn_pred = "U"
                elif horizontal_spread > 0.10:  # Very clear V
                    nn_pred = "V"
                # else: trust neural network

            # --- Fusion Logic ---
            # HYBRID MODE: Only if enabled by user
            if use_hybrid_mode and geo_pred: 
                final_pred = geo_pred
                confidence = 1.0
                used_geo = True
                confidence = 1.0
                used_geo = True
                # Debug print to see if Geoms are firing correctly
                # print(f"Hybrid: NN says {nn_pred} ({nn_conf:.2f}), Math Override -> {geo_pred}")
            else:
                final_pred = nn_pred
        
        # Gameplay Logic (Hold Timer)
        if final_pred == target_letter and confidence > 0.40:
            if not is_holding:
                hold_start = time.time()
                is_holding = True
            
            hold_dur = time.time() - hold_start
            hold_progress = min(hold_dur / 2.0, 1.0)
            
            # Draw Timer
            cx, cy = int(hand_landmarks.landmark[9].x * w), int(hand_landmarks.landmark[9].y * h)
            cv2.ellipse(frame, (cx, cy), (40, 40), -90, 0, 360 * hold_progress, (0, 255, 0), 4)

            if hold_progress >= 1.0:
                # Success!
                cv2.rectangle(frame, (0,0), (w,h), (0,255,0), 10)
                cv2.putText(frame, "CORRECT!", (w//2 - 150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)
                
                score += 10
                its.record_attempt(target_letter, True)
                
                if its.current_mode == 'random':
                    target_letter = its.get_next_letter()
                    is_holding = False
                    hold_progress = 0.0
                    cv2.imshow("ASL Tutor [ROBUST]", frame)
                    cv2.waitKey(500)
                else:
                    active_message = "GOOD JOB!"
                    message_timer = time.time() + 1.0
                    is_holding = False
                    hold_progress = 0.0
        else:
            is_holding = False
            hold_progress = 0.0
        
        # UI Overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (280, h), (30, 30, 30), -1) # Wider sidebar for stats
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Target
        cv2.putText(frame, "TARGET", (20, 100), cv2.FONT_HERSHEY_PLAIN, 2, (200, 200, 200), 2)
        cv2.putText(frame, target_letter, (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 4)
        
        # Live Hint Display
        if target_letter in HINTS:
            cv2.putText(frame, HINTS[target_letter], (300, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Your Move
        cv2.putText(frame, "AI SEES:", (20, 300), cv2.FONT_HERSHEY_PLAIN, 2, (200, 200, 200), 2)
        color = (0, 255, 255) if confidence > 0.40 else (0, 0, 255)
        display_pred = final_pred if final_pred != "..." else "?"
        cv2.putText(frame, display_pred, (50, 380), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 4)
        
        # Debug Probabilities
        cv2.putText(frame, debug_pred_1, (20, 450), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1)
        cv2.putText(frame, debug_pred_2, (20, 480), cv2.FONT_HERSHEY_PLAIN, 1.2, (150, 150, 150), 1)
        cv2.putText(frame, debug_pred_3, (20, 510), cv2.FONT_HERSHEY_PLAIN, 1.2, (100, 100, 100), 1)
        
        # Mode Status Display
        mode_text = "HYBRID MODE" if use_hybrid_mode else "TEMPLATE MATCH"
        mode_color = (255, 200, 0) if use_hybrid_mode else (0, 255, 255)
        cv2.putText(frame, "MODE:", (20, 560), cv2.FONT_HERSHEY_PLAIN, 1.2, (200, 200, 200), 1)
        cv2.putText(frame, mode_text, (20, 585), cv2.FONT_HERSHEY_PLAIN, 1.5, mode_color, 2)
        cv2.putText(frame, "[H] Toggle", (20, 610), cv2.FONT_HERSHEY_PLAIN, 1, (150, 150, 150), 1)
        
        # Temporary mode change notification
        if time.time() < hybrid_msg_timer:
            notify_msg = f"MODE: {mode_text}"
            cv2.putText(frame, notify_msg, (350, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, mode_color, 3)
        
        # Score
        cv2.putText(frame, f"Score: {score}", (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Debug Info
        cv2.putText(frame, "[ROBUST AI]", (20, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 1)
        
        # VISUAL REFERENCE (SKELETON)
        if target_letter in REFERENCE_LANDMARKS:
            ref_lm = REFERENCE_LANDMARKS[target_letter]
            
            # Convert canonical array to proto for drawing
            proto_landmarks = convert_array_to_proto(ref_lm)
            
            # Create blank "Ghost" canvas - BIGGER
            ghost_h, ghost_w = 400, 400
            # Solid Black Background for High Contrast
            ghost_img = np.zeros((ghost_h, ghost_w, 3), dtype=np.uint8)
            
            # Draw landmarks on the ghost image - THICKER
            mp_drawing.draw_landmarks(
                ghost_img, 
                proto_landmarks,  # Use converted proto format
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=5, circle_radius=5), # Thicker Lines/Dots
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=3)
            )
            
            # Region to draw (Bottom Right)
            y_offset = h - ghost_h - 20
            x_offset = w - ghost_w - 20
            
            if y_offset >= 0 and x_offset >= 0:
                roi = frame[y_offset:y_offset+ghost_h, x_offset:x_offset+ghost_w]
                
                # SOLID OVERLAY (No Transparency)
                # Just replace the pixels
                frame[y_offset:y_offset+ghost_h, x_offset:x_offset+ghost_w] = ghost_img
                
                # Border
                cv2.rectangle(frame, (x_offset, y_offset), (x_offset+ghost_w, y_offset+ghost_h), (255, 0, 0), 4)
                # Label
                cv2.putText(frame, "LANDMARKS GUIDE", (x_offset, y_offset - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if used_geo:
            cv2.putText(frame, "MATH OVERRIDE!", (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        if time.time() < message_timer:
            cv2.putText(frame, active_message, (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # Show Frame
    cv2.imshow("ASL Tutor [ROBUST]", frame)
    
    # Quit Handler
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('h') or key == ord('H'):
        use_hybrid_mode = not use_hybrid_mode
        hybrid_msg_timer = time.time() + 2.0  # Show message for 2 seconds
        mode_str = "HYBRID (Math ON)" if use_hybrid_mode else "PURE AI"
        print(f"[MODE SWITCH] {mode_str}")

cap.release()
cv2.destroyAllWindows()
