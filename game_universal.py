"""
UNIVERSAL ASL GAME - The "Intelligent" Version
Combines:
1. Robust ML (Space-Invariant Random Forest)
2. Symbolic AI (Geometric Heuristics/Rules)
3. Intelligent Tutoring (Adaptive logic)

This version works for ANY webcam, ANY position, ANY hand size.
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from its_engine import IntelligentTutorSystem

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

# --- 1. FEATURE NORMALIZATION (The "Intelligence" Part) ---
def normalize_landmarks(landmarks):
    """
    Convert raw coordinates to spatially invariant features (2D ONLY)
    Matches train_robust.py
    """
    points_3d = np.array(landmarks).reshape(-1, 3)
    points = points_3d[:, :2]  # Only X, Y
    
    wrist = points[0]
    points = points - wrist  # Translation
    
    distances = np.linalg.norm(points, axis=1)
    max_dist = np.max(distances)
    
    if max_dist > 0:
        points = points / max_dist  # Scale
        
    return points.flatten().tolist()

# --- 2. HEURISTIC GUARDRAILS (Symbolic AI) ---
class HeuristicSanitizer:
    """
    Rule-based system to vet ML predictions.
    Prevents "impossible" predictions (e.g. 'A' with open hand).
    """
    def sanitize(self, predicted_letter, hand_landmarks):
        lm = hand_landmarks.landmark
        
        # Are fingers extended?
        thumb_tip = lm[4]
        thumb_ip = lm[3]
        
        wrist = lm[0]
        thumb_len = np.linalg.norm([thumb_tip.x-wrist.x, thumb_tip.y-wrist.y])
        thumb_base_len = np.linalg.norm([thumb_ip.x-wrist.x, thumb_ip.y-wrist.y])
        thumb_up = thumb_len > thumb_base_len * 1.1 # Heuristic for thumb extension
        
        index_up = lm[8].y < lm[6].y
        middle_up = lm[12].y < lm[10].y
        ring_up = lm[16].y < lm[14].y
        pinky_up = lm[20].y < lm[18].y
        
        # --- LOGIC RULES (The "Brain") ---
        
        # 1. Distinguish Y (Thumb+Pinky) vs I (Pinky only)
        if predicted_letter == 'I':
            if thumb_up: # If thumb is also up, it's likely Y
                return "Y" if pinky_up else "Unknown"
            if not pinky_up:
                return "Unknown"
                
        if predicted_letter == 'Y':
            if not (thumb_up and pinky_up):
                return "Unknown"
                
        # 2. Distinguish L (Thumb+Index) vs D/Index (Index only)
        if predicted_letter == 'L':
            if not (thumb_up and index_up):
                return "Unknown"
            if middle_up or ring_up or pinky_up: 
                return "Unknown"
                
        if predicted_letter == 'D': 
            if not index_up:
                return "Unknown"
            if pinky_up and ring_up and middle_up: 
                return "Unknown"

        # 3. Fist Letters (A, E, M, N, S, T)
        fist_letters = ['A', 'E', 'M', 'N', 'S', 'T']
        if predicted_letter in fist_letters:
            extended_count = sum([index_up, middle_up, ring_up, pinky_up])
            if extended_count >= 3:
                return "Unknown"

        # 4. Open Hand Letters (B, F, W)
        if predicted_letter == 'B':
            if not (index_up and middle_up and ring_up and pinky_up):
                return "Unknown"
                
        return predicted_letter

class GeometricClassifier:
    """
    Mathematical classifier to identify signs purely by geometry.
    Used when ML model is unsure.
    """
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
            
        uv_dist = dist(index_tip, middle_tip)
        # Scale distance by hand size (wrist to middle finger mcp)
        hand_size = dist(lm[0], lm[9])
        if hand_size == 0: return None
        uv_dist_norm = uv_dist / hand_size
        
        # LOGIC FOR U vs V
        if index_up and middle_up and not ring_up and not pinky_up:
            # If fingers close -> U
            if uv_dist_norm < 0.20:
                print(f"[GEO] Detect U (Dist: {uv_dist_norm:.2f})")
                return "U"
            # If fingers apart -> V
            elif uv_dist_norm > 0.25:
                print(f"[GEO] Detect V (Dist: {uv_dist_norm:.2f})")
                return "V"
                
        # LOGIC FOR W (Index, Mid, Ring up)
        if index_up and middle_up and ring_up and not pinky_up:
            return "W"

        # LOGIC FOR I (Pinky only) vs Y (Pinky + Thumb)
        wrist = lm[0]
        thumb_len = np.linalg.norm([thumb_tip.x-wrist.x, thumb_tip.y-wrist.y])
        thumb_base_len = np.linalg.norm([lm[3].x-wrist.x, lm[3].y-wrist.y])
        thumb_is_extended = thumb_len > thumb_base_len * 1.1

        if pinky_up and not index_up and not middle_up and not ring_up:
            if thumb_is_extended:
                return "Y"
            else:
                return "I"

        # LOGIC FOR L (Index + Thumb) vs D (Index only)
        if index_up and not middle_up and not ring_up and not pinky_up:
            if thumb_is_extended:
                return "L"
            else:
                return "D"
            
        # LOGIC FOR C (Curved fingers) vs O (Closed) vs F (Index/Thumb touch + others UP)
        ti_dist = dist(thumb_tip, index_tip)
        ti_dist_norm = ti_dist / hand_size
        
        # F: Index+Thumb touching, Middle/Ring/Pinky UP
        if ti_dist_norm < 0.2:
            if middle_up and ring_up and pinky_up:
                return "F"
        
        # O: Index+Thumb touching (or close), ALL fingers curved/down
        # Key distinction E vs O: 'E' fingers curled tight, 'O' fingers curved open
        index_wrist_dist = dist(index_tip, wrist) / hand_size
        
        if ti_dist_norm < 0.25: 
             if not index_up and not middle_up and not ring_up and not pinky_up:
                 if index_wrist_dist > 0.35: # O has "air" inside
                     return "O"
         
        if 0.25 < ti_dist_norm < 0.6: # Medium gap -> C
            if index_tip.y < thumb_tip.y:
                return "C"

        # LOGIC FOR B (All 4 up, tight) vs 5/Space (All 4 up, spread)
        if index_up and middle_up and ring_up and pinky_up:
            spread = dist(index_tip, pinky_tip) / hand_size
            if spread < 0.5:
                return "B"
            else:
                return "Space" 
        
        return None

# Load Universal Model
print("Loading Universal Robust Model...")
with open('universal_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    rf_model = model_data['model']
    CLASSES = model_data['classes']
print("[OK] Model Loaded - Ready for any webcam!")

# Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

sanitizer = HeuristicSanitizer()
geo_classifier = GeometricClassifier()
its = IntelligentTutorSystem()
its.load_session()

# ==============================================================================
# MAIN GAME LOOP WITH SCENES
# ==============================================================================

cap = cv2.VideoCapture(0)
if not cap.isOpened(): cap = cv2.VideoCapture(1)

cv2.namedWindow("Universal ASL Tutor", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Universal ASL Tutor", 1280, 720)
cv2.setMouseCallback("Universal ASL Tutor", mouse_callback)

# SCENE STATES
SCENE_MENU = 0
SCENE_GAME = 1
current_scene = SCENE_MENU

# Initialize Menu Buttons
menu_buttons = []
letters_grid = ['A','B','C','D','E','F','G','H','I','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
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

target_letter = "A" # Default
score = 0
active_message = ""
message_timer = 0
is_holding = False
hold_start = 0.0

print("="*80)
print("ASL TUTOR UI LAUNCHED")
print("=====================")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # Mirror
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    # Process Hand Landmarks (Always needed for logic)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    # --------------------------------------------------------------------------
    # SCENE: MENU
    # --------------------------------------------------------------------------
    if current_scene == SCENE_MENU:
        # Dim background
        cv2.addWeighted(frame, 0.3, np.zeros_like(frame), 0.7, 0, frame)
        
        # Title
        cv2.putText(frame, "SELECT A LETTER TO PRACTICE", (100, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(frame, "Use Mouse to Select", (100, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

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
                
                mouse_click = False # Consumed
                current_scene = SCENE_GAME
                score = 0
                
        # Reset click if not consumed (clicked empty space)
        if mouse_click: mouse_click = False 

    # --------------------------------------------------------------------------
    # SCENE: GAME
    # --------------------------------------------------------------------------
    elif current_scene == SCENE_GAME:
        
        # 1. Prediction Logic (Same as before)
        current_pred = "..."
        confidence = 0.0
        final_pred = "..."
        ml_conf = 0.0
        used_geo = False
        
        # Draw "Back to Menu" Button (Small, top left)
        # Hacky button draw here
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
            raw_features = []
            for lm in hand_landmarks.landmark:
                raw_features.extend([lm.x, lm.y, lm.z]) # Use Z for extraction, but function only uses X,Y
            
            # --- Normalize (Invariant) ---
            normalized_features = normalize_landmarks(raw_features)
            
            # --- Run Dual-Engine ---
            # 1. Geometric
            geo_pred = geo_classifier.predict(hand_landmarks.landmark, frame.shape)
            
            # 2. ML
            features_array = np.array(normalized_features).reshape(1, -1)
            pred_idx = rf_model.predict(features_array)[0]
            probs = rf_model.predict_proba(features_array)[0]
            ml_conf = np.max(probs)
            ml_pred = CLASSES[pred_idx]
            
            # --- Fusion Logic ---
            if geo_pred:
                final_pred = geo_pred
                confidence = 1.0
                used_geo = True
                if ml_pred != geo_pred:
                    print(f"Override: ML({ml_pred} {int(ml_conf*100)}%) -> Geo({geo_pred})")
            else:
                # Heuristic Guardrail
                final_pred = sanitizer.sanitize(ml_pred, hand_landmarks)
                confidence = ml_conf
        
        # 2. Gameplay Logic
        if final_pred == target_letter and confidence > 0.40:
             # Logic for Hold Timer
             if not is_holding:
                 hold_start = time.time()
                 is_holding = True
             
             hold_dur = time.time() - hold_start
             hold_progress = min(hold_dur / 2.0, 1.0) # 2.0 Second Hold
             
             # Draw Timer at Hand Center (Index MCP is stable)
             cx, cy = int(hand_landmarks.landmark[9].x * w), int(hand_landmarks.landmark[9].y * h)
             cv2.ellipse(frame, (cx, cy), (40, 40), -90, 0, 360 * hold_progress, (0, 255, 0), 4)

             if hold_progress >= 1.0:
                 # Success!
                 cv2.rectangle(frame, (0,0), (w,h), (0,255,0), 10)
                 cv2.putText(frame, "CORRECT!", (w//2 - 150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)
                 
                 score += 10
                 its.record_attempt(target_letter, True)
                 
                 # Logic for next letter
                 if its.current_mode == 'random':
                     target_letter = its.get_next_letter()
                     is_holding = False
                     hold_progress = 0.0
                     # Small pause to celebrate
                     cv2.imshow("Universal ASL Tutor", frame)
                     cv2.waitKey(500)
                 else:
                     # Selection mode: just flash success
                     active_message = "GOOD JOB!"
                     message_timer = time.time() + 1.0
                     is_holding = False
                     hold_progress = 0.0
        else:
             is_holding = False
             hold_progress = 0.0

        # 3. UI Overlay (Neon Style)
        # Sidebar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (250, h), (30, 30, 30), -1) # Dark sidebar
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Target
        cv2.putText(frame, "TARGET", (20, 100), cv2.FONT_HERSHEY_PLAIN, 2, (200, 200, 200), 2)
        cv2.putText(frame, target_letter, (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 5)
        
        # Your Move
        cv2.putText(frame, "YOU", (20, 350), cv2.FONT_HERSHEY_PLAIN, 2, (200, 200, 200), 2)
        color = (0, 255, 255) if confidence > 0.40 else (0, 0, 255)
        
        display_pred = final_pred if final_pred != "..." else "?"
        cv2.putText(frame, display_pred, (50, 470), cv2.FONT_HERSHEY_SIMPLEX, 4, color, 5)
        
        # Score
        cv2.putText(frame, f"Score: {score}", (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Debug Info
        if used_geo:
            cv2.putText(frame, "MATH MATCH!", (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        if time.time() < message_timer:
             cv2.putText(frame, active_message, (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # Show Frame
    cv2.imshow("Universal ASL Tutor", frame)
    
    # Quit Handler
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
