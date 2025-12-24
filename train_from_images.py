"""
TRAIN FROM KAGGLE IMAGES
------------------------
1. Scans 'dataset/ASL Alphabet/asl_alphabet_train'
2. Extracts MediaPipe Landmarks from each image
3. Caches data to 'kaggle_landmarks.npy' (for faster re-runs)
4. Trains the 'robust_model_gpu.pth'
"""

import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ==============================================================================
# CONFIG
# ==============================================================================
DATASET_DIR = r"dataset/ASL Alphabet/asl_alphabet_train"
MODEL_PATH = "robust_model_gpu.pth"
CLASSES_PATH = "robust_classes.pkl"
CACHE_FILE = "kaggle_landmarks.pkl"

# We'll map Kaggle folder names to compatible class names
# 'space' -> 'Space', 'del' -> skip or 'Delete'?
# Game expects: A-Z, Space
CLASS_MAP = {
    'space': 'Space',
    'nothing': None, # Skip
    'del': None      # Skip for now to match game logic
}

# ==============================================================================
# 1. LANDMARK EXTRACTION
# ==============================================================================
def get_classes():
    dirs = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
    classes = []
    for d in dirs:
        if d in CLASS_MAP:
            if CLASS_MAP[d] is not None:
                classes.append(CLASS_MAP[d])
        else:
            classes.append(d) # A, B, C...
    return sorted(list(set(classes)))

def extract_landmarks(img_path, hands):
    img = cv2.imread(img_path)
    if img is None: return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        # Take first hand
        lm = results.multi_hand_landmarks[0]
        # Normalize
        points = np.array([[p.x, p.y] for p in lm.landmark])
        wrist = points[0]
        points = points - wrist
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0: points /= max_dist
        return points.flatten().astype(np.float32)
    return None

def prepare_dataset():
    if os.path.exists(CACHE_FILE):
        print(f"[INFO] Loading cached data from {CACHE_FILE}...")
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)

    print(f"[INFO] Scanning {DATASET_DIR}...")
    
    # Setup MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )
    
    X = []
    y = []
    classes = get_classes()
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    print(f"[INFO] Classes detected: {classes}")
    
    # Walk folders
    original_dirs = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
    
    total_files = sum([len(os.listdir(os.path.join(DATASET_DIR, d))) for d in original_dirs])
    print(f"[INFO] Processing {total_files} images...")
    
    pbar = tqdm(total=total_files)
    
    for folder_name in original_dirs:
        # Map folder to class
        target_class = folder_name
        if folder_name in CLASS_MAP:
            target_class = CLASS_MAP[folder_name]
            
        if target_class is None: # Skip 'del', 'nothing'
            pbar.update(len(os.listdir(os.path.join(DATASET_DIR, folder_name))))
            continue
            
        label_idx = class_to_idx[target_class]
        folder_path = os.path.join(DATASET_DIR, folder_name)
        
        # Process every image in folder
        files = os.listdir(folder_path)
        # Optional: Limit for speed testing? 
        # files = files[:500] 
        
        for f in files:
            file_path = os.path.join(folder_path, f)
            features = extract_landmarks(file_path, hands)
            
            if features is not None:
                X.append(features)
                y.append(label_idx)
            
            pbar.update(1)
            
    pbar.close()
    
    data = (np.array(X), np.array(y), classes)
    
    print(f"[INFO] Saving cache to {CACHE_FILE}...")
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(data, f)
        
    return data

# ==============================================================================
# 2. MODEL DEFINITION
# ==============================================================================
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

class HandLandmarkDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ==============================================================================
# 3. TRAINING LOOP
# ==============================================================================
def train():
    # Load Data
    X_data, y_data, classes = prepare_dataset()
    print(f"[INFO] Training on {len(X_data)} samples. Classes: {len(classes)}")
    
    # Save Classes for Game
    with open(CLASSES_PATH, 'wb') as f:
        pickle.dump(classes, f)
    print(f"[INFO] Saved classes to {CLASSES_PATH}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    # Hyperparams
    input_size = 42
    hidden_size = 128
    num_classes = len(classes)
    num_epochs = 30 # A bit longer for big dataset
    batch_size = 64
    learning_rate = 0.001
    
    # Setup
    dataset = HandLandmarkDataset(X_data, y_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = SignLanguageModel(input_size, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("[INFO] Starting Training...")
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}')

    # Save
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[SUCCESS] Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
