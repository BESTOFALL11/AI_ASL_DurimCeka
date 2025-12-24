"""
EXTRACT CANONICAL LANDMARKS
---------------------------
Calculates the 'Average Skeleton' for each letter (A-Z) from the Kaggle dataset.
This provides a PERFECT visual guide for the user, based on real training data.
"""

import pickle
import numpy as np

CACHE_FILE = "kaggle_landmarks.pkl"
OUTPUT_FILE = "canonical_landmarks.pkl"

def main():
    print(f"Loading {CACHE_FILE}...")
    try:
        with open(CACHE_FILE, 'rb') as f:
            X, y, classes = pickle.load(f)
    except FileNotFoundError:
        print("Error: cache file not found! Run train_from_images.py first.")
        return

    print(f"Loaded {len(X)} samples. Classes: {classes}")

    # Accumulators
    sums = {}
    counts = {}

    for i in range(len(X)):
        label_idx = y[i]
        features = X[i]

        if label_idx not in sums:
            sums[label_idx] = np.zeros_like(features)
            counts[label_idx] = 0
            
        sums[label_idx] += features
        counts[label_idx] += 1

    # Calculate Averages
    canonical = {}
    for idx in sums:
        class_name = classes[idx]
        avg_features = sums[idx] / counts[idx]
        
        # Enforce re-normalization just to be safe (Mean of unit vectors isn't unit vector)
        # But for visualization, raw mean is actually better (shows variance as shrinking)
        # Let's keep it simple: Raw Mean
        canonical[class_name] = avg_features
        print(f"  {class_name}: Averaged {counts[idx]} samples")

    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(canonical, f)
    
    print("Done!")

if __name__ == "__main__":
    main()
