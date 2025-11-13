# extract_features.py
# Convert raw hand landmark data (21x3) into robust features (angles + distances)

import numpy as np
import os
import pandas as pd

GESTURES = ["left", "right", "fire", "accelerate", "brake"]
INPUT_DIR = "data"
OUTPUT_DIR = "features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def pairwise_distances(points):
    dists = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            d = np.linalg.norm(points[i] - points[j])
            dists.append(d)
    return np.array(dists)

def angle_between(v1, v2):
    v1_u = v1 / (np.linalg.norm(v1) + 1e-6)
    v2_u = v2 / (np.linalg.norm(v2) + 1e-6)
    dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    return np.arccos(dot)

def extract_features_from_row(row):
    points = np.array(row).reshape(21, 3)
    wrist = points[0]
    points = points - wrist  # normalize origin

    # 1️⃣ pairwise distances
    dist_features = pairwise_distances(points)

    # 2️⃣ angles between adjacent finger joints
    angle_features = []
    finger_indices = [
        [0, 1, 2, 3, 4],      # thumb
        [0, 5, 6, 7, 8],      # index
        [0, 9, 10, 11, 12],   # middle
        [0, 13, 14, 15, 16],  # ring
        [0, 17, 18, 19, 20]   # pinky
    ]
    for f in finger_indices:
        for i in range(len(f)-2):
            v1 = points[f[i+1]] - points[f[i]]
            v2 = points[f[i+2]] - points[f[i+1]]
            angle_features.append(angle_between(v1, v2))

    return np.concatenate([dist_features, angle_features])

all_data = []
all_labels = []

for gesture in GESTURES:
    path = os.path.join(INPUT_DIR, f"{gesture}.csv")
    if not os.path.exists(path):
        print(f"⚠️ Missing file: {path}")
        continue

    arr = np.loadtxt(path, delimiter=",")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    features = np.array([extract_features_from_row(row) for row in arr])
    np.savetxt(os.path.join(OUTPUT_DIR, f"{gesture}_features.csv"), features, delimiter=",")
    all_data.append(features)
    all_labels += [gesture] * features.shape[0]

print("✅ Feature extraction complete! Saved to /features/")
