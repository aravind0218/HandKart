import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

gestures = ["left", "right", "accelerate", "fire"]
data = []
labels = []

for gesture in gestures:
    file_path = f"data/{gesture}.csv"
    if os.path.exists(file_path):
        gesture_data = np.loadtxt(file_path, delimiter=",")
        for row in gesture_data:
            data.append(row)
            labels.append(gesture)
    else:
        print(f"Warning: {gesture}.csv not found!")

# Convert to NumPy arrays
X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"✅ Model accuracy: {accuracy*100:.2f}%")

joblib.dump(model, "gesture_model.pkl")
print("💾 Model saved as gesture_model.pkl ✅")

