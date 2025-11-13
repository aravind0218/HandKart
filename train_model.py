import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

GESTURES = ["left", "right", "fire", "accelerate", "brake"]
DATA_DIR = "features"

data = []
labels = []

for g in GESTURES:
    path = os.path.join(DATA_DIR, f"{g}_features.csv")
    if not os.path.exists(path):
        print("Missing:", path)
        continue
    arr = np.loadtxt(path, delimiter=",")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    data.append(arr)
    labels += [g] * arr.shape[0]

X = np.vstack(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

print("✅ Training complete")
print(f"Training accuracy: {model.score(X_train, y_train)*100:.2f}%")
print(f"Test accuracy: {model.score(X_test, y_test)*100:.2f}%")

joblib.dump(model, "gesture_model.pkl")
print("💾 Model saved as gesture_model.pkl")
