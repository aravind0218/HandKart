# app_live.py
import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyautogui
import time
from collections import deque, Counter

# ---------- CONFIG ----------
MODEL_PATH = "gesture_model.pkl"   # path to your trained model
SMOOTHING_WINDOW = 7               # number of recent predictions to majority-vote
CONFIDENCE_THRESHOLD = 0.60        # min probability to accept prediction
ACTION_COOLDOWN = 0.12             # min seconds between repeated actions (per hand)

# Gesture -> key mapping (change to your labels)
# LEFT-HAND controls steering (a,d), RIGHT-HAND controls accel/fire/brake (w, space, s)
LEFT_HAND_MAP = {
    "left": ("press", "a"),
    "right": ("press", "d"),
    "none": (None, None)
}
RIGHT_HAND_MAP = {
    "accelerate": ("keydown", "w"),   # keyDown while accelerate persists
    "fire": ("press", "space"),       # one-shot press
    "brake": ("keydown", "s"),
    "none": (None, None)
}

# Keys that are kept down while gesture persists
HOLD_KEYS = {"w", "s"}   # set of keys that use keyDown/keyUp mapping
# ----------------------------

# Load model
model = joblib.load(MODEL_PATH)

# Mediapipe init
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Cannot open camera")

# smoothing deques per hand label 'Left'/'Right'
recent_preds = {
    "Left": deque(maxlen=SMOOTHING_WINDOW),
    "Right": deque(maxlen=SMOOTHING_WINDOW)
}
last_action_time = {"Left": 0.0, "Right": 0.0}
current_hold_keys = set()  # track which keys are currently held down by keyDown()

print("HandKart live (two-hand) started. Press 'q' to quit.")

def majority_vote(deq):
    if not deq:
        return "none"
    cnt = Counter(deq)
    most = cnt.most_common(1)[0][0]
    return most

def safe_keydown(key):
    if key not in current_hold_keys:
        pyautogui.keyDown(key)
        current_hold_keys.add(key)

def safe_keyup(key):
    if key in current_hold_keys:
        pyautogui.keyUp(key)
        current_hold_keys.discard(key)

def release_all_hold_keys():
    # call on exit
    for k in list(current_hold_keys):
        try:
            pyautogui.keyUp(k)
        except Exception:
            pass
    current_hold_keys.clear()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb)

        # default labels shown on overlay
        overlay_texts = {"Left": "No hand", "Right": "No hand"}

        # if hands detected, MediaPipe returns lists aligned: multi_hand_landmarks and multi_handedness
        if results.multi_hand_landmarks and results.multi_handedness:
            # iterate paired lists
            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = hand_handedness.classification[0].label  # 'Left' or 'Right' from camera perspective
                # extract flattened landmarks list
                row = []
                for lm in hand_landmarks.landmark:
                    row += [lm.x, lm.y, lm.z]
                # predict with model and probability
                try:
                    probs = model.predict_proba([row])[0]
                    pred_idx = np.argmax(probs)
                    pred_label = model.classes_[pred_idx]
                    confidence = probs[pred_idx]
                except Exception:
                    # if model doesn't support predict_proba, fallback to predict
                    pred_label = model.predict([row])[0]
                    confidence = 1.0

                # accept only if confidence high enough
                if confidence >= CONFIDENCE_THRESHOLD:
                    recent_preds[label].append(pred_label)
                else:
                    recent_preds[label].append("none")

                # draw landmarks and label
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # compute center of hand for text
                cx = int(np.mean([int(lm.x * w) for lm in hand_landmarks.landmark]))
                cy = int(np.mean([int(lm.y * h) for lm in hand_landmarks.landmark])) - 20

                # show raw pred + conf near hand
                disp_text = f"{label}: {pred_label} ({confidence:.2f})" if confidence >= 0 else f"{label}: {pred_label}"
                cv2.putText(frame, disp_text, (cx - 60, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        else:
            # no hands: push 'none' for both to smoothers to allow release
            recent_preds["Left"].append("none")
            recent_preds["Right"].append("none")

        # majority vote results
        voted_left = majority_vote(recent_preds["Left"])
        voted_right = majority_vote(recent_preds["Right"])
        overlay_texts["Left"] = voted_left
        overlay_texts["Right"] = voted_right

        # Map left-hand voted gesture to steering
        now = time.time()
        # LEFT HAND ACTION
        if now - last_action_time["Left"] > ACTION_COOLDOWN:
            last_action_time["Left"] = now
            action_type, key = LEFT_HAND_MAP.get(voted_left, (None, None))
            # Release previously held steering keys before new press to avoid conflicts
            # For steering we will use press (one-shot) for simplicity
            if action_type == "press" and key:
                pyautogui.press(key)

        # RIGHT HAND ACTION
        if now - last_action_time["Right"] > ACTION_COOLDOWN:
            last_action_time["Right"] = now
            action_type, key = RIGHT_HAND_MAP.get(voted_right, (None, None))
            # handle hold keys (accelerate/brake)
            if action_type == "keydown" and key:
                safe_keydown(key)
                # Also ensure we don't hold conflicting key (e.g., release opposite)
                if key == "w" and "s" in current_hold_keys:
                    safe_keyup("s")
                if key == "s" and "w" in current_hold_keys:
                    safe_keyup("w")
            elif action_type == "press" and key:
                # one-shot actions (fire)
                pyautogui.press(key)
            else:
                # if voted_right == 'none' or no action => release hold keys if any
                # Release only keys that belong to RIGHT_HAND_MAP holders
                for k in list(current_hold_keys):
                    if k in HOLD_KEYS:
                        safe_keyup(k)

        # Draw overlay on top-left with both hands' voted gestures
        cv2.rectangle(frame, (0,0), (340,70), (0,0,0), -1)
        cv2.putText(frame, f"Left (steer): {overlay_texts['Left']}", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"Right (accel/fire): {overlay_texts['Right']}", (10,55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("HandKart Live (two-hand)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    release_all_hold_keys()
    cap.release()
    cv2.destroyAllWindows()
    print("Exiting HandKart. Released keys.")
