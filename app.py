import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyautogui
import time

# Load model
model = joblib.load("gesture_model.pkl")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
prev_gesture = None
last_action_time = time.time()

print("🖐️ HandKart Live Control Started! Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    gesture = "None"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            row = []
            for lm in hand_landmarks.landmark:
                row += [lm.x, lm.y, lm.z]

            prediction = model.predict([row])[0]
            gesture = prediction

            # To avoid repeating same action rapidly
            if gesture != prev_gesture or (time.time() - last_action_time) > 1:
                prev_gesture = gesture
                last_action_time = time.time()

                if gesture == "left":
                    pyautogui.press('a')
                elif gesture == "right":
                    pyautogui.press('d')
                elif gesture == "accelerate":
                    pyautogui.keyDown('w')
                elif gesture == "fire":
                    pyautogui.press('space')

    cv2.putText(frame, f"Gesture: {gesture}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("HandKart Live", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
