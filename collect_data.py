import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Make dataset folder
os.makedirs("data", exist_ok=True)

# Gestures you want
gestures = ["left", "right", "accelerate", "fire"]

for gesture in gestures:
    cap = cv2.VideoCapture(0)
    print(f"Show gesture: {gesture}")
    print("Press 's' to start recording, 'q' to quit.")
    
    data = []

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                row = []
                for lm in hand_landmarks.landmark:
                    row += [lm.x, lm.y, lm.z]
                data.append(row)

        cv2.putText(frame, f"Gesture: {gesture}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Collecting Data", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            np.savetxt(f"data/{gesture}.csv", np.array(data), delimiter=",")
            print(f"Saved {gesture} data!")
            break

    cap.release()
    cv2.destroyAllWindows()
