import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyautogui

st.set_page_config(page_title="HandKart - Gesture Controller", layout="centered")
st.title("🖐️ HandKart – Play Smash Karts with Hand Gestures")

# Load your trained model
model = joblib.load("gesture_model.pkl")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Streamlit camera input
camera_input = st.camera_input("Show your hand 👇")

if camera_input:
    # Convert camera input to OpenCV image
    bytes_data = camera_input.getvalue()
    np_arr = np.frombuffer(bytes_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Process with MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    gesture = "None"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks for prediction
            row = []
            for lm in hand_landmarks.landmark:
                row += [lm.x, lm.y, lm.z]

            # Predict gesture
            prediction = model.predict([row])[0]
            gesture = prediction

            # Map gestures to game actions
            if gesture == "left":
                pyautogui.press('a')
            elif gesture == "right":
                pyautogui.press('d')
            elif gesture == "accelerate":
                pyautogui.keyDown('w')
            elif gesture == "fire":
                pyautogui.press('space')

    # Display camera feed with gesture label
    st.image(frame, channels="BGR", caption=f"Detected Gesture: {gesture}")