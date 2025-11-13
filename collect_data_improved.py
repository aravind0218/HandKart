# collect_data_improved.py
# Automatic data collector with countdown + normalization
import cv2
import mediapipe as mp
import numpy as np
import os
import time

# CONFIG
GESTURES = ["left", "right", "fire", "accelerate", "brake"]
SAMPLES_PER_GESTURE = 400      # total frames collected per gesture (increase for accuracy)
CAPTURE_FPS = 10               # approximate frames captured per second
COUNTDOWN_SECONDS = 3          # seconds countdown before recording each gesture
SAVE_DIR = "data"

os.makedirs(SAVE_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.6, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

def normalize_landmarks(landmarks):
    # landmarks: list of 21 normalized landmarks (x,y,z)
    # Convert to array (21,3)
    arr = np.array(landmarks).reshape(21,3)
    # Use wrist (landmark 0) as center
    wrist = arr[0, :2].copy()
    coords2d = arr[:, :2] - wrist  # translate to wrist = (0,0)
    # scale by max distance to keep scale invariance
    max_val = np.max(np.linalg.norm(coords2d, axis=1))
    if max_val == 0:
        max_val = 1e-6
    coords2d = coords2d / max_val
    # keep z relative to wrist z
    z = (arr[:, 2] - arr[0,2]).reshape(21,1)
    normalized = np.hstack([coords2d, z])
    return normalized.flatten().tolist()  # length 63

def collect_for_gesture(cap, gesture):
    print(f"\n===> Prepare to record gesture: '{gesture}'")
    print(f"Position your hand(s) consistently. Recording will start after countdown.")
    print(f"Countdown: {COUNTDOWN_SECONDS} seconds")
    time.sleep(1)
    for i in range(COUNTDOWN_SECONDS, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    print("Recording now...")

    data = []
    start_time = time.time()
    frames_collected = 0
    last_capture = 0
    while frames_collected < SAMPLES_PER_GESTURE:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed, retrying...")
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        now = time.time()
        # throttle capture to approx CAPTURE_FPS
        if (now - last_capture) < 1.0 / CAPTURE_FPS:
            # still show preview but skip heavy processing
            cv2.putText(frame, f"{gesture} - collecting... {frames_collected}/{SAMPLES_PER_GESTURE}",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow("Collecting", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Interrupted by user.")
                break
            continue

        last_capture = now

        if results.multi_hand_landmarks:
            # For consistent model, pick the hand you want to record:
            # If two hands present, we pick the biggest (closest) hand
            chosen = None
            max_area = 0
            for hand_landmarks in results.multi_hand_landmarks:
                xs = [lm.x for lm in hand_landmarks.landmark]
                ys = [lm.y for lm in hand_landmarks.landmark]
                area = (max(xs)-min(xs))*(max(ys)-min(ys))
                if area > max_area:
                    max_area = area
                    chosen = hand_landmarks

            if chosen is None:
                continue

            # extract normalized landmarks
            landmarks = []
            for lm in chosen.landmark:
                landmarks += [lm.x, lm.y, lm.z]
            normalized = normalize_landmarks(landmarks)
            data.append(normalized)
            frames_collected += 1

            # draw and overlay progress
            mp_draw.draw_landmarks(frame, chosen, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, f"{gesture} - {frames_collected}/{SAMPLES_PER_GESTURE}",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        else:
            cv2.putText(frame, f"No hand detected - show the gesture clearly!",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow("Collecting", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Interrupted by user.")
            break

    # Save to CSV
    if len(data) > 0:
        arr = np.array(data)
        save_path = os.path.join(SAVE_DIR, f"{gesture}.csv")
        np.savetxt(save_path, arr, delimiter=",")
        print(f"Saved {len(data)} samples to {save_path}")
    else:
        print("No data collected for", gesture)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    try:
        for gesture in GESTURES:
            print("\n=====================================")
            print(f"Next gesture: {gesture}")
            print("Make sure you use the correct hand for this gesture:")
            print(" - For steering (left/right) use your LEFT hand consistently.")
            print(" - For accelerate/fire/brake use your RIGHT hand consistently.")
            input("Press Enter to begin countdown for this gesture...")
            collect_for_gesture(cap, gesture)
            print("Take a short break. Next gesture will begin when you press Enter.")
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Finished collection.")

if __name__ == "__main__":
    main()
