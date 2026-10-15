"""
SmartSign - Live static gesture detection
------------------------------------------
Opens the webcam, tracks your hand with MediaPipe, and shows the
predicted letter with a confidence score.

Run (after train_static.py has created models/static_rf.joblib):
    python predict_live.py

Press Q to quit.
"""

import os
import cv2
import joblib
import numpy as np
import mediapipe as mp

from utils import normalize_landmarks, extract_landmarks

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "static_rf.joblib")
CONFIDENCE_THRESHOLD = 0.60  # below this we show "?" instead of guessing

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def main():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit("Model not found. Run 'python train_static.py' first.")

    bundle = joblib.load(MODEL_PATH)
    model, encoder = bundle["model"], bundle["encoder"]
    print(f"Loaded model. Known letters: {list(encoder.classes_)}")

    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=0.7, min_tracking_confidence=0.7,
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip exactly like the data collection script did
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        label_text = "No hand detected"
        color = (0, 0, 255)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            features = normalize_landmarks(extract_landmarks(hand)).reshape(1, -1)
            probs = model.predict_proba(features)[0]
            best = int(np.argmax(probs))
            confidence = float(probs[best])

            if confidence >= CONFIDENCE_THRESHOLD:
                label_text = f"{encoder.classes_[best]}  ({confidence * 100:.0f}%)"
                color = (0, 200, 0)
            else:
                label_text = f"?  ({confidence * 100:.0f}%)"
                color = (0, 165, 255)

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (30, 30, 30), -1)
        cv2.putText(frame, label_text, (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
        cv2.imshow("SmartSign - Live Detection (Q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()