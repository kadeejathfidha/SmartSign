"""
SmartSign - Data Collection Script
-----------------------------------
Captures hand landmarks using MediaPipe for training the gesture
recognition models.

Two modes:
  STATIC  -> for alphabet signs (single frame per sample)
  DYNAMIC -> for words/phrases (sequence of frames per sample)

Usage:
  python collect_landmarks.py --mode static  --gesture A --samples 50
  python collect_landmarks.py --mode dynamic --gesture hello --samples 40 --seq_len 30

Controls while running:
  SPACE  -> capture a sample (static) / start recording a sequence (dynamic)
  Q      -> quit early

Output:
  data/static/<gesture>.csv        (one row per sample, 63 columns: 21 landmarks x,y,z)
  data/dynamic/<gesture>/<idx>.npy (one file per sample, shape: seq_len x 63)
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import argparse
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")


def extract_landmarks(hand_landmarks):
    """Flatten 21 (x, y, z) landmarks into a single 63-length list."""
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return coords


def collect_static(gesture_name, num_samples):
    out_dir = os.path.join(DATA_DIR, "static")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{gesture_name}.csv")

    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=0.7, min_tracking_confidence=0.7
    )

    collected = 0
    print(f"[STATIC] Collecting '{gesture_name}'. Press SPACE to capture, Q to quit.")

    with open(out_file, mode="a", newline="") as f:
        writer = csv.writer(f)

        while collected < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.putText(frame, f"Gesture: {gesture_name}  Samples: {collected}/{num_samples}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "SPACE=capture  Q=quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("SmartSign - Static Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and results.multi_hand_landmarks:
                landmarks = extract_landmarks(results.multi_hand_landmarks[0])
                writer.writerow(landmarks + [gesture_name])
                collected += 1
                print(f"  captured {collected}/{num_samples}")
            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Saved to {out_file}")


def collect_dynamic(gesture_name, num_samples, seq_len):
    out_dir = os.path.join(DATA_DIR, "dynamic", gesture_name)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=0.7, min_tracking_confidence=0.7
    )

    existing = len([f for f in os.listdir(out_dir) if f.endswith(".npy")])
    collected = 0
    print(f"[DYNAMIC] Collecting '{gesture_name}'. Press SPACE to start each {seq_len}-frame sequence, Q to quit.")

    while collected < num_samples:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Gesture: {gesture_name}  Sample: {collected}/{num_samples}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE=start recording  Q=quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("SmartSign - Dynamic Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            sequence = []
            frame_count = 0
            while frame_count < seq_len:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    landmarks = extract_landmarks(hand_landmarks)
                else:
                    landmarks = [0.0] * 63  # no hand detected this frame

                sequence.append(landmarks)
                frame_count += 1

                cv2.putText(frame, f"RECORDING frame {frame_count}/{seq_len}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("SmartSign - Dynamic Collection", frame)
                cv2.waitKey(1)

            sample_idx = existing + collected
            np.save(os.path.join(out_dir, f"{sample_idx}.npy"), np.array(sequence))
            collected += 1
            print(f"  captured sequence {collected}/{num_samples}")
            time.sleep(0.5)
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartSign landmark data collector")
    parser.add_argument("--mode", choices=["static", "dynamic"], required=True)
    parser.add_argument("--gesture", required=True, help="Label for this gesture, e.g. 'A' or 'hello'")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples to collect")
    parser.add_argument("--seq_len", type=int, default=30, help="Frames per sequence (dynamic mode only)")
    args = parser.parse_args()

    if args.mode == "static":
        collect_static(args.gesture, args.samples)
    else:
        collect_dynamic(args.gesture, args.samples, args.seq_len)
