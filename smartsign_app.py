"""
SmartSign - Two-way communication app
--------------------------------------
ONE window, both directions:

  SIGN -> SPEECH : hold a sign steady (~0.5 s) and it is added to the word.
                   Press ENTER and the computer speaks the word aloud.
  SPEECH -> TEXT : press L, speak, and your words appear as a live caption.

Run (after train_static.py has created models/static_rf.joblib):
    python smartsign_app.py

Keys:
    SPACE      add a space (finish a word)
    BACKSPACE  delete last character
    ENTER      speak the current text
    L          listen to the other person (speech -> caption)  [needs internet]
    C          clear current text
    Q          quit
"""

import os
import queue
import textwrap
import threading

import cv2
import joblib
import numpy as np
import mediapipe as mp

from utils import normalize_landmarks, extract_landmarks, WordBuilder

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    import speech_recognition as sr
except ImportError:
    sr = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "static_rf.joblib")

CONFIDENCE_THRESHOLD = 0.60
STABLE_FRAMES = 15          # frames a sign must be held before it is added
PANEL_HEIGHT = 250

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ---------------------------------------------------------------- shared state
transcript = []                       # list of (speaker, text)
status = {"msg": "Ready", "listening": False}
speech_queue = queue.Queue()


# ---------------------------------------------------------- text-to-speech
def tts_worker():
    """Runs in its own thread so the camera never freezes while speaking."""
    if pyttsx3 is None:
        return
    try:                              # Windows needs COM initialised per thread
        import pythoncom
        pythoncom.CoInitialize()
    except Exception:
        pass
    try:
        engine = pyttsx3.init()
    except Exception as e:
        print(f"Text-to-speech unavailable: {e}")
        return
    while True:
        text = speech_queue.get()
        if text is None:
            break
        try:
            engine.say(text.lower())
            engine.runAndWait()
        except Exception as e:
            print(f"Speech error: {e}")


# ---------------------------------------------------------- speech-to-text
def listen_worker():
    """Listens once, then adds the recognised sentence to the transcript."""
    status["listening"] = True
    status["msg"] = "Listening... speak now"
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=6, phrase_time_limit=10)
        status["msg"] = "Recognizing..."
        text = recognizer.recognize_google(audio)
        transcript.append(("Hearing", text))
        status["msg"] = "Ready"
    except sr.WaitTimeoutError:
        status["msg"] = "No speech heard - press L to try again"
    except sr.UnknownValueError:
        status["msg"] = "Could not understand - press L to try again"
    except sr.RequestError:
        status["msg"] = "Speech service unreachable - check internet"
    except Exception as e:
        status["msg"] = f"Microphone problem: {e}"
    finally:
        status["listening"] = False


# ------------------------------------------------------------------ drawing
def draw_panel(width, builder):
    panel = np.full((PANEL_HEIGHT, width, 3), 28, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # current word being built
    cv2.putText(panel, "YOUR TEXT (press ENTER to speak):", (12, 22), font, 0.45, (160, 160, 160), 1)
    shown = builder.text[-32:]
    cv2.putText(panel, shown + "_", (12, 58), font, 1.0, (255, 255, 255), 2)
    cv2.line(panel, (10, 72), (width - 10, 72), (70, 70, 70), 1)

    # conversation transcript (last few wrapped lines)
    lines = []
    wrap_chars = max(20, width // 11)
    for speaker, text in transcript[-6:]:
        color = (0, 210, 0) if speaker == "You" else (255, 190, 40)
        wrapped = textwrap.wrap(f"{speaker}: {text}", wrap_chars) or [""]
        for ln in wrapped:
            lines.append((ln, color))
    y = 96
    for ln, color in lines[-5:]:
        cv2.putText(panel, ln, (12, y), font, 0.6, color, 1)
        y += 24

    # status + controls
    status_color = (0, 165, 255) if status["listening"] else (150, 150, 150)
    cv2.putText(panel, status["msg"], (12, PANEL_HEIGHT - 32), font, 0.5, status_color, 1)
    cv2.putText(panel, "SPACE space | BKSP delete | ENTER speak | L listen | C clear | Q quit",
                (12, PANEL_HEIGHT - 10), font, 0.42, (120, 120, 120), 1)
    return panel


def draw_top_bar(frame, label_text, color, progress):
    w = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (w, 54), (30, 30, 30), -1)
    cv2.putText(frame, label_text, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    # hold-to-add progress bar
    cv2.rectangle(frame, (w - 160, 18), (w - 12, 34), (80, 80, 80), 1)
    cv2.rectangle(frame, (w - 160, 18), (w - 160 + int(148 * progress), 34), (0, 200, 0), -1)


# --------------------------------------------------------------------- main
def main():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit("Model not found. Run 'python train_static.py' first.")

    bundle = joblib.load(MODEL_PATH)
    model, encoder = bundle["model"], bundle["encoder"]
    print(f"Loaded model. Known signs: {list(encoder.classes_)}")

    if pyttsx3 is None:
        print("pyttsx3 not installed - speech output disabled.")
    else:
        threading.Thread(target=tts_worker, daemon=True).start()
    if sr is None:
        print("SpeechRecognition not installed - listening disabled.")

    builder = WordBuilder(stable_frames=STABLE_FRAMES)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=0.7, min_tracking_confidence=0.7,
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)          # same mirror as data collection
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        label_text, color = "No hand detected", (0, 0, 255)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            features = normalize_landmarks(extract_landmarks(hand)).reshape(1, -1)
            probs = model.predict_proba(features)[0]
            best = int(np.argmax(probs))
            confidence = float(probs[best])

            if confidence >= CONFIDENCE_THRESHOLD:
                label = str(encoder.classes_[best])
                label_text, color = f"{label}  ({confidence * 100:.0f}%)", (0, 200, 0)
                builder.update(label)
            else:
                label_text, color = f"?  ({confidence * 100:.0f}%)", (0, 165, 255)
                builder.update(None)
        else:
            builder.release()

        draw_top_bar(frame, label_text, color, builder.progress)
        cv2.imshow("SmartSign", np.vstack([frame, draw_panel(frame.shape[1], builder)]))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == 32:                                     # SPACE
            builder.space()
        elif key == 8:                                      # BACKSPACE
            builder.backspace()
        elif key == ord("c"):
            builder.clear()
        elif key == 13:                                     # ENTER -> speak
            text = builder.pop_text()
            if text:
                transcript.append(("You", text))
                speech_queue.put(text)
        elif key == ord("l"):
            if sr is None:
                status["msg"] = "SpeechRecognition not installed"
            elif not status["listening"]:
                threading.Thread(target=listen_worker, daemon=True).start()

    speech_queue.put(None)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
