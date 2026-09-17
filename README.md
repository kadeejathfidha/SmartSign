# SmartSign — Getting Started (Step 1 & 2)

## 1. Environment Setup

Run these on your own laptop (needs a webcam):

```bash
# Create a virtual environment (recommended)
python -m venv smartsign-env

# Activate it
# Windows:
smartsign-env\Scripts\activate
# Mac/Linux:
source smartsign-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

If `pyaudio` fails to install on Windows, run:
```bash
pip install pipwin
pipwin install pyaudio
```

## 2. Collecting Data

### Static gestures (alphabet signs)
Each teammate should collect samples for a few letters to keep the dataset diverse (different hands, lighting).

```bash
python data_collection/collect_landmarks.py --mode static --gesture A --samples 50
```
- A window opens showing your camera feed with hand landmarks drawn.
- Position your hand to form the sign, press **SPACE** to capture a sample.
- Repeat until the sample count reaches your target (50 recommended).
- Press **Q** to quit early.
- Repeat this command for each letter A–Z, swapping `--gesture A` for the next letter.

### Dynamic gestures (words/phrases)
```bash
python data_collection/collect_landmarks.py --mode dynamic --gesture hello --samples 40 --seq_len 30
```
- Press **SPACE** to begin recording one 30-frame sequence of you performing the gesture.
- Repeat until you reach 40 samples per phrase.
- Do this for each of your 15–20 chosen phrases.

## 3. Output

- Static data saves to: `data/static/<gesture>.csv` (one row per sample, 63 landmark values + label)
- Dynamic data saves to: `data/dynamic/<gesture>/<index>.npy` (one file per sequence, shape 30×63)

## 4. Tips for Better Data Quality

- Vary lighting slightly between sessions (natural light, indoor light).
- Have each team member contribute samples — improves generalization across different hand shapes/sizes.
- Keep your hand fully in frame; MediaPipe struggles with partial/cut-off hands.
- For dynamic gestures, keep motion speed roughly consistent across samples of the same gesture.

## Next Steps

Once data collection is done, we move to:
- Step 3: Preprocessing (normalization, label encoding, train/test split)
- Step 4: Train the Random Forest static classifier
- Step 5: Train the LSTM dynamic classifier
