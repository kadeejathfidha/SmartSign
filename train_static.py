"""
SmartSign - Train the static gesture classifier (Random Forest)
----------------------------------------------------------------
Reads every CSV in data/static/ (one file per letter), normalizes the
landmarks, trains a Random Forest, prints accuracy, and saves the model.

Run:
    python train_static.py

You need at least 2 different letters collected for this to work.
"""

import os
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save figures to file instead of opening a window
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from utils import normalize_landmarks

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "static")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    frames = []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        if name.lower() == "test":  # skip the 3-sample camera test file
            continue
        df = pd.read_csv(f, header=None)
        frames.append(df)
        print(f"  loaded {name}: {len(df)} samples")

    if len(frames) < 2:
        raise SystemExit(
            "Need at least 2 different letters in data/static/ to train. "
            "Collect e.g. A and B first."
        )

    data = pd.concat(frames, ignore_index=True)
    X = data.iloc[:, :63].values
    y = data.iloc[:, 63].astype(str).values
    return X, y


def main():
    print("Loading data...")
    X, y = load_data()

    print("Normalizing landmarks...")
    X = np.array([normalize_landmarks(row) for row in X])

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    print(f"Classes: {list(encoder.classes_)}")

    print("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nTest accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, preds, target_names=encoder.classes_, zero_division=0))

    # Confusion matrix image (useful for your Results slide)
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Static Gesture Confusion Matrix (accuracy {acc * 100:.1f}%)")
    plt.tight_layout()
    cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)

    joblib.dump({"model": model, "encoder": encoder}, os.path.join(MODEL_DIR, "static_rf.joblib"))
    print(f"Model saved to models/static_rf.joblib")
    print(f"Confusion matrix saved to {cm_path}")


if __name__ == "__main__":
    main()