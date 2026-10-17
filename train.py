"""
train.py — Dataset extraction, feature engineering, and model training.
Stroke Rehabilitation Gamification Project.
"""

import numpy as np
import cv2
import mediapipe as mp
mp_hands = mp.solutions.hands
import joblib
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from typing import Union

# ─── MediaPipe instance (inisialisasi SEKALI, bukan per gambar) ───────────────
_hands_detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
)


def extract_landmarks(image_bgr: np.ndarray) -> Union[np.ndarray, None]:

# def extract_landmarks(image_bgr: np.ndarray) -> np.ndarray | None:
    """
    Extract and normalize 21 hand landmarks from a BGR image.

    Normalization: subtract wrist (landmark 0) x/y/z from all 21 landmarks
    so the feature vector is invariant to hand position and camera distance.

    Returns a 63-dim float32 array, or None if no hand is detected.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = _hands_detector.process(image_rgb)

    if not result.multi_hand_landmarks:
        return None

    lm = result.multi_hand_landmarks[0].landmark  # 21 landmarks

    wx, wy, wz = lm[0].x, lm[0].y, lm[0].z
    features = []
    for point in lm:
        features.extend([point.x - wx, point.y - wy, point.z - wz])

    return np.array(features, dtype=np.float32)


# ─── HaGRID Dataset Loader ────────────────────────────────────────────────────

TARGET_CLASSES = ["fist", "palm"]
MAX_PER_CLASS = 2000


def load_hagrid_dataset() -> tuple[np.ndarray, np.ndarray]:
    """
    Download and process the HaGRID sample dataset from Hugging Face.
    Filters only 'fist' and 'palm' classes, caps at MAX_PER_CLASS each.
    Returns (X, y) as numpy arrays.
    """
    print("[HaGRID] Downloading dataset...")
    ds = load_dataset("cj-mills/hagrid-sample-30k-384p", split="train")

    # Build label name → index mapping from dataset features
    label_names = ds.features["label"].names
    target_indices = {
        label_names.index(f"train_val_{cls}"): cls
        for cls in TARGET_CLASSES
        if f"train_val_{cls}" in label_names
    }

    if not target_indices:
        raise ValueError(f"Classes {TARGET_CLASSES} not found in dataset. Available: {label_names}")

    print(f"[HaGRID] Target label indices: {target_indices}")

    target_idx_set = set(target_indices.keys())
    X, y = [], []
    counts = {cls: 0 for cls in TARGET_CLASSES}

    print("[HaGRID] Mengurai dataset satu per satu (Super-Fast Mode)...")
    
    # 1. Tarik semua data labelnya saja (proses ini sangat cepat, tanpa me-load gambar)
    all_labels = ds["label"]
    
    # 2. Ambil index yang HANYA berisi fist dan palm
    valid_indices = [i for i, lbl in enumerate(all_labels) if lbl in target_idx_set]
    print(f"[HaGRID] Total target: {len(valid_indices)} gambar. Mulai ekstraksi landmark...")

    # 3. Looping HANYA pada index yang valid
    for idx in valid_indices:
        sample = ds[idx] # Gambar dari disk BARU DIBACA di sini (menghemat waktu 90%)
        class_name = target_indices[sample["label"]]

        # 4. Jika sudah mencapai batas 2000 sampel per kelas, lewati
        if counts[class_name] >= MAX_PER_CLASS:
            continue

        # 3. Load gambar (Hanya diproses di RAM jika lolos tahap 1 & 2)
        pil_img = sample["image"]
        img_rgb = np.array(pil_img.resize((224, 224)))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        features = extract_landmarks(img_bgr)
        if features is None:
            continue

        X.append(features)
        y.append(class_name)
        counts[class_name] += 1

        total = sum(counts.values())
        if total % 200 == 0:
            print(f"[HaGRID] Progress: {counts}")

        # 4. Berhenti total jika target 2000/kelas sudah tercapai
        if all(v >= MAX_PER_CLASS for v in counts.values()):
            break

    print(f"[HaGRID] Extracted samples: {counts}")
    return np.array(X), np.array(y)


# ─── Custom Dataset Loader ────────────────────────────────────────────────────

def load_custom_dataset(base_path: str = "custom_dataset/") -> tuple[np.ndarray, np.ndarray]:
    """
    Load custom images from:
      custom_dataset/bola_fist/  → label "fist"
      custom_dataset/bola_palm/  → label "palm"
    Returns (X, y). Returns empty arrays if folder is missing or empty.
    """
    import os

    X, y = [], []
    class_folders = {
        "fist": os.path.join(base_path, "bola_fist"),
        "palm": os.path.join(base_path, "bola_palm"),
    }

    for class_name, folder in class_folders.items():
        if not os.path.isdir(folder):
            print(f"[Custom] Folder not found, skipping: {folder}")
            continue

        image_files = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]

        count = 0
        for fname in image_files:
            img_path = os.path.join(folder, fname)
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue

            features = extract_landmarks(img_bgr)
            if features is None:
                continue

            X.append(features)
            y.append(class_name)
            count += 1

        print(f"[Custom] {class_name}: {count} samples extracted from {folder}")

    if not X:
        return np.empty((0, 63), dtype=np.float32), np.array([])

    return np.array(X), np.array(y)


# ─── Main Training Pipeline ───────────────────────────────────────────────────

def main():
    # 1. Load datasets
    X_hagrid, y_hagrid = load_hagrid_dataset()
    X_custom, y_custom = load_custom_dataset()

    # 2. Combine
    if X_custom.shape[0] > 0:
        X = np.vstack([X_hagrid, X_custom])
        y = np.concatenate([y_hagrid, y_custom])
        print(f"\n[Train] Combined dataset: {X.shape[0]} samples (HaGRID: {X_hagrid.shape[0]}, Custom: {X_custom.shape[0]})")
    else:
        X, y = X_hagrid, y_hagrid
        print(f"\n[Train] Using HaGRID only: {X.shape[0]} samples")

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"[Train] Train: {len(X_train)}, Test: {len(X_test)}")

    # 4. Train Random Forest
    print("[Train] Training RandomForestClassifier...")
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = clf.predict(X_test)
    print("\n[Train] Classification Report:")
    print(classification_report(y_test, y_pred))

    # 6. Save model
    joblib.dump(clf, "model_rehab.pkl")
    print("[Train] Model saved to model_rehab.pkl")


if __name__ == "__main__":
    main()
