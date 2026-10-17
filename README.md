# Pulse Dungeon — Hand Gesture Recognition Pipeline

This project folder contains the Python training and inference pipeline and the custom ball-interaction images used by **Pulse Dungeon**, a pixel-art game prototype designed to support hand-opening and hand-closing practice in post-stroke rehabilitation. The public HaGRID images are downloaded from Hugging Face by the training script; they are not stored in this folder.

The gesture recognition system uses **Google MediaPipe Hands** for landmark extraction and a **Random Forest classifier** to distinguish open-palm and closed-fist postures in real-time via webcam.

---

## Repository Structure

```
strokehanddetection/
├── README.md
├── requirements.txt
├── collect_data.py          # Webcam-based custom dataset collection
├── train.py                 # Training pipeline (feature extraction + RF classifier)
├── main.py                  # Real-time inference module (MediaPipe + UDP to Unity)
├── model_rehab.pkl          # Pre-trained Random Forest model
├── custom_dataset/
│   ├── bola_fist/           # Custom images of the hand squeezing a ball
│   └── bola_palm/           # Custom images of the hand releasing the ball
```

The custom data shared in this project are the raw images in `custom_dataset/bola_fist/` and `custom_dataset/bola_palm/`. No extracted-feature CSV is provided. During training, `train.py` extracts features directly from the images in memory and saves the trained model; it does not export a CSV.

---

## Dataset

The combined dataset contains **3,377 samples** across two gesture classes:

| Source | Fist samples | Palm samples |
|---|---|---|
| HaGRID (public) | 1,474 | 1,631 |
| Custom (ball-based; successfully extracted in the recorded training run) | 181 | 91 |
| **Total** | **1,655** | **1,722** |

The `custom_dataset/bola_fist/` and `custom_dataset/bola_palm/` folders currently contain 200 image files each. The counts in the table are the samples from which MediaPipe successfully extracted landmarks in the recorded training run; some images may not yield a detected hand and are skipped.

### Custom Dataset
Custom images were collected using `collect_data.py` while interacting with a physical ball:
- `bola_fist/` — hand squeezing the ball (labelled `fist` / closed-fist condition)
- `bola_palm/` — hand releasing the ball (labelled `palm` / open-palm condition)

These labels describe the posture classes used by the classifier. The system does not measure ball deformation or grip force.

### HaGRID
The public portion of the dataset is sourced from:
> Kapitanov et al., "HaGRID — HAnd Gesture Recognition Image Dataset," WACV 2024.
> Hugging Face: `cj-mills/hagrid-sample-30k-384p`

### Feature Representation
Each sample is represented as a **63-dimensional wrist-relative feature vector** extracted from MediaPipe Hands landmarks:

```
f_i = (x_i - x_0, y_i - y_0, z_i - z_0)
```

Where landmark 0 is the wrist. Subtracting the wrist coordinates removes translation from the landmark representation. It does not by itself make the features invariant to scale, camera distance, or hand rotation.

The shared custom dataset consists of raw images only; extracted features are not distributed as a CSV. During training, the script computes the 63 values for each image with a detected hand and uses them in memory. The raw images and training script are the available custom-data artifacts.

---

## Model

- **Algorithm:** Random Forest Classifier (`n_estimators=100`, `random_state=42`)
- **Split:** 80% train / 20% test (stratified, `random_state=42`)
- **Test samples:** 676
- **Recorded accuracy:** ~98% (F1-score: 0.98 for both classes)

> ⚠️ The split is performed at the sample level without separating participants or recording sessions. The reported accuracy should not be interpreted as evidence of generalisation to unseen participants.

---

## Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Collect custom data (optional)
```bash
python collect_data.py
# Press 'f' to save fist photo
# Press 'p' to save palm photo
# Press 'q' to quit
```

### 3. Train the model
```bash
python train.py
# Downloads HaGRID via Hugging Face datasets
# Combines with custom_dataset/
# Saves model_rehab.pkl
```

### 4. Run real-time inference
```bash
python main.py
# Requires Unity game running on UDP port 5052
# Camera preview streamed on UDP port 5053
```

---

## Citation

If you use this dataset or pipeline in your research, please cite:

```
Please cite the HaGRID dataset according to its published paper and follow the citation information at the source repository. A citation for this project can be added here once the paper or dataset record has a stable publication or DOI.
```

---

## Authors

- Angel Felicia — BINUS University
- Jarvis Adriel — BINUS University
- Rosamond Patricia Selamat Lie — BINUS University
- Aditya Kurniawan — BINUS University (Supervisor)

---

## Data and License Notes

The custom dataset images were collected by the authors for this project. Before redistribution, confirm that publication of the images is permitted and does not expose identifiable or sensitive information. HaGRID images are obtained from the source dataset and remain subject to its original license and terms; this project does not include a local copy of HaGRID. No formal license file is currently present in this folder, so reuse permissions for this project should not be inferred from this README alone.
