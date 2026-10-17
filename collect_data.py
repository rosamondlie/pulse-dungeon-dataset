"""
collect_data.py — Custom dataset collector for stroke rehabilitation project.
Press 'f' to save FIST photo, 'p' to save PALM photo, 'q' to quit.
"""

import cv2
import os

# ─── Buat folder otomatis ─────────────────────────────────────────────────────
FIST_DIR = "custom_dataset/bola_fist"
PALM_DIR = "custom_dataset/bola_palm"
os.makedirs(FIST_DIR, exist_ok=True)
os.makedirs(PALM_DIR, exist_ok=True)

# ─── Buka webcam ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[Error] Tidak bisa membuka webcam.")
    exit()

fist_count = len([f for f in os.listdir(FIST_DIR) if f.endswith(".jpg")])
palm_count = len([f for f in os.listdir(PALM_DIR) if f.endswith(".jpg")])
print(f"[Info] Melanjutkan dari: fist={fist_count}, palm={palm_count}")
print("[Info] Tekan 'f' = foto FIST | 'p' = foto PALM | 'q' = keluar")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    display = frame.copy()

    # ── UI Text ──────────────────────────────────────────────────────────────
    cv2.putText(display, "Tekan 'f' untuk foto FIST",  (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 60, 255), 2, cv2.LINE_AA)
    cv2.putText(display, "Tekan 'p' untuk foto PALM",  (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0),  2, cv2.LINE_AA)
    cv2.putText(display, "Tekan 'q' untuk Keluar",     (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(display, f"FIST tersimpan : {fist_count}", (10, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 60, 255), 2, cv2.LINE_AA)
    cv2.putText(display, f"PALM tersimpan : {palm_count}", (10, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0),  2, cv2.LINE_AA)

    cv2.imshow("Pengumpul Data - Rehabilitasi Stroke", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("f"):
        path = os.path.join(FIST_DIR, f"fist_{fist_count}.jpg")
        cv2.imwrite(path, frame)
        fist_count += 1
        print(f"[Fist] Foto tersimpan! Total: {fist_count}")

    elif key == ord("p"):
        path = os.path.join(PALM_DIR, f"palm_{palm_count}.jpg")
        cv2.imwrite(path, frame)
        palm_count += 1
        print(f"[Palm] Foto tersimpan! Total: {palm_count}")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n[Selesai] Total foto: FIST={fist_count}, PALM={palm_count}")
print(f"Folder: {os.path.abspath(FIST_DIR)} | {os.path.abspath(PALM_DIR)}")
