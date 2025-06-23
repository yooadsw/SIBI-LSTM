import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time
from datetime import datetime

# --- Pengaturan Awal ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
file_path = 'data_sibi_statis.csv'

def initialize_csv_with_header():
    """Membuat file CSV dengan header jika belum ada."""
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['label'] + [f'landmark_{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]
            writer.writerow(header)
        print(f"File CSV baru dibuat: {file_path}")

def validate_and_get_hand(results):
    """Memvalidasi dan mengembalikan data tangan yang terdeteksi."""
    hand_landmarks = None
    hand_type = "TIDAK DIKETAHUI"

    # Prioritaskan tangan kanan, lalu kiri
    if results.right_hand_landmarks:
        hand_landmarks = results.right_hand_landmarks
        hand_type = "KANAN"
    elif results.left_hand_landmarks:
        hand_landmarks = results.left_hand_landmarks
        hand_type = "KIRI"
    else:
        return None, "Tidak ada tangan"

    # Validasi apakah semua landmark di dalam frame
    for lm in hand_landmarks.landmark:
        if not (0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0):
            return None, "Tangan keluar frame"
            
    return hand_landmarks, f"Tangan {hand_type} - OK"

# --- Mulai Program ---
print("=== Perekam Data Isyarat Statis SIBI (Versi Final) ===")
valid_letters = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
while True:
    isyarat = input(f"Masukkan huruf yang akan direkam ({', '.join(valid_letters)}): ").upper().strip()
    if isyarat in valid_letters:
        break
    else:
        print(f"Error: '{isyarat}' bukan huruf yang valid!")

initialize_csv_with_header()

existing_count = 0
if os.path.exists(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        try:
            next(reader) # Skip header
            for row in reader:
                if row and row[0] == isyarat:
                    existing_count += 1
        except StopIteration:
            pass # File is empty after header

print(f"\n✓ Target: '{isyarat}' | Sudah ada: {existing_count} data | Output: {file_path}")
print("--- INSTRUKSI ---")
print("  [S] Simpan Data | [R] Reset Counter | [SPACE] Pause/Resume | [Q] Keluar")

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

simpan_count = 0
paused = False
last_save_time = 0
min_save_interval = 0.5 

with mp_holistic.Holistic(min_detection_confidence=0.7, model_complexity=1) as holistic:
    print("\n🎥 Kamera aktif. Mulai perekaman...")
    while cam.isOpened():
        if not paused:
            success, frame = cam.read()
            if not success:
                print("Error: Gagal membaca frame.")
                continue

            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            
            hand_landmarks, status_text = validate_and_get_hand(results)
            
            # Gambar landmark jika ada
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            # Tampilkan status di layar
            status_color = (0, 255, 0) if "OK" in status_text else (0, 0, 255)
            cv2.putText(frame, f"HURUF: {isyarat}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f"DATA BARU: {simpan_count} (Total: {existing_count + simpan_count})", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"STATUS: {status_text}", (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        else: # Jika di-pause
            cv2.putText(frame, "PAUSED", (frame.shape[1] // 2 - 100, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 3)

        cv2.imshow('SIBI Data Recorder', frame)
        
        key = cv2.waitKey(5) & 0xFF
        current_time = time.time()

        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('r'):
            simpan_count = 0
            print("🔄 Counter direset.")
        elif key == ord('s') and not paused:
            if hand_landmarks and "OK" in status_text:
                if current_time - last_save_time >= min_save_interval:
                    keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten().tolist()
                    
                    with open(file_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([isyarat] + keypoints)
                        
                    simpan_count += 1
                    last_save_time = current_time
                    print(f"✅ Data ke-{simpan_count} untuk '{isyarat}' disimpan - {datetime.now().strftime('%H:%M:%S')}")
                else:
                    print("⚠️ Terlalu cepat! Beri jeda saat menyimpan.")
            else:
                print(f"❌ Gagal menyimpan: Status deteksi bukan OK ({status_text})")

print(f"\n🏁 Selesai! Total {simpan_count} data baru direkam untuk huruf '{isyarat}'.")
cam.release()
cv2.destroyAllWindows()