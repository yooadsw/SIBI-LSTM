import cv2
import numpy as np
import os
import mediapipe as mp

DATA_PATH = "data_sibi_dinamis"
# Kita definisikan aksi dengan huruf besar untuk perbandingan
actions = ['J', 'Z', 'LAINNYA'] 
sequence_length = 30 

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

def main():
    print("--- Perekam Data Dinamis Interaktif ---")
    
    
    prompt_actions = ['J', 'Z', 'Lainnya'] # Ini yang ditampilkan ke pengguna
    while True:
        user_input = input(f"Masukkan aksi yang akan direkam ({', '.join(prompt_actions)}): ").strip()
        action_upper = user_input.upper() 
        
        if action_upper in actions:
            # Jika input adalah 'LAINNYA', kita gunakan nama folder yang benar ('Lainnya')
            if action_upper == 'LAINNYA':
                action = 'Lainnya'
            else:
                action = action_upper # Untuk 'J' dan 'Z'
            break
        else:
            print(f"Error: Aksi '{user_input}' tidak valid. Pilih dari: {', '.join(prompt_actions)}")

    action_path = os.path.join(DATA_PATH, action)
    os.makedirs(action_path, exist_ok=True)

    start_sequence = 0
    while os.path.exists(os.path.join(action_path, str(start_sequence))):
        start_sequence += 1
    
    print(f"\nSiap merekam untuk aksi '{action}'. Akan dimulai dari video ke-{start_sequence}.")
    print("Arahkan tangan ke kamera.")
    print("Tekan 'S' untuk MULAI MEREKAM satu video.")
    print("Tekan 'Q' untuk keluar.")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    sequence_count = start_sequence
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            
            frame = cv2.flip(frame, 1)
            image, results = mediapipe_detection(frame, holistic)

            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            cv2.putText(frame, f"AKSI: {action}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Video Tersimpan: {sequence_count}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Tekan 'S' untuk Merekam Video Berikutnya", (15, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow('Kamera Pengumpul Data Dinamis', frame)

            key = cv2.waitKey(10) & 0xFF

            if key == ord('q') or key == ord('Q'):
                break

            if key == ord('s') or key == ord('S'):
                print(f"\n--- MULAI MEREKAM VIDEO KE-{sequence_count} UNTUK '{action}' ---")
                
                sequence_folder = os.path.join(action_path, str(sequence_count))
                os.makedirs(sequence_folder, exist_ok=True)
                
                for frame_num in range(sequence_length):
                    ret, record_frame = cap.read()
                    if not ret: break

                    record_frame = cv2.flip(record_frame, 1)
                    record_image, record_results = mediapipe_detection(record_frame, holistic)
                    
                    cv2.putText(record_image, "MEREKAM...", (frame.shape[1] // 2 - 100, frame.shape[0] // 2), 
                                cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 255), 2)
                    cv2.imshow('Kamera Pengumpul Data Dinamis', record_image)
                    cv2.waitKey(1) 
                    
                    keypoints = extract_keypoints(record_results)
                    npy_path = os.path.join(sequence_folder, str(frame_num))
                    np.save(npy_path, keypoints)

                print(f"--- VIDEO KE-{sequence_count} SELESAI DISIMPAN ---")
                sequence_count += 1
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()