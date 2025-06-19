import cv2
import mediapipe as mp
import csv
import os
import numpy as np

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

NAMA_FILE_CSV = 'data_sibi_statis.csv'

def extract_keypoints(results):
    if results.right_hand_landmarks:
        hand_landmarks = results.right_hand_landmarks
    elif results.left_hand_landmarks:
        hand_landmarks = results.left_hand_landmarks
    else:
        return np.zeros(21 * 3)

    keypoints = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
    return keypoints

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Tidak bisa membuka webcam.")
        return

    with mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6) as holistic:
        
        print("Program Pengumpul Data Isyarat Statis")
        nama_isyarat = input("Masukkan nama huruf yang akan direkam (misal: A, B, S): ").upper()
        if not nama_isyarat:
            print("Nama isyarat tidak boleh kosong.")
            return

        print(f"\nSiap merekam untuk isyarat '{nama_isyarat}'.")
        print("Arahkan satu tangan Anda ke kamera.")
        print("Tekan tombol 'S' untuk menyimpan data isyarat.")
        print("Tekan tombol 'Q' untuk keluar dari program.")
        
        data_count = 0
        
        while True:
            success, frame = cap.read()
            if not success:
                print("Gagal membaca frame.")
                break

            frame = cv2.flip(frame, 1)
            
            image, results = mediapipe_detection(frame, holistic)

            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            cv2.putText(image, f"Rekam: {nama_isyarat} ({data_count} data)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Pengumpul Data Statis', image)

            key = cv2.waitKey(5) & 0xFF

            if key == ord('s'):
                keypoints = extract_keypoints(results)
                
                row_data = [nama_isyarat] + list(keypoints)
                
                with open(NAMA_FILE_CSV, mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row_data)
                
                data_count += 1
                print(f"Data ke-{data_count} untuk isyarat '{nama_isyarat}' berhasil disimpan!")

            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSelesai. Total {data_count} data untuk '{nama_isyarat}' telah direkam.")

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

if __name__ == "__main__":
    main()