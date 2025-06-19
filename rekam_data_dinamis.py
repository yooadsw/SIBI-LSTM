import cv2
import numpy as np
import os
import mediapipe as mp

DATA_PATH = "data_sibi_dinamis"
actions = np.array(['J', 'Z', 'Lainnya'])
no_sequences = 30
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

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        if not os.path.exists(os.path.join(DATA_PATH, action)):
            os.makedirs(os.path.join(DATA_PATH, action))

        cv2.namedWindow('Kamera Pengumpul Data Dinamis')
        cv2.moveWindow('Kamera Pengumpul Data Dinamis', 80, 80)
        
        print(f'\nMempersiapkan untuk merekam aksi: {action}')
        cv2.waitKey(2000)

        for sequence in range(no_sequences):
            sequence_folder = os.path.join(DATA_PATH, action, str(sequence))
            if not os.path.exists(sequence_folder):
                os.makedirs(sequence_folder)
                
            print(f"Rekaman ke-{sequence+1} untuk aksi '{action}' dimulai dalam 3 detik...")
            cv2.waitKey(3000)

            for frame_num in range(sequence_length):
                success, frame = cap.read()
                if not success:
                    break
                
                image, results = mediapipe_detection(frame, holistic)
                
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                
                info_text = f'Aksi: {action} | Video: {sequence+1}/{no_sequences} | Frame: {frame_num+1}/{sequence_length}'
                cv2.putText(image, info_text, (15, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('Kamera Pengumpul Data Dinamis', image)
                
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(sequence_folder, str(frame_num))
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()
                    
            print(f"Rekaman ke-{sequence+1} untuk aksi '{action}' selesai.")

    cap.release()
    cv2.destroyAllWindows()