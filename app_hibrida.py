import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
from keras.models import load_model
from collections import deque

st.set_page_config(page_title="Penerjemah SIBI Hibrida", layout="wide")
st.title("🤟 Penerjemah Isyarat SIBI Hibrida (Statis + Dinamis)")

# ----------------- FUNGSI-FUNGSI HELPER -----------------

@st.cache_resource
def load_all_models():
    model_statis = joblib.load('models/model_statis.pkl')
    scaler_statis = joblib.load('models/scaler_statis.pkl')
    le_statis = joblib.load('models/label_encoder_statis.pkl')
    model_dinamis = load_model('models/model_dinamis_jz.h5')
    return model_statis, scaler_statis, le_statis, model_dinamis

mp_holistic = mp.solutions.holistic

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

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def extract_geometric_features(landmarks):
    wrist_coords = np.array(landmarks[63:66]) if np.any(landmarks[63:126]) else np.array(landmarks[0:3])
    normalized_landmarks = []
    
    for i in range(0, len(landmarks), 3):
        normalized_landmarks.extend(np.array(landmarks[i:i+3]) - wrist_coords)
        
    features = []
    hand_landmarks = normalized_landmarks[63:126] if np.any(landmarks[63:126]) else normalized_landmarks[0:63]
    
    for i in [4, 8, 12, 16, 20]:
        point = np.array(hand_landmarks[i*3 : i*3+3])
        features.append(np.linalg.norm(point))
        
    for i in range(5):
        base_joint = np.array(hand_landmarks[(i*4+1)*3 : (i*4+1)*3+2])
        pip_joint  = np.array(hand_landmarks[(i*4+2)*3 : (i*4+2)*3+2])
        dip_joint  = np.array(hand_landmarks[(i*4+3)*3 : (i*4+3)*3+2])
        tip_joint  = np.array(hand_landmarks[(i*4+4)*3 : (i*4+4)*3+2])
        
        angle1 = calculate_angle(base_joint, pip_joint, dip_joint)
        features.append(angle1)
        angle2 = calculate_angle(pip_joint, dip_joint, tip_joint)
        features.append(angle2)
        
    return features

# ----------------- INISIALISASI APLIKASI -----------------

try:
    model_statis, scaler_statis, le_statis, model_dinamis = load_all_models()
    st.success("Semua model berhasil dimuat!")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

actions_dinamis = np.array(['J', 'Z', 'Lainnya'])
sequence_length = 30
confidence_threshold = 0.85

if 'sequence_buffer' not in st.session_state:
    st.session_state.sequence_buffer = deque(maxlen=sequence_length)
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = "..."

run = st.checkbox('Jalankan Kamera')
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while run and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Tidak bisa membaca frame dari kamera.")
            break

        image, results = mediapipe_detection(frame, holistic)
        keypoints = extract_keypoints(results)
        st.session_state.sequence_buffer.append(keypoints)

        if len(st.session_state.sequence_buffer) == sequence_length:
            
            # Prediksi Dinamis
            res_dinamis = model_dinamis.predict(np.expand_dims(list(st.session_state.sequence_buffer), axis=0))[0]
            pred_dinamis = actions_dinamis[np.argmax(res_dinamis)]
            confidence = res_dinamis[np.argmax(res_dinamis)]

            # Logika Hibrida
            if (pred_dinamis == 'J' or pred_dinamis == 'Z') and confidence > confidence_threshold:
                st.session_state.prediction_result = f"{pred_dinamis} ({confidence*100:.0f}%)"
            else:
                # Prediksi Statis
                try:
                    last_frame_keypoints = st.session_state.sequence_buffer[-1]
                    if np.any(last_frame_keypoints):
                        features_statis = extract_geometric_features(last_frame_keypoints)
                        scaled_features = scaler_statis.transform([features_statis])
                        pred_statis_encoded = model_statis.predict(scaled_features)
                        st.session_state.prediction_result = le_statis.inverse_transform(pred_statis_encoded)[0]
                    else:
                        st.session_state.prediction_result = "..."
                except Exception as e:
                    st.session_state.prediction_result = "Error Statis"

        cv2.putText(image, st.session_state.prediction_result, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2, cv2.LINE_AA)
        FRAME_WINDOW.image(image, channels="BGR")

cap.release()
cv2.destroyAllWindows()