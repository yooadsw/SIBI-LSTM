import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
from keras.models import load_model
from collections import deque
import tensorflow as tf

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="SIBI Translator", 
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Inisialisasi MediaPipe ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


@st.cache_resource
def load_all_models():
    model_statis = joblib.load('models/model_statis.pkl')
    scaler_statis = joblib.load('models/scaler_statis.pkl')
    le_statis = joblib.load('models/label_encoder_statis.pkl')
    model_dinamis = load_model('models/model_dinamis_jz.keras')
    return model_statis, scaler_statis, le_statis, model_dinamis


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
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

def extract_geometric_features(landmarks):
    wrist_coords = np.array(landmarks[0:3])
    normalized_landmarks = []
    for i in range(0, len(landmarks), 3):
        normalized_landmarks.extend(np.array(landmarks[i:i+3]) - wrist_coords)
    
    features = []
    for i in [4, 8, 12, 16, 20]:
        point = np.array(normalized_landmarks[i*3 : i*3+3])
        features.append(np.linalg.norm(point))
        
    for i in range(5):
        base_joint = np.array(normalized_landmarks[(i*4+1)*3 : (i*4+1)*3+2])
        pip_joint  = np.array(normalized_landmarks[(i*4+2)*3 : (i*4+2)*3+2])
        dip_joint  = np.array(normalized_landmarks[(i*4+3)*3 : (i*4+3)*3+2])
        tip_joint  = np.array(normalized_landmarks[(i*4+4)*3 : (i*4+4)*3+2])
        angle1 = calculate_angle(base_joint, pip_joint, dip_joint)
        features.append(angle1)
        angle2 = calculate_angle(pip_joint, dip_joint, tip_joint)
        features.append(angle2)
    return features

# --- Sidebar: Informasi dan Kontrol ---
with st.sidebar:
    st.title("🤟 SIBI Translator")
    st.markdown("**Sistem Penerjemah Isyarat Bahasa Indonesia**")
    st.markdown("---")
    
    # Load Models
    try:
        model_statis, scaler_statis, le_statis, model_dinamis = load_all_models()
        st.success("✅ Semua Model Berhasil Dimuat")
        
        # Model Information
        with st.expander("📊 Informasi Model"):
            st.markdown("**🔥 Model Statis**")
            st.write("- Random Forest Classifier")
            st.write("- 24 huruf diam (A-I, K-Y)")
            st.write("- Fitur geometris tangan")
            
            st.markdown("**⚡ Model Dinamis**")
            st.write("- LSTM Neural Network")
            st.write("- 2 huruf bergerak (J, Z)")
            st.write("- Sequence keypoints")
            
    except Exception as e:
        st.error(f"❌ Error memuat model: {str(e)}")
        st.stop()
    
    # Control Panel
    st.markdown("### 🎮 Kontrol Kamera")
    run = st.checkbox('🎥 Aktifkan Kamera', value=False)
    
    st.markdown("---")
    
    # Usage Instructions
    with st.expander("📝 Cara Penggunaan"):
        st.markdown("""
        1. **Aktifkan** kamera dengan centang checkbox
        2. **Posisikan** tangan di depan kamera
        3. **Buat** isyarat alfabet SIBI
        4. **Lihat** hasil prediksi real-time
        """)
    
    # Tips
    with st.expander("💡 Tips Optimal"):
        st.markdown("""
        - Gunakan pencahayaan yang cukup
        - Pastikan background kontras
        - Posisikan tangan jelas di tengah
        - Jaga jarak optimal dari kamera
        """)
    
    # Statistics
    with st.expander("📈 Statistik Model"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Huruf Statis", "24")
        with col2:
            st.metric("Huruf Dinamis", "2")

# --- Main Content Area ---
st.header("🎯 Penerjemahan Real-time")

# Layout utama dengan 3 kolom
col1, col2, col3 = st.columns([1, 4, 1])

with col1:
    st.subheader("📊 Status")
    status_container = st.container()
    
with col2:
    # Area video utama (tengah dan besar)
    st.subheader("🎥 Live Camera Feed")
    video_container = st.container()
    
    # Hasil prediksi di bawah video
    st.subheader("🔮 Hasil Prediksi")
    prediction_container = st.container()

with col3:
    st.subheader("📋 Info")
    info_container = st.container()

# --- Inisialisasi Session State ---
actions_dinamis = np.array(['J', 'Z', 'Lainnya'])
sequence_length = 30
confidence_threshold = 0.98 
prediction_interval = 5 
frame_counter = 0

if 'sequence_buffer' not in st.session_state:
    st.session_state.sequence_buffer = deque(maxlen=sequence_length)
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = "Menunggu..."

# --- Main Camera Logic ---
if run:
    cap = cv2.VideoCapture(0)
    
    # Container untuk video
    with video_container:
        FRAME_WINDOW = st.image([])
    
    # Container untuk prediksi
    with prediction_container:
        prediction_placeholder = st.empty()
        confidence_placeholder = st.empty()
    
    # Container untuk status
    with status_container:
        status_placeholder = st.empty()
        hand_status_placeholder = st.empty()
    
    # Container untuk info - Initialize once outside the loop
    with info_container:
        info_placeholder = st.empty()
        info_placeholder.info("🔄 Sistem berjalan normal")
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while run and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                status_placeholder.error("⚠️ Gagal membaca frame")
                continue

            frame = cv2.flip(frame, 1)
            image, results = mediapipe_detection(frame, holistic)
            
            # --- Deteksi Tangan dan Gambar Kerangka ---
            hand_label = "Tidak Terdeteksi"
            if results.right_hand_landmarks:
                hand_label = "Terdeteksi"
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            elif results.left_hand_landmarks:
                hand_label = "Terdeteksi"
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            keypoints = extract_keypoints(results)
            st.session_state.sequence_buffer.append(keypoints)

            frame_counter += 1
            if frame_counter % prediction_interval == 0:
                if len(st.session_state.sequence_buffer) == sequence_length:
                    
                    input_data = np.expand_dims(list(st.session_state.sequence_buffer), axis=0)
                    input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
                    res_dinamis = model_dinamis.predict(input_tensor, verbose=0)[0]
                    
                    pred_dinamis = actions_dinamis[np.argmax(res_dinamis)]
                    confidence = res_dinamis[np.argmax(res_dinamis)]

                    if (pred_dinamis in ['J', 'Z']) and confidence > confidence_threshold:
                        st.session_state.prediction_result = pred_dinamis
                        confidence_placeholder.success(f"🎯 Keyakinan: {confidence*100:.1f}%")
                    else:
                        try:
                            last_frame_keypoints = st.session_state.sequence_buffer[-1]
                            
                            if np.any(last_frame_keypoints[63:]):
                                single_hand_keypoints = last_frame_keypoints[63:]
                            elif np.any(last_frame_keypoints[:63]):
                                single_hand_keypoints = last_frame_keypoints[:63]
                            else:
                                single_hand_keypoints = None

                            if single_hand_keypoints is not None:
                                features_statis = extract_geometric_features(single_hand_keypoints)
                                scaled_features = scaler_statis.transform([features_statis])
                                pred_statis_encoded = model_statis.predict(scaled_features)
                                st.session_state.prediction_result = le_statis.inverse_transform(pred_statis_encoded)[0]
                                confidence_placeholder.info("📊 Model Statis Aktif")
                            else:
                                st.session_state.prediction_result = "..."
                                confidence_placeholder.warning("⏳ Menunggu deteksi...")
                        except Exception:
                            st.session_state.prediction_result = "Error"
                            confidence_placeholder.error("❌ Error dalam prediksi")

            # Tampilkan hasil prediksi di frame
            cv2.putText(image, f"Prediksi: {st.session_state.prediction_result}", (15, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3, cv2.LINE_AA)
            
            # Update UI
            FRAME_WINDOW.image(image, channels="BGR")
            
            # Update containers
            with prediction_placeholder.container():
                st.markdown(f"## **{st.session_state.prediction_result}**")
            
            with status_placeholder.container():
                st.success("🟢 Kamera Aktif")
            
            with hand_status_placeholder.container():
                if hand_label == "Terdeteksi":
                    st.success(f"✅ {hand_label}")
                else:
                    st.warning(f"❌ {hand_label}")

    if 'cap' in locals() and cap.isOpened():
        cap.release()
    
    # Update status when camera is stopped
    with status_container:
        st.info("🔴 Kamera Dimatikan")
        
else:
    # Tampilan default ketika kamera tidak aktif
    with video_container:
        st.info("📷 Kamera tidak aktif. Aktifkan kamera dari sidebar untuk memulai.")
    
    with prediction_container:
        st.markdown("## **Menunggu...**")
    
    with status_container:
        st.info("📷 Kamera Siap")
    
    with info_container:
        st.info("💤 Sistem dalam mode standby")

# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🎓 Alfabet SIBI | 💻 Teknologi: MediaPipe + TensorFlow + Scikit-learn + Streamlit"
    "</div>", 
    unsafe_allow_html=True
)