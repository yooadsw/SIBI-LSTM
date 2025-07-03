import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
from keras.models import load_model
from collections import deque
import tensorflow as tf

# --- Konfigurasi Halaman & Styling ---
st.set_page_config(
    page_title="SIBI Translator", 
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    /* Background dan theme utama */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
    }
    
    /* Header styling */
    .header-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(45deg, #FFD700, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.8);
        margin-bottom: 1rem;
    }
    
    /* Stats cards */
    .stats-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .stat-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        flex: 1;
        text-align: center;
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FFD700;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.7);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Control panel */
    .control-panel {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 2rem;
    }
    
    /* Video container */
    .video-container {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 15px;
        padding: 1rem;
        border: 2px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Prediction display */
    .prediction-display {
        background: linear-gradient(45deg, #ff6b6b, #ff8e8e);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .prediction-text {
        font-size: 2rem;
        font-weight: 700;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Status indicators */
    .status-success {
        background: linear-gradient(45deg, #4ecdc4, #44a08d);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .status-error {
        background: linear-gradient(45deg, #ff6b6b, #ff8e8e);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #FFD700, #FFA500);
        color: #333;
        font-weight: 700;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 215, 0, 0.4);
    }
    
    /* Checkbox styling */
    .stCheckbox > label {
        font-size: 1.2rem;
        font-weight: 600;
        color: white;
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #FFD700;
        margin: 1rem 0;
    }
    
    .info-box h4 {
        color: #FFD700;
        margin-bottom: 0.5rem;
    }
    
    .info-box p {
        color: rgba(255, 255, 255, 0.8);
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.markdown("""
<div class="header-container">
    <h1 class="main-title">🤟 SIBI Translator</h1>
    <p class="subtitle">Sistem Penerjemah Isyarat Bahasa Indonesia Berbasis AI</p>
    <p class="subtitle">Teknologi Hibrida: Model Statis + Dinamis</p>
</div>
""", unsafe_allow_html=True)

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

# --- Sidebar dengan Informasi ---
with st.sidebar:
    st.markdown("### 📊 Informasi Sistem")
    
    # Status Model
    try:
        model_statis, scaler_statis, le_statis, model_dinamis = load_all_models()
        st.markdown('<div class="status-success">✅ Semua Model Aktif</div>', unsafe_allow_html=True)
        
        # Model Info
        st.markdown("""
        <div class="info-box">
            <h4>🔥 Model Statis</h4>
            <p>Random Forest untuk 24 huruf diam (A-I, K-Y)</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h4>⚡ Model Dinamis</h4>
            <p>LSTM untuk huruf bergerak (J, Z)</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.markdown(f'<div class="status-error">❌ Error: {str(e)}</div>', unsafe_allow_html=True)
        st.stop()
    
    st.markdown("---")
    
    # Cara Penggunaan
    st.markdown("""
    ### 📝 Cara Penggunaan
    
    1. **Centang** kotak "Jalankan Kamera"
    2. **Posisikan** tangan di depan kamera
    3. **Buat** isyarat alfabet SIBI
    4. **Lihat** hasil prediksi real-time
    
    ### 🎯 Tips Optimal
    - Gunakan pencahayaan yang cukup
    - Pastikan background kontras
    - Posisikan tangan jelas di tengah
    - Jaga jarak optimal dari kamera
    """)

# --- Main Content Area ---
col1, col2 = st.columns([2, 1])

with col1:
    # Control Panel
    st.markdown("""
    <div class="control-panel">
        <h3 style="color: white; margin-bottom: 1rem;">🎮 Panel Kontrol</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Camera Control
    run = st.checkbox('🎥 Jalankan Kamera', value=False)
    
    # Video Display
    st.markdown('<div class="video-container">', unsafe_allow_html=True)
    FRAME_WINDOW = st.image([])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Prediction Display
    st.markdown("""
    <div class="prediction-display">
        <h3 style="color: white; margin-bottom: 1rem;">🔮 Hasil Prediksi</h3>
        <div class="prediction-text" id="prediction-result">Menunggu...</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats Display
    st.markdown("""
    <div class="stats-container">
        <div class="stat-card">
            <div class="stat-number">24</div>
            <div class="stat-label">Huruf Statis</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">2</div>
            <div class="stat-label">Huruf Dinamis</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Status Indicators
    st.markdown("### 📡 Status Real-time")
    status_placeholder = st.empty()
    confidence_placeholder = st.empty()

# --- Inisialisasi Session State ---
actions_dinamis = np.array(['J', 'Z', 'Lainnya'])
sequence_length = 30
confidence_threshold = 0.98 
prediction_interval = 5 
frame_counter = 0

if 'sequence_buffer' not in st.session_state:
    st.session_state.sequence_buffer = deque(maxlen=sequence_length)
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = "..."

# --- Main Camera Loop (LOGIKA TIDAK DIUBAH) ---
if run:
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while run and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                status_placeholder.error("⚠️ Gagal membaca frame dari kamera")
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
                        st.session_state.prediction_result = f"{pred_dinamis} ({confidence*100:.0f}%)"
                        confidence_placeholder.success(f"🎯 Tingkat Keyakinan: {confidence*100:.1f}%")
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
                            st.session_state.prediction_result = "Error Statis"
                            confidence_placeholder.error("❌ Error dalam prediksi")

            # Tampilkan hasil prediksi di frame
            cv2.putText(image, f"Prediksi: {st.session_state.prediction_result}", (15, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(image, f"Deteksi: {hand_label}", (15, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Update prediction display dengan JavaScript
            prediction_text = st.session_state.prediction_result
            st.markdown(f"""
            <script>
                document.getElementById('prediction-result').innerText = '{prediction_text}';
            </script>
            """, unsafe_allow_html=True)
            
            FRAME_WINDOW.image(image, channels="BGR")
            
            # Status update
            status_placeholder.success("🟢 Kamera Aktif - Sistem Berjalan")

    if 'cap' in locals() and cap.isOpened():
        cap.release()
    status_placeholder.info("🔴 Kamera Dimatikan")
else:
    status_placeholder.info("📷 Kamera Siap - Klik checkbox untuk memulai")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: rgba(255, 255, 255, 0.6); margin-top: 2rem;">
    <p>🎓  Alfabet SIBI</p>
    <p>💻 Teknologi: MediaPipe + TensorFlow + Scikit-learn + Streamlit</p>
</div>
""", unsafe_allow_html=True)