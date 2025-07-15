import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import tensorflow as tf
from keras.models import load_model
from keras.layers import Input, LSTM, Dense, Dropout
from keras.models import Sequential
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import threading
import time
import os

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

# --- Custom Model Loader ---
def load_model_with_custom_objects(model_path):
    """Load model with custom objects to handle compatibility issues"""
    try:
        # Method 1: Load with custom objects
        model = load_model(model_path, custom_objects={
            'Input': Input,
            'LSTM': LSTM,
            'Dense': Dense,
            'Dropout': Dropout
        })
        return model
    except Exception as e1:
        st.warning(f"Method 1 failed: {str(e1)}")
        
        try:
            # Method 2: Recreate model architecture
            model = Sequential()
            model.add(Input(shape=(30, 126)))  # Update batch_shape to shape
            model.add(LSTM(64, return_sequences=True, activation='relu'))
            model.add(LSTM(128, return_sequences=True, activation='relu'))
            model.add(LSTM(64, return_sequences=False, activation='relu'))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(3, activation='softmax'))
            
            # Try to load weights
            model.load_weights(model_path)
            return model
        except Exception as e2:
            st.warning(f"Method 2 failed: {str(e2)}")
            
            try:
                # Method 3: Load with compile=False
                model = load_model(model_path, compile=False)
                return model
            except Exception as e3:
                st.error(f"All methods failed: {str(e3)}")
                return None

@st.cache_resource
def load_all_models():
    """Load all models with error handling"""
    try:
        # Load static models
        model_statis = joblib.load('model_statis.pkl')
        scaler_statis = joblib.load('scaler_statis.pkl')
        le_statis = joblib.load('label_encoder_statis.pkl')
        
        # Load dynamic model with custom loader
        model_dinamis = load_model_with_custom_objects('model_dinamis_jz.keras')
        
        if model_dinamis is None:
            raise Exception("Failed to load dynamic model")
        
        return model_statis, scaler_statis, le_statis, model_dinamis
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        raise e

# --- Alternative Model Loader untuk H5 format ---
def convert_keras_to_h5():
    """Convert .keras model to .h5 format if needed"""
    try:
        if os.path.exists('model_dinamis_jz.keras') and not os.path.exists('model_dinamis_jz.h5'):
            # Load the .keras model
            model = tf.keras.models.load_model('model_dinamis_jz.keras')
            # Save as .h5
            model.save('model_dinamis_jz.h5')
            st.success("Model converted to H5 format")
            return 'model_dinamis_jz.h5'
        elif os.path.exists('model_dinamis_jz.h5'):
            return 'model_dinamis_jz.h5'
        else:
            return 'model_dinamis_jz.keras'
    except:
        return 'model_dinamis_jz.keras'

# --- Backup Model Creation ---
def create_backup_model():
    """Create a simple backup model if main model fails"""
    model = Sequential([
        Input(shape=(30, 126)),
        LSTM(64, return_sequences=True, activation='relu'),
        LSTM(128, return_sequences=True, activation='relu'),
        LSTM(64, return_sequences=False, activation='relu'),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

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

# --- Global Variables untuk Threading ---
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = "Menunggu..."
if 'confidence_score' not in st.session_state:
    st.session_state.confidence_score = 0.0
if 'hand_detected' not in st.session_state:
    st.session_state.hand_detected = False
if 'model_type' not in st.session_state:
    st.session_state.model_type = "Standby"

# --- WebRTC Video Processor ---
class SIBIVideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        
        # Load models
        try:
            self.model_statis, self.scaler_statis, self.le_statis, self.model_dinamis = load_all_models()
            self.models_loaded = True
            st.success("✅ Models loaded successfully")
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            # Try to create backup model
            try:
                self.model_dinamis = create_backup_model()
                st.warning("⚠️ Using backup model for dynamic predictions")
                self.models_loaded = False  # Set to False to avoid dynamic predictions
            except:
                self.models_loaded = False
        
        # Configuration
        self.actions_dinamis = np.array(['J', 'Z', 'Lainnya'])
        self.sequence_length = 30
        self.confidence_threshold = 0.98
        self.prediction_interval = 5
        self.frame_counter = 0
        self.sequence_buffer = deque(maxlen=self.sequence_length)
        
        # Thread lock
        self.lock = threading.Lock()
    
    def transform(self, frame):
        # Convert av.VideoFrame to numpy array
        img = frame.to_ndarray(format="bgr24")
        
        if not self.models_loaded:
            cv2.putText(img, "Warning: Using limited functionality", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Flip frame horizontally for mirror effect
        img = cv2.flip(img, 1)
        
        try:
            # MediaPipe detection
            image, results = mediapipe_detection(img, self.holistic)
            
            # Initialize variables
            hand_detected = False
            
            # Draw landmarks
            if results.right_hand_landmarks:
                hand_detected = True
                mp_drawing.draw_landmarks(
                    image, results.right_hand_landmarks, 
                    mp_holistic.HAND_CONNECTIONS
                )
            elif results.left_hand_landmarks:
                hand_detected = True
                mp_drawing.draw_landmarks(
                    image, results.left_hand_landmarks, 
                    mp_holistic.HAND_CONNECTIONS
                )
            
            # Update session state
            with self.lock:
                st.session_state.hand_detected = hand_detected
            
            # Extract keypoints and add to buffer
            keypoints = extract_keypoints(results)
            self.sequence_buffer.append(keypoints)
            
            # Prediction logic
            self.frame_counter += 1
            if self.frame_counter % self.prediction_interval == 0:
                if self.models_loaded:
                    self._perform_prediction()
                else:
                    self._static_only_prediction()
            
            # Draw prediction on frame
            prediction_text = st.session_state.prediction_result
            confidence = st.session_state.confidence_score
            model_type = st.session_state.model_type
            
            # Main prediction display
            cv2.putText(image, f"Prediksi: {prediction_text}", (15, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)
            
            # Confidence and model type
            cv2.putText(image, f"Confidence: {confidence:.1f}%", (15, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(image, f"Model: {model_type}", (15, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Hand detection status
            hand_status = "Terdeteksi" if hand_detected else "Tidak Terdeteksi"
            color = (0, 255, 0) if hand_detected else (0, 0, 255)
            cv2.putText(image, f"Tangan: {hand_status}", (15, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
        except Exception as e:
            cv2.putText(image, f"Error: {str(e)}", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return image
    
    def _perform_prediction(self):
        """Perform prediction using both dynamic and static models"""
        try:
            if len(self.sequence_buffer) == self.sequence_length:
                # Dynamic model prediction
                input_data = np.expand_dims(list(self.sequence_buffer), axis=0)
                
                # Ensure input data is the right shape and type
                input_data = input_data.astype(np.float32)
                
                # Make prediction with error handling
                try:
                    res_dinamis = self.model_dinamis.predict(input_data, verbose=0)[0]
                    
                    pred_dinamis = self.actions_dinamis[np.argmax(res_dinamis)]
                    confidence = res_dinamis[np.argmax(res_dinamis)]
                    
                    # Update session state with thread lock
                    with self.lock:
                        if (pred_dinamis in ['J', 'Z']) and confidence > self.confidence_threshold:
                            st.session_state.prediction_result = pred_dinamis
                            st.session_state.confidence_score = confidence * 100
                            st.session_state.model_type = "Dinamis"
                        else:
                            # Static model prediction
                            self._static_prediction()
                            
                except Exception as pred_error:
                    # If dynamic prediction fails, use static only
                    with self.lock:
                        st.session_state.model_type = "Static Only"
                    self._static_prediction()
                    
        except Exception as e:
            with self.lock:
                st.session_state.prediction_result = "Error"
                st.session_state.confidence_score = 0.0
                st.session_state.model_type = "Error"
    
    def _static_only_prediction(self):
        """Perform only static prediction when dynamic model fails"""
        try:
            self._static_prediction()
        except Exception:
            with self.lock:
                st.session_state.prediction_result = "Static Error"
                st.session_state.confidence_score = 0.0
                st.session_state.model_type = "Error"
    
    def _static_prediction(self):
        """Perform static model prediction"""
        try:
            last_frame_keypoints = self.sequence_buffer[-1]
            
            # Try right hand first, then left hand
            if np.any(last_frame_keypoints[63:]):
                single_hand_keypoints = last_frame_keypoints[63:]
            elif np.any(last_frame_keypoints[:63]):
                single_hand_keypoints = last_frame_keypoints[:63]
            else:
                single_hand_keypoints = None
            
            if single_hand_keypoints is not None:
                features_statis = extract_geometric_features(single_hand_keypoints)
                scaled_features = self.scaler_statis.transform([features_statis])
                pred_statis_encoded = self.model_statis.predict(scaled_features)
                prediction = self.le_statis.inverse_transform(pred_statis_encoded)[0]
                
                st.session_state.prediction_result = prediction
                st.session_state.confidence_score = 85.0  # Placeholder confidence
                st.session_state.model_type = "Statis"
            else:
                st.session_state.prediction_result = "..."
                st.session_state.confidence_score = 0.0
                st.session_state.model_type = "Menunggu"
        except Exception:
            st.session_state.prediction_result = "Error"
            st.session_state.confidence_score = 0.0
            st.session_state.model_type = "Error"

# --- Sidebar ---
with st.sidebar:
    st.title("🤟 SIBI Translator")
    st.markdown("**Sistem Penerjemah Isyarat Bahasa Indonesia**")
    st.markdown("---")
    
    # Model Loading Section
    model_status = st.empty()
    
    # Load Models Status
    try:
        # Try to convert model format first
        model_path = convert_keras_to_h5()
        st.info(f"Using model: {model_path}")
        
        model_statis, scaler_statis, le_statis, model_dinamis = load_all_models()
        model_status.success("✅ Semua Model Berhasil Dimuat")
        
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
        model_status.error(f"❌ Error memuat model: {str(e)}")
        st.info("🔄 Aplikasi akan berjalan dengan fungsionalitas terbatas")
    
    st.markdown("---")
    
    # Version Information
    with st.expander("🔧 Version Info"):
        st.write(f"TensorFlow: {tf.__version__}")
        st.write(f"Python: {os.sys.version}")
        st.write(f"Streamlit: {st.__version__}")
    
    # Usage Instructions
    with st.expander("📝 Cara Penggunaan"):
        st.markdown("""
        1. **Klik Play** pada video stream
        2. **Izinkan** akses kamera browser
        3. **Posisikan** tangan di depan kamera
        4. **Buat** isyarat alfabet SIBI
        5. **Lihat** hasil prediksi real-time
        """)
    
    # Tips
    with st.expander("💡 Tips Optimal"):
        st.markdown("""
        - Gunakan pencahayaan yang cukup
        - Pastikan background kontras
        - Posisikan tangan jelas di tengah
        - Jaga jarak optimal dari kamera
        - Buat gerakan yang jelas dan stabil
        """)

# --- Main Content Area ---
st.header("🎯 Penerjemahan Real-time")

# Layout utama
col1, col2, col3 = st.columns([1, 4, 1])

with col1:
    st.subheader("📊 Status")
    status_placeholder = st.empty()
    hand_status_placeholder = st.empty()
    model_status_placeholder = st.empty()

with col2:
    st.subheader("🎥 Live Camera Feed")
    
    # WebRTC Configuration
    RTC_CONFIGURATION = RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    })
    
    # WebRTC Streamer
    webrtc_ctx = webrtc_streamer(
        key="sibi-translator",
        video_processor_factory=SIBIVideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": {
                "width": {"min": 640, "ideal": 1280, "max": 1920},
                "height": {"min": 480, "ideal": 720, "max": 1080},
                "frameRate": {"min": 15, "ideal": 30, "max": 60}
            },
            "audio": False
        },
        async_processing=True,
        video_html_attrs={
            "style": {
                "width": "100%",
                "height": "auto",
                "border": "2px solid #4CAF50",
                "border-radius": "10px"
            }
        }
    )

with col3:
    st.subheader("📋 Info")
    info_placeholder = st.empty()

# --- Prediction Results Area ---
st.subheader("🔮 Hasil Prediksi")
col_pred1, col_pred2, col_pred3 = st.columns(3)

with col_pred1:
    prediction_placeholder = st.empty()

with col_pred2:
    confidence_placeholder = st.empty()

with col_pred3:
    model_type_placeholder = st.empty()

# --- Status Updates (Real-time) ---
def update_status():
    """Update status displays"""
    if webrtc_ctx.state.playing:
        status_placeholder.success("🟢 Kamera Aktif")
        
        if st.session_state.hand_detected:
            hand_status_placeholder.success("✅ Tangan Terdeteksi")
        else:
            hand_status_placeholder.warning("❌ Tangan Tidak Terdeteksi")
        
        model_status_placeholder.info(f"🧠 Model: {st.session_state.model_type}")
        info_placeholder.info("🔄 Sistem aktif")
        
        prediction_placeholder.markdown(f"## **{st.session_state.prediction_result}**")
        confidence_placeholder.metric("Confidence", f"{st.session_state.confidence_score:.1f}%")
        model_type_placeholder.metric("Model Type", st.session_state.model_type)
        
    else:
        status_placeholder.info("📷 Kamera Siap")
        hand_status_placeholder.info("⏸️ Standby")
        model_status_placeholder.info("⏸️ Model Standby")
        info_placeholder.info("💤 Klik Play untuk memulai")
        prediction_placeholder.markdown("## **Menunggu...**")
        confidence_placeholder.metric("Confidence", "0.0%")
        model_type_placeholder.metric("Model Type", "Standby")

# Auto-refresh status
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()

current_time = time.time()
if current_time - st.session_state.last_update > 1:
    update_status()
    st.session_state.last_update = current_time

update_status()

# --- Troubleshooting Section ---
with st.expander("🔧 Troubleshooting"):
    st.markdown("""
    **Jika model tidak dapat dimuat:**
    1. Pastikan file model ada di repository
    2. Periksa kompatibilitas versi TensorFlow
    3. Aplikasi akan tetap berjalan dengan fungsionalitas terbatas
    
    **Jika kamera tidak berfungsi:**
    1. Pastikan browser mengizinkan akses kamera
    2. Refresh halaman dan coba lagi
    3. Periksa apakah kamera sedang digunakan aplikasi lain
    4. Gunakan browser Chrome/Firefox untuk kompatibilitas terbaik
    """)

# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🎓 Alfabet SIBI | 💻 Teknologi: WebRTC + MediaPipe + TensorFlow + Scikit-learn + Streamlit"
    "</div>", 
    unsafe_allow_html=True
)