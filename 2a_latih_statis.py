import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

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

print("Memulai Pelatihan Model Statis...")

NAMA_FILE_CSV = 'data_sibi_statis.csv'
print(f"Membaca data dari {NAMA_FILE_CSV}...")
df_raw = pd.read_csv(NAMA_FILE_CSV, header=None)

print("Melakukan Feature Engineering...")
X_raw = df_raw.iloc[:, 1:]
y_raw = df_raw.iloc[:, 0]

X_features = X_raw.apply(lambda row: extract_geometric_features(row.tolist()), axis=1, result_type='expand')
print(f"Dataset baru dengan {X_features.shape[1]} fitur geometris berhasil dibuat.")

le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Melatih model RandomForestClassifier...")
model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

print("\nMengevaluasi performa model...")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi Model Statis pada Data Uji: {accuracy * 100:.2f}%")
print("\nLaporan Klasifikasi Lengkap:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

MODELS_DIR = 'models'
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

joblib.dump(model, os.path.join(MODELS_DIR, 'model_statis.pkl'))
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler_statis.pkl'))
joblib.dump(le, os.path.join(MODELS_DIR, 'label_encoder_statis.pkl'))

print(f"\nModel statis dan komponennya berhasil disimpan di folder '{MODELS_DIR}'.")
print("Pelatihan model statis selesai!")