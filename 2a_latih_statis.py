import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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

def augment_hand_landmarks(landmarks, noise_factor=0.02, rotation_angle_range=10):
    """
    Melakukan augmentasi data pada landmark tangan
    
    Args:
        landmarks: Array landmark tangan (63 nilai: 21 titik × 3 koordinat)
        noise_factor: Faktor noise untuk augmentasi (default: 0.02)
        rotation_angle_range: Range rotasi dalam derajat (default: ±10°)
    
    Returns:
        List augmented landmarks
    """
    augmented_data = []
    original_landmarks = np.array(landmarks)
    
    # 1. Data asli
    augmented_data.append(landmarks)
    
    # 2. Gaussian Noise Augmentation
    for i in range(2):  # 2 variasi noise
        noise = np.random.normal(0, noise_factor, len(landmarks))
        noisy_landmarks = original_landmarks + noise
        augmented_data.append(noisy_landmarks.tolist())
    
    # 3. Rotation Augmentation (rotasi 2D pada bidang x-y)
    for i in range(2):  # 2 variasi rotasi
        angle = np.random.uniform(-rotation_angle_range, rotation_angle_range)
        angle_rad = np.radians(angle)
        
        # Matrix rotasi 2D
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        rotated_landmarks = []
        for j in range(0, len(landmarks), 3):
            x, y, z = landmarks[j], landmarks[j+1], landmarks[j+2]
            
            # Rotasi pada bidang x-y, z tetap
            x_rot = x * cos_angle - y * sin_angle
            y_rot = x * sin_angle + y * cos_angle
            
            rotated_landmarks.extend([x_rot, y_rot, z])
        
        augmented_data.append(rotated_landmarks)
    
    # 4. Scale Augmentation
    for i in range(2):  # 2 variasi skala
        scale_factor = np.random.uniform(0.9, 1.1)  # Skala ±10%
        scaled_landmarks = (original_landmarks * scale_factor).tolist()
        augmented_data.append(scaled_landmarks)
    
    # 5. Translation Augmentation (translasi kecil)
    for i in range(1):  # 1 variasi translasi
        translation = np.random.uniform(-0.05, 0.05, 3)  # Translasi ±5%
        translated_landmarks = []
        for j in range(0, len(landmarks), 3):
            x, y, z = landmarks[j], landmarks[j+1], landmarks[j+2]
            translated_landmarks.extend([
                x + translation[0], 
                y + translation[1], 
                z + translation[2]
            ])
        augmented_data.append(translated_landmarks)
    
    return augmented_data

def perform_data_augmentation(df_raw, augmentation_factor=8):
    """
    Melakukan augmentasi data pada seluruh dataset
    
    Args:
        df_raw: DataFrame asli
        augmentation_factor: Total jumlah sampel per data asli (termasuk data asli)
    
    Returns:
        DataFrame yang sudah diaugmentasi
    """
    print(f"Melakukan augmentasi data dengan faktor {augmentation_factor}...")
    print(f"Dataset asli: {len(df_raw)} sampel")
    
    augmented_rows = []
    
    for idx, row in df_raw.iterrows():
        label = row.iloc[0]
        landmarks = row.iloc[1:].tolist()
        
        # Augmentasi landmarks
        augmented_landmarks_list = augment_hand_landmarks(landmarks)
        
        # Ambil sesuai augmentation_factor (termasuk data asli)
        for i in range(min(augmentation_factor, len(augmented_landmarks_list))):
            new_row = [label] + augmented_landmarks_list[i]
            augmented_rows.append(new_row)
    
    # Buat DataFrame baru
    columns = df_raw.columns
    df_augmented = pd.DataFrame(augmented_rows, columns=columns)
    
    print(f"Dataset setelah augmentasi: {len(df_augmented)} sampel")
    print(f"Peningkatan data: {len(df_augmented) / len(df_raw):.1f}x")
    
    return df_augmented

print("Memulai Pelatihan Model Statis dengan Data Augmentation...")

NAMA_FILE_CSV = 'data_sibi_statis.csv'
print(f"Membaca data dari {NAMA_FILE_CSV}...")
df_raw = pd.read_csv(NAMA_FILE_CSV, header=None)

# Tampilkan distribusi data asli
print("\nDistribusi Data Asli:")
print(df_raw.iloc[:, 0].value_counts().sort_index())

# Lakukan augmentasi data
USE_AUGMENTATION = True  # Set False jika tidak ingin menggunakan augmentasi
AUGMENTATION_FACTOR = 8  # Jumlah total sampel per data asli

if USE_AUGMENTATION:
    df_processed = perform_data_augmentation(df_raw, AUGMENTATION_FACTOR)
    print("\nDistribusi Data Setelah Augmentasi:")
    print(df_processed.iloc[:, 0].value_counts().sort_index())
else:
    df_processed = df_raw
    print("Augmentasi data dinonaktifkan.")

print("\nMelakukan Feature Engineering...")
X_raw = df_processed.iloc[:, 1:]
y_raw = df_processed.iloc[:, 0]

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

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Simpan confusion matrix untuk visualisasi di Streamlit
MODELS_DIR = 'models'
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# Simpan model dan komponen
joblib.dump(model, os.path.join(MODELS_DIR, 'model_statis.pkl'))
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler_statis.pkl'))
joblib.dump(le, os.path.join(MODELS_DIR, 'label_encoder_statis.pkl'))

# Simpan confusion matrix dan metadata untuk visualisasi
evaluation_data = {
    'confusion_matrix': cm,
    'class_names': le.classes_,
    'accuracy': accuracy,
    'y_test': y_test,
    'y_pred': y_pred
}
joblib.dump(evaluation_data, os.path.join(MODELS_DIR, 'evaluation_statis.pkl'))

print(f"\nModel statis dan data evaluasi berhasil disimpan di folder '{MODELS_DIR}'.")
print("Pelatihan model statis selesai!")