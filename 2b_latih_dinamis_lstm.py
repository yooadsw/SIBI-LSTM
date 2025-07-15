import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
from keras.optimizers import Adam

# --- PENGATURAN AWAL ---
DATA_PATH = "data_sibi_dinamis"
actions = np.array(['J', 'Z', 'Lainnya'])
# Sesuaikan angka ini dengan jumlah video yang Anda rekam
no_sequences_per_action = 100 
no_sequences_lainnya = 200
sequence_length = 30

# --- MEMPROSES DATA SEKUENS ---
print("Memuat dan memproses data sekuensial...")
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    # Sesuaikan jumlah video 'Lainnya' jika berbeda
    num_vids = no_sequences_lainnya if action == 'Lainnya' else no_sequences_per_action
    print(f"Memuat {num_vids} video untuk aksi '{action}'...")
    for sequence in range(num_vids):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

MODELS_DIR = 'models' 
os.makedirs(MODELS_DIR, exist_ok=True)
print("Menyimpan data uji (X_test, y_test) ke file...")
joblib.dump({'X_test': X_test, 'y_test': y_test}, os.path.join(MODELS_DIR, 'test_data_dinamis.pkl'))


print(f"Data berhasil dimuat. Ukuran data latih: {X_train.shape}, Ukuran data uji: {X_test.shape}")

# --- MEMBANGUN ARSITEKTUR MODEL LSTM ---
print("Membangun arsitektur model LSTM...")
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()

# --- MELATIH MODEL ---
print("\nMemulai Pelatihan Model...")
# Simpan riwayat pelatihan ke dalam variabel 'history'
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[tb_callback])

# --- EVALUASI & SIMPAN MODEL ---
print("\nMengevaluasi Model pada data uji...")
model.evaluate(X_test, y_test)

MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

print("\nMenyimpan Model dan Riwayat Pelatihan...")
# Simpan model dalam format .keras
model.save(os.path.join(MODELS_DIR, 'model_dinamis_jz.keras'))

# Simpan pemetaan label
joblib.dump(label_map, os.path.join(MODELS_DIR, 'label_map_dinamis.pkl'))

# Simpan objek history yang berisi data akurasi dan loss per epoch
joblib.dump(history.history, os.path.join(MODELS_DIR, 'history_dinamis.pkl'))

print(f"Model dinamis dan riwayat pelatihan berhasil disimpan di folder '{MODELS_DIR}'.")
print("Pelatihan model dinamis selesai!")