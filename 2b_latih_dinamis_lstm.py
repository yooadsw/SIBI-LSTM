import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

DATA_PATH = "data_sibi_dinamis"
actions = np.array(['J', 'Z', 'Lainnya'])
no_sequences = 30
sequence_length = 30

print("Memulai Pelatihan Model Dinamis (LSTM)...")

print("Memuat dan memproses data sekuensial...")
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

print(f"Data berhasil dimuat. Ukuran data latih: {X_train.shape}, Ukuran data uji: {X_test.shape}")

print("Membangun arsitektur model LSTM...")
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, 126)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()

print("\nMemulai Pelatihan Model...")
model.fit(X_train, y_train, epochs=100, callbacks=[tb_callback], validation_data=(X_test, y_test))

print("\nMengevaluasi Model pada data uji...")
model.evaluate(X_test, y_test)

MODELS_DIR = 'models'
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

print("\nMenyimpan Model...")
model.save(os.path.join(MODELS_DIR, 'model_dinamis_jz.keras'))
joblib.dump(label_map, os.path.join(MODELS_DIR, 'label_map_dinamis.pkl'))

print(f"Model dinamis berhasil disimpan di folder '{MODELS_DIR}'.")
print("Pelatihan model dinamis selesai!")