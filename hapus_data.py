import pandas as pd
import os

# --- PENGATURAN ---
# Nama file yang akan dibaca sekaligus diubah
file_to_modify = 'data_sibi_statis.csv'
# Daftar label yang ingin dihapus
labels_to_remove = ['M', 'N', 'P', 'Q']

# Cek dulu apakah filenya ada
if not os.path.exists(file_to_modify):
    print(f"ERROR: File '{file_to_modify}' tidak ditemukan! Tidak ada yang bisa dilakukan.")
else:
    print(f"Membaca data dari: {file_to_modify}")
    # Baca file csv. header=None karena file kita tidak punya baris header.
    df = pd.read_csv(file_to_modify, header=None)

    initial_rows = len(df)
    print(f"Jumlah baris awal: {initial_rows}")
    print(f"Label yang akan dihapus: {labels_to_remove}")

    # Lakukan filtering. Kolom 0 adalah kolom label.
    # Logikanya: Simpan semua baris di mana labelnya TIDAK ADA di dalam list `labels_to_remove`
    df_cleaned = df[~df[0].isin(labels_to_remove)]

    final_rows = len(df_cleaned)
    rows_removed = initial_rows - final_rows

    # Hanya tulis ulang file jika memang ada data yang dihapus
    if rows_removed > 0:
        # Simpan kembali ke file yang SAMA, ini akan menimpa (overwrite) file yang lama.
        df_cleaned.to_csv(file_to_modify, index=False, header=False)
        
        print(f"\nPROSES SELESAI!")
        print(f"File '{file_to_modify}' telah berhasil diperbarui.")
        print(f"{rows_removed} baris telah dihapus.")
    else:
        print("\nTidak ada data dengan label tersebut yang ditemukan. File tidak diubah.")