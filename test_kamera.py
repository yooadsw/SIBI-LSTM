import cv2

print("Mencoba membuka kamera...")

# Coba akses kamera utama (indeks 0)
cap = cv2.VideoCapture(0) 

# Cek apakah kamera berhasil dibuka
if not cap.isOpened():
    print("GAGAL: Kamera tidak bisa dibuka sama sekali.")
    print("Coba cek apakah kamera sedang dipakai aplikasi lain (Zoom, dll) atau coba ganti ke cv2.VideoCapture(1)")
else:
    print("BERHASIL: Kamera terbuka. Menampilkan jendela...")
    print("Tekan tombol 'q' di jendela kamera untuk keluar.")

    # Loop untuk menampilkan video
    while True:
        # Baca satu frame dari kamera
        success, frame = cap.read()

        # Jika gagal membaca frame, hentikan loop
        if not success:
            print("GAGAL: Tidak bisa membaca frame dari kamera.")
            break

        # Tampilkan frame di sebuah jendela bernama "Tes Kamera"
        cv2.imshow("Tes Kamera - Tekan Q untuk Keluar", frame)

        # Tunggu 1 milidetik, dan cek jika tombol 'q' ditekan
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Menutup kamera...")

# Lepaskan kamera dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()