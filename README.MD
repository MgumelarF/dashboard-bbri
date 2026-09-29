# Dashboard Prediksi Harga Saham BBRI

Dashboard interaktif untuk membandingkan model Bi-LSTM baseline (data harga saja)
dengan Bi-LSTM hybrid (data harga ditambah sentimen berita) pada saham
PT Bank Rakyat Indonesia (Persero) Tbk. (BBRI.JK). Dibuat sebagai bagian dari
penelitian skripsi dengan kerangka CRISP-DM.

## Fitur

- Analisis model baseline: grafik aktual vs prediksi, metrik evaluasi, konfigurasi model
- Analisis model hybrid: tampilan yang sama untuk model dengan fitur sentimen berita
- Perbandingan: metrik MAE, RMSE, MAPE kedua model, grafik gabungan, kesimpulan,
  dan simulasi prediksi harga untuk hari perdagangan setelah data terakhir

## Struktur proyek

```text
dashboard_bbri/
├── artifacts/        model, scaler, dataset, dan data sequence testing
├── core/             pemuatan artifacts, evaluasi, dan prediksi
├── ui/               grafik, format angka, dan halaman dashboard
├── app.py            titik masuk aplikasi
├── config.py         konfigurasi path, fitur, dan hyperparameter
└── requirements.txt  daftar dependensi
```

## Menjalankan secara lokal

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run app.py
```

## Catatan

- Data testing berjumlah 224 sample dengan panjang sequence 5 hari.
- Model dan scaler dilatih di Google Colab, sehingga versi library pada
  `requirements.txt` dikunci agar sama dengan lingkungan pelatihan.
- Prediksi harga hari berikutnya dihitung dari lima hari data terakhir pada dataset,
  bukan dari data pasar real-time.
- Dashboard ini adalah hasil penelitian akademik dan bukan saran investasi.

## Penulis

Muhammad Gumelar Fajari — Informatika, Universitas Gunadarma, 2026