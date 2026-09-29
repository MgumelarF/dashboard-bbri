import streamlit as st

from core.data_loader import info_kekosongan_berita
from core.evaluation import (
    evaluasi_model, model_terbaik, sentimen_meningkatkan_akurasi, tanggal_pengujian,
)
from core.prediction import prediksi_hari_berikutnya
from ui.charts import grafik_gabungan
from ui.formatting import fmt_angka, fmt_persen, fmt_rp, fmt_tanggal


def _teks(nama, nilai):
    return fmt_persen(nilai) if nama == "MAPE" else fmt_rp(nilai)


def tampilkan_perbandingan():
    base = evaluasi_model("baseline")
    hyb = evaluasi_model("hybrid")
    mb, mh = base["metrik"], hyb["metrik"]
    tanggal = tanggal_pengujian(base["y_true"])

    st.subheader("Perbandingan metrik evaluasi")
    for kolom, nama in zip(st.columns(3), ("MAE", "RMSE", "MAPE")):
        with kolom:
            with st.container(border=True):
                selisih = mh[nama] - mb[nama]
                st.metric(
                    f"{nama} hybrid",
                    _teks(nama, mh[nama]),
                    delta=fmt_angka(selisih, signed=True),
                    delta_color="inverse",
                )
                st.caption(f"Baseline: {_teks(nama, mb[nama])}")

    st.subheader("Grafik aktual, baseline, dan hybrid (data testing)")
    st.plotly_chart(
        grafik_gabungan(base["y_true"], base["y_pred"], hyb["y_pred"], tanggal),
        key="grafik_gabungan",
    )

    kiri, kanan = st.columns(2)

    with kiri:
        st.subheader("Kesimpulan")
        terbaik = model_terbaik(mb, mh)
        for nama in ("MAE", "RMSE", "MAPE"):
            st.write(f"{nama} terbaik: **{terbaik[nama]}**")
        if sentimen_meningkatkan_akurasi(mb, mh):
            st.success(
                "Penambahan fitur sentimen berita meningkatkan akurasi prediksi "
                "(MAE dan RMSE lebih kecil)."
            )
        else:
            st.warning(
                "Penambahan fitur sentimen berita tidak meningkatkan akurasi "
                "pada data testing ini."
            )

    with kanan:
        st.subheader("Prediksi harga hari berikutnya")
        pb = prediksi_hari_berikutnya("baseline")
        ph = prediksi_hari_berikutnya("hybrid")
        for kolom, label, p in zip(st.columns(2), ("Baseline", "Hybrid"), (pb, ph)):
            with kolom:
                with st.container(border=True):
                    st.metric(
                        label,
                        fmt_rp(p["harga"]),
                        delta=fmt_angka(p["harga"] - p["harga_terakhir"], signed=True),
                        delta_color="off",
                    )
        st.caption(
            "Prediksi untuk hari perdagangan setelah data terakhir "
            f"({fmt_tanggal(ph['tanggal_data_terakhir'])}, penutupan "
            f"{fmt_rp(ph['harga_terakhir'])}). Angka kecil di bawah prediksi adalah "
            "selisihnya terhadap penutupan tersebut."
        )
        info_kosong = info_kekosongan_berita()
        if info_kosong["hari_kosong_beruntun"] > 0:
            st.caption(
                "Catatan: 5 hari data terakhir yang digunakan untuk simulasi ini "
                "tidak memiliki data berita (data sentimen terakhir tersedia "
                f"{fmt_tanggal(info_kosong['tanggal_berita_terakhir'])}), sehingga "
                "kontribusi fitur sentimen pada prediksi di atas minimal."
            )