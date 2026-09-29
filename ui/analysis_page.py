import streamlit as st

from config import DATE_COL, HYPERPARAMETER, MODELS
from core.data_loader import info_kekosongan_berita, load_dataset
from core.evaluation import evaluasi_model, ringkasan_sentimen, tanggal_pengujian
from ui.charts import grafik_prediksi, grafik_sentimen
from ui.formatting import fmt_persen, fmt_rp, fmt_tanggal


def _kartu_konfigurasi():
    for kolom, (nama, nilai) in zip(st.columns(2), HYPERPARAMETER.items()):
        with kolom:
            with st.container(border=True):
                st.metric(nama, nilai)
    kolom3, kolom4 = st.columns(2)
    sisa = list(HYPERPARAMETER.items())[2:]
    for kolom, (nama, nilai) in zip((kolom3, kolom4), sisa):
        with kolom:
            with st.container(border=True):
                st.metric(nama, nilai)


def _tabel_rincian(df_periode, y_pred):
    tabel = df_periode[[DATE_COL, "Close", "sentiment_score", "news_count"]].copy()
    tabel["Prediksi"] = y_pred
    tabel = tabel.rename(columns={
        DATE_COL: "Tanggal", "Close": "Close (Rp)",
        "sentiment_score": "Sentimen", "news_count": "Berita",
    })
    tabel = tabel[["Tanggal", "Close (Rp)", "Prediksi", "Sentimen", "Berita"]]
    tabel = tabel.sort_values("Tanggal", ascending=False)
    st.dataframe(
        tabel,
        hide_index=True,
        height=280,
        column_config={
            "Tanggal": st.column_config.DateColumn("Tanggal", format="DD MMM YYYY"),
            "Close (Rp)": st.column_config.NumberColumn("Close (Rp)", format="Rp %,.0f"),
            "Prediksi": st.column_config.NumberColumn("Prediksi (Rp)", format="Rp %,.2f"),
            "Sentimen": st.column_config.NumberColumn("Sentimen", format="%.2f"),
            "Berita": st.column_config.NumberColumn("Berita", format="%d"),
        },
    )
    st.caption(f"Menampilkan seluruh {len(tabel)} data, gulir tabel untuk melihat lainnya.")


def _tampilkan_baseline():
    cfg = MODELS["baseline"]
    hasil = evaluasi_model("baseline")
    tanggal = tanggal_pengujian(hasil["y_true"])

    kiri, kanan = st.columns([2, 1])
    with kiri:
        st.subheader("Grafik aktual vs prediksi (data testing)")
        st.plotly_chart(
            grafik_prediksi(hasil["y_true"], hasil["y_pred"], cfg["label"], tanggal),
            key="grafik_baseline",
        )
        st.subheader("Konfigurasi model")
        for kolom, (nama, nilai) in zip(st.columns(4), HYPERPARAMETER.items()):
            with kolom:
                with st.container(border=True):
                    st.metric(nama, nilai)
        st.caption(f"Fitur masukan ({len(cfg['features'])}): {', '.join(cfg['features'])}")

    with kanan:
        st.subheader("Metrik evaluasi")
        for nama, teks in (
            ("MAE", fmt_rp(hasil["metrik"]["MAE"])),
            ("RMSE", fmt_rp(hasil["metrik"]["RMSE"])),
            ("MAPE", fmt_persen(hasil["metrik"]["MAPE"])),
        ):
            with st.container(border=True):
                st.metric(nama, teks)


def _tampilkan_hybrid():
    cfg = MODELS["hybrid"]
    hasil = evaluasi_model("hybrid")
    tanggal = tanggal_pengujian(hasil["y_true"])
    info_kosong = info_kekosongan_berita()

    df = load_dataset()
    df_periode = df.tail(len(hasil["y_true"])).reset_index(drop=True)

    kiri, kanan = st.columns([1.6, 1])

    with kiri:
        st.subheader("Grafik harga aktual vs prediksi")
        st.plotly_chart(
            grafik_prediksi(hasil["y_true"], hasil["y_pred"], cfg["label"], tanggal),
            key="grafik_hybrid",
        )
        st.subheader("Rincian data harian")
        _tabel_rincian(df_periode, hasil["y_pred"])

    with kanan:
        st.subheader("Konfigurasi model")
        _kartu_konfigurasi()

        st.subheader("Metrik evaluasi")
        m1, m2, m3 = st.columns(3)
        for kolom, (nama, teks) in zip(
            (m1, m2, m3),
            (
                ("MAE", fmt_rp(hasil["metrik"]["MAE"])),
                ("RMSE", fmt_rp(hasil["metrik"]["RMSE"])),
                ("MAPE", fmt_persen(hasil["metrik"]["MAPE"])),
            ),
        ):
            with kolom:
                with st.container(border=True):
                    st.metric(nama, teks)

        st.subheader("Sentimen berita (periode data testing)")
        ringkas = ringkasan_sentimen(df_periode)
        p1, p2, p3 = st.columns(3)
        for kolom, (nama, nilai) in zip(
            (p1, p2, p3),
            (
                ("Positif", ringkas["positif"]),
                ("Netral", ringkas["netral"]),
                ("Negatif", ringkas["negatif"]),
            ),
        ):
            with kolom:
                with st.container(border=True):
                    st.metric(nama, fmt_persen(nilai))
        st.plotly_chart(
            grafik_sentimen(
                df_periode[DATE_COL], df_periode["sentiment_score"], df_periode["news_count"],
            ),
            key="grafik_sentimen",
        )
        st.caption(
            f"Data sentimen tersedia sampai {fmt_tanggal(info_kosong['tanggal_berita_terakhir'])}, "
            f"{info_kosong['hari_kosong_beruntun']} hari sebelum data harga terakhir "
            f"({fmt_tanggal(info_kosong['tanggal_data_terakhir'])}). Hari tanpa data berita "
            "dihitung sebagai skor sentimen netral (0)."
        )


def tampilkan_analisis(model_key: str):
    if model_key == "baseline":
        _tampilkan_baseline()
    else:
        _tampilkan_hybrid()