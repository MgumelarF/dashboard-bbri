import streamlit as st

from config import DATE_COL, SEQ_LENGTH
from core.data_loader import load_dataset
from ui.analysis_page import tampilkan_analisis
from ui.comparison_page import tampilkan_perbandingan
from ui.formatting import fmt_angka, fmt_tanggal

st.set_page_config(
    page_title="Prediksi Harga Saham BBRI", page_icon="📈", layout="wide",
)

st.title("Dashboard prediksi harga saham BBRI")
st.caption("Bi-LSTM baseline dan hybrid sentimen berita")

tab_base, tab_hybrid, tab_banding = st.tabs(
    ["Analisis baseline", "Analisis hybrid", "Perbandingan"]
)

with tab_base:
    tampilkan_analisis("baseline")
with tab_hybrid:
    tampilkan_analisis("hybrid")
with tab_banding:
    tampilkan_perbandingan()

df = load_dataset()
st.divider()
kiri, kanan = st.columns([3, 1])
kiri.caption(
    f"BBRI.JK, {fmt_tanggal(df[DATE_COL].min())} sampai "
    f"{fmt_tanggal(df[DATE_COL].max())}, {fmt_angka(len(df), 0)} baris, "
    f"sequence {SEQ_LENGTH} hari"
)
kanan.caption("© 2026 Muhammad Gumelar Fajari")