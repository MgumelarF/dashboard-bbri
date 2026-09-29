import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

from config import (
    ARTIFACTS_DIR, COLUMNS_FILE, DATASET_FILE, DATE_COL, MODELS, SCALER_FILE,
)


@st.cache_resource(show_spinner="Memuat model...")
def load_keras_model(model_key: str):
    return load_model(str(ARTIFACTS_DIR / MODELS[model_key]["model_file"]))


@st.cache_resource
def load_scaler():
    return joblib.load(ARTIFACTS_DIR / SCALER_FILE)


@st.cache_data
def load_feature_columns() -> list:
    with open(ARTIFACTS_DIR / COLUMNS_FILE) as f:
        return json.load(f)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS_DIR / DATASET_FILE, parse_dates=[DATE_COL])


@st.cache_data
def load_test_sequences(model_key: str):
    cfg = MODELS[model_key]
    X = np.load(ARTIFACTS_DIR / cfg["x_test"])
    y = np.load(ARTIFACTS_DIR / cfg["y_test"])
    return X, y


@st.cache_data
def info_kekosongan_berita() -> dict:
    df = load_dataset()
    nz = df[df["news_count"] > 0]
    return {
        "tanggal_berita_terakhir": nz[DATE_COL].max(),
        "tanggal_data_terakhir": df[DATE_COL].max(),
        "hari_kosong_beruntun": int((df["news_count"] == 0)[::-1].cumprod().sum()),
    }