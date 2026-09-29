import pandas as pd
import streamlit as st

from config import DATE_COL, MODELS, SEQ_LENGTH, TARGET_COL
from core.data_loader import (
    load_dataset, load_feature_columns, load_keras_model, load_scaler,
)
from core.evaluation import inverse_close


@st.cache_data(show_spinner="Memprediksi hari perdagangan berikutnya...")
def prediksi_hari_berikutnya(model_key: str) -> dict:
    cfg = MODELS[model_key]
    model = load_keras_model(model_key)
    scaler = load_scaler()
    columns = load_feature_columns()
    df = load_dataset()

    df_last = df.tail(SEQ_LENGTH)
    scaled = scaler.transform(df_last[columns])
    df_scaled = pd.DataFrame(scaled, columns=columns)

    fitur = cfg["features"]
    X = df_scaled[fitur].to_numpy().reshape(1, SEQ_LENGTH, len(fitur))

    pred_scaled = model.predict(X, verbose=0)
    harga = float(inverse_close(pred_scaled, scaler, columns)[0])

    return {
        "harga": harga,
        "harga_terakhir": float(df[TARGET_COL].iloc[-1]),
        "tanggal_data_terakhir": df[DATE_COL].iloc[-1],
    }