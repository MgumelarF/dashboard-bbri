import numpy as np
import streamlit as st
from sklearn.metrics import (
    mean_absolute_error, mean_absolute_percentage_error, mean_squared_error,
)

from config import DATE_COL, TARGET_COL
from core.data_loader import (
    load_dataset, load_feature_columns, load_keras_model,
    load_scaler, load_test_sequences,
)


def inverse_close(values, scaler, columns) -> np.ndarray:
    close_idx = columns.index(TARGET_COL)
    values = np.asarray(values).reshape(-1)
    dummy = np.zeros((len(values), scaler.n_features_in_))
    dummy[:, close_idx] = values
    return scaler.inverse_transform(dummy)[:, close_idx]


def hitung_metrik(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
    }


@st.cache_data(show_spinner="Menghitung prediksi data testing...")
def evaluasi_model(model_key: str) -> dict:
    model = load_keras_model(model_key)
    scaler = load_scaler()
    columns = load_feature_columns()
    X_test, y_test = load_test_sequences(model_key)

    pred_scaled = model.predict(X_test, verbose=0)
    y_pred = inverse_close(pred_scaled, scaler, columns)
    y_true = inverse_close(y_test, scaler, columns)

    return {"y_true": y_true, "y_pred": y_pred, "metrik": hitung_metrik(y_true, y_pred)}


def tanggal_pengujian(y_true):
    df = load_dataset()
    tail = df.tail(len(y_true))
    if np.allclose(tail[TARGET_COL].to_numpy(), y_true, rtol=1e-4, atol=1e-2):
        return tail[DATE_COL].reset_index(drop=True)
    return None


def model_terbaik(metrik_base: dict, metrik_hybrid: dict) -> dict:
    return {
        k: ("Hybrid" if metrik_hybrid[k] < metrik_base[k] else "Baseline")
        for k in metrik_base
    }


def sentimen_meningkatkan_akurasi(metrik_base: dict, metrik_hybrid: dict) -> bool:
    return (
        metrik_hybrid["MAE"] < metrik_base["MAE"]
        and metrik_hybrid["RMSE"] < metrik_base["RMSE"]
    )


def ringkasan_sentimen(df_periode) -> dict:
    s = df_periode["sentiment_score"]
    total = len(s)
    return {
        "positif": float((s > 0).sum()) / total * 100,
        "netral": float((s == 0).sum()) / total * 100,
        "negatif": float((s < 0).sum()) / total * 100,
    }