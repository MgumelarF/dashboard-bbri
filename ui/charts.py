import numpy as np
import plotly.graph_objects as go

WARNA = {"aktual": "#0051FF", "baseline": "#DC2626", "hybrid": "#16A34A"}


def _sumbu_x(n, tanggal):
    if tanggal is not None:
        return tanggal, "Tanggal"
    return np.arange(1, n + 1), "Hari ke- (data testing)"


def _layout(fig, judul_x):
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title=judul_x,
        yaxis_title="Harga penutupan (Rp)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    return fig


def grafik_prediksi(y_true, y_pred, label, tanggal=None):
    x, judul_x = _sumbu_x(len(y_true), tanggal)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y_true, name="Harga aktual",
        line=dict(color=WARNA["aktual"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y_pred, name=f"Prediksi {label}",
        line=dict(color=WARNA[label.lower()], width=2, dash="dash"),
    ))
    return _layout(fig, judul_x)


def grafik_gabungan(y_true, pred_base, pred_hybrid, tanggal=None):
    x, judul_x = _sumbu_x(len(y_true), tanggal)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y_true, name="Harga aktual",
        line=dict(color=WARNA["aktual"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=pred_base, name="Prediksi baseline",
        line=dict(color=WARNA["baseline"], width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=pred_hybrid, name="Prediksi hybrid",
        line=dict(color=WARNA["hybrid"], width=2, dash="dot"),
    ))
    return _layout(fig, judul_x)


def grafik_sentimen(tanggal, sentiment_score, news_count):
    warna = [
        "#16A34A" if v > 0 else ("#DC2626" if v < 0 else "#9CA3AF")
        for v in sentiment_score
    ]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=tanggal, y=sentiment_score, name="Skor sentimen",
        marker_color=warna,
    ))
    fig.add_trace(go.Scatter(
        x=tanggal, y=news_count, name="Jumlah berita",
        line=dict(color="#7C3AED", width=1.5), yaxis="y2",
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="Skor sentimen", range=[-1.1, 1.1]),
        yaxis2=dict(title="Jumlah berita", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        bargap=0.25,
    )
    return fig