"""Streamlit dashboard — IMS/NASA Bearing Prognostics.

UX priority
-----------
1. Entender em 30 s: status atual + KPIs + card de previsao de falha
2. Timeline com timestamps reais (um snapshot a cada ~10 min por 7 dias)
3. Linha de tendencia + projecao futura (regressao linear)
4. Detalhe do snapshot selecionado: radar vs. normal + histograma de score
5. Explicacao SHAP on-demand

Dataset
-------
IMS/NASA (University of Cincinnati) — Run 2
2004-02-12 a 2004-02-19 | 984 snapshots | 4 rolamentos | 20 kHz
Rolamento 1: falha na pista externa (outer race) ao final do periodo.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must come before any other matplotlib import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.explain import explain
from src.models.autoencoder import AutoEncoderDetector
from src.models.iforest import IForestDetector
from src.models.lof import LOFDetector
from src.models.ocsvm import OCSVMDetector

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_RESULTS = Path("results")
_DATA_FEATURES = Path("data/features/features.parquet")
_THRESHOLD_JSON = _RESULTS / "threshold.json"

_MODEL_THRESHOLD_KEY = {
    "IsolationForest": "iforest",
    "OC-SVM": "ocsvm",
    "LOF": "lof",
    "AutoEncoder": "ae",
}
_MODEL_FILES = {
    "IsolationForest": _RESULTS / "iforest_model.joblib",
    "OC-SVM": _RESULTS / "ocsvm_model.joblib",
    "LOF": _RESULTS / "lof_model.joblib",
    "AutoEncoder": _RESULTS / "ae_model.joblib",
}
_MODEL_CLASSES = {
    "IsolationForest": IForestDetector,
    "OC-SVM": OCSVMDetector,
    "LOF": LOFDetector,
    "AutoEncoder": AutoEncoderDetector,
}
_SLOW_MODELS = {"OC-SVM", "LOF", "AutoEncoder"}

st.set_page_config(
    page_title="IMS Bearing Prognostics",
    page_icon="🔧",
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }
    .status-ok   { background:#1a6a3a; color:#fff; border-radius:8px;
                   padding:12px 20px; text-align:center; font-size:1.1rem; font-weight:700; }
    .status-warn { background:#7d1a1a; color:#fff; border-radius:8px;
                   padding:12px 20px; text-align:center; font-size:1.1rem; font-weight:700; }
    .pred-card   { background:#1e1e1e; border:2px solid #e67e22; border-radius:10px;
                   padding:16px; text-align:center; }
    .pred-card h2 { color:#e67e22; margin:0 0 4px 0; font-size:1.4rem; }
    .pred-card p  { color:#ccc; margin:2px 0; font-size:0.9rem; }
    .ok-card     { background:#1e1e1e; border:2px solid #2ecc71; border-radius:10px;
                   padding:16px; text-align:center; }
    .ok-card h2  { color:#2ecc71; margin:0 0 4px 0; font-size:1.4rem; }
    .ok-card p   { color:#ccc; margin:2px 0; font-size:0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Loaders (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_test_data() -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]] | None:
    x_path = _RESULTS / "X_test.npy"
    y_path = _RESULTS / "y_test.npy"
    meta_path = _RESULTS / "meta_test.parquet"
    if not (x_path.exists() and y_path.exists() and _DATA_FEATURES.exists()):
        return None
    X_test = np.load(x_path)
    y_test = np.load(y_path)
    df_feat = pd.read_parquet(_DATA_FEATURES)
    feature_names = [c for c in df_feat.columns if not c.startswith("_meta_")]
    if meta_path.exists():
        meta_test = pd.read_parquet(meta_path).reset_index(drop=True)
    else:
        meta_test = pd.DataFrame({"_meta_y": y_test})
    return X_test, y_test, meta_test, feature_names


@st.cache_resource
def load_model(model_name: str):
    path = _MODEL_FILES.get(model_name)
    if path is None or not path.exists():
        return None
    return _MODEL_CLASSES[model_name].load(path)


@st.cache_data
def compute_scores(model_name: str, X_bytes: bytes, n_rows: int) -> np.ndarray | None:
    model = load_model(model_name)
    if model is None:
        return None
    X = np.frombuffer(X_bytes, dtype=np.float64).reshape(n_rows, -1)
    return model.score(X)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _default_threshold(scores: np.ndarray, y_test: np.ndarray, model_name: str) -> float:
    if _THRESHOLD_JSON.exists():
        try:
            data = json.loads(_THRESHOLD_JSON.read_text())
            key = _MODEL_THRESHOLD_KEY.get(model_name, model_name.lower())
            if key in data:
                return float(data[key])
        except Exception:
            pass
    normal_mask = y_test == 0
    if normal_mask.any():
        return float(np.percentile(scores[normal_mask], 99))
    return float(np.median(scores))


def _get_timestamps(meta: pd.DataFrame, n: int) -> pd.DatetimeIndex:
    """Return real timestamps from meta if available, otherwise return empty index."""
    if "_meta_timestamp" in meta.columns:
        ts = pd.to_datetime(meta["_meta_timestamp"])
        return pd.DatetimeIndex(ts.values)
    return pd.DatetimeIndex([])


def _predict_failure(
    scores: np.ndarray,
    timestamps: pd.DatetimeIndex,
    threshold: float,
    trend_frac: float = 0.25,
) -> dict | None:
    """Fit a linear trend on the last *trend_frac* of scores and project to threshold crossing.

    Returns a dict with prediction details, or None if trend is flat/decreasing
    or the crossing is already in the past.
    """
    n = len(scores)
    if n < 20 or len(timestamps) != n:
        return None

    n_trend = max(int(n * trend_frac), 5)
    x = np.arange(n_trend, dtype=float)
    y = scores[n - n_trend :]

    slope, intercept = np.polyfit(x, y, 1)
    if slope <= 1e-10:
        return None

    # t where trend line crosses threshold (relative to trend window start)
    t_cross_rel = (threshold - intercept) / slope
    if t_cross_rel <= n_trend:
        return None  # crossing already in past

    abs_cross = (n - n_trend) + t_cross_rel
    extra = abs_cross - n

    if len(timestamps) >= 2:
        dt = (timestamps[-1] - timestamps[-2])
    else:
        return None

    predicted_ts = timestamps[-1] + dt * extra
    hours_away = extra * dt.total_seconds() / 3600

    # Arrays for plotting the trend line and projection
    trend_x_idx = np.arange(n - n_trend, n)
    trend_y_vals = slope * np.arange(n_trend) + intercept

    proj_steps = int(extra) + 5
    proj_x_idx = np.arange(n, n + proj_steps)
    proj_y_vals = slope * np.arange(n_trend, n_trend + proj_steps) + intercept

    return {
        "predicted_ts": predicted_ts,
        "hours_away": hours_away,
        "slope": slope,
        "n_trend": n_trend,
        "trend_x_idx": trend_x_idx,
        "trend_y_vals": trend_y_vals,
        "proj_x_idx": proj_x_idx,
        "proj_y_vals": proj_y_vals,
        "abs_cross": abs_cross,
    }


def _hex_to_rgb(hex_color: str) -> list[str]:
    h = hex_color.lstrip("#")
    return [str(int(h[i : i + 2], 16)) for i in (0, 2, 4)]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _fig_timeline(
    scores: np.ndarray,
    meta: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    y_test: np.ndarray,
    threshold: float,
    selected_idx: int,
    prediction: dict | None,
) -> go.Figure:
    n = len(scores)
    has_ts = len(timestamps) == n
    above = scores >= threshold

    y_min = float(scores.min()) * 0.95
    y_max = float(scores.max()) * 1.05

    fig = go.Figure()

    # Background zone shading
    fig.add_hrect(
        y0=threshold, y1=y_max,
        fillcolor="rgba(231,76,60,0.10)", layer="below", line_width=0,
    )
    fig.add_hrect(
        y0=y_min, y1=threshold,
        fillcolor="rgba(46,204,113,0.07)", layer="below", line_width=0,
    )
    fig.add_annotation(
        text="⚠️ ZONA DE RISCO", x=0.01, xref="paper",
        y=threshold + (y_max - threshold) * 0.55, yref="y",
        font=dict(color="#e74c3c", size=11), showarrow=False, xanchor="left",
    )
    fig.add_annotation(
        text="✅ ZONA SEGURA", x=0.01, xref="paper",
        y=y_min + (threshold - y_min) * 0.25, yref="y",
        font=dict(color="#2ecc71", size=11), showarrow=False, xanchor="left",
    )

    # X-axis values: real timestamps or sequential integers
    x_vals = list(timestamps) if has_ts else list(range(n))

    # Healthy (below threshold) scatter
    mask_ok = ~above
    if mask_ok.any():
        x_ok = [x_vals[i] for i in range(n) if mask_ok[i]]
        hover_ok = [
            f"<b>{timestamps[i].strftime('%d/%m/%Y %H:%M') if has_ts else f'Snapshot #{i}'}</b><br>"
            f"Score: {scores[i]:.4f}<br>Rotulo: {'saudavel' if y_test[i] == 0 else 'degradado'}<br>"
            f"Diagnostico: Normal<extra></extra>"
            for i in range(n) if mask_ok[i]
        ]
        fig.add_trace(go.Scatter(
            x=x_ok, y=scores[mask_ok],
            mode="markers",
            marker=dict(color="#2ecc71", size=5, opacity=0.65),
            name="Score normal",
            customdata=np.where(mask_ok)[0].tolist(),
            hovertemplate=hover_ok,
        ))

    # Anomalous (above threshold) scatter
    if above.any():
        x_ab = [x_vals[i] for i in range(n) if above[i]]
        hover_ab = [
            f"<b>{timestamps[i].strftime('%d/%m/%Y %H:%M') if has_ts else f'Snapshot #{i}'}</b><br>"
            f"Score: {scores[i]:.4f}<br>Rotulo: {'saudavel' if y_test[i] == 0 else 'degradado'}<br>"
            f"Diagnostico: ANOMALIA<extra></extra>"
            for i in range(n) if above[i]
        ]
        fig.add_trace(go.Scatter(
            x=x_ab, y=scores[above],
            mode="markers",
            marker=dict(color="#e74c3c", size=8, symbol="diamond", opacity=0.90,
                        line=dict(width=1, color="#111")),
            name="Anomalia detectada",
            customdata=np.where(above)[0].tolist(),
            hovertemplate=hover_ab,
        ))

    # Threshold line
    fig.add_hline(
        y=threshold, line_dash="dash", line_color="#e74c3c", line_width=2,
        annotation_text=f"  Limite: {threshold:.4f}",
        annotation_position="bottom right",
        annotation_font_color="#e74c3c", annotation_font_size=11,
    )

    # Trend line (past portion)
    if prediction is not None:
        trend_idx = prediction["trend_x_idx"]
        trend_y = prediction["trend_y_vals"]
        x_trend = [x_vals[i] for i in trend_idx if i < n]
        y_trend = trend_y[: len(x_trend)]
        if len(x_trend) > 1:
            fig.add_trace(go.Scatter(
                x=x_trend, y=y_trend,
                mode="lines",
                line=dict(color="#e67e22", width=2.5, dash="solid"),
                name="Tendencia (ultimos 25%)",
                hoverinfo="skip",
            ))

        # Projection (future portion)
        proj_idx = prediction["proj_x_idx"]
        proj_y = prediction["proj_y_vals"]
        if has_ts:
            dt = timestamps[-1] - timestamps[-2]
            x_proj = [timestamps[-1] + dt * (i - n + 1) for i in proj_idx]
        else:
            x_proj = list(proj_idx)
        x_proj_valid = x_proj[: len(proj_y)]
        y_proj_clip = np.clip(proj_y, y_min, y_max * 1.5)
        fig.add_trace(go.Scatter(
            x=x_proj_valid, y=y_proj_clip,
            mode="lines",
            line=dict(color="#e67e22", width=2, dash="dash"),
            name=f"Projecao → {prediction['predicted_ts'].strftime('%d/%m %H:%M') if has_ts else ''}",
            hoverinfo="skip",
        ))

        # Vertical line at predicted failure
        x_fail = prediction["predicted_ts"] if has_ts else prediction["abs_cross"]
        fig.add_vline(
            x=x_fail if not has_ts else x_fail.isoformat(),
            line_dash="dot", line_color="#e67e22", line_width=2,
            annotation_text="  Falha prevista",
            annotation_position="top right",
            annotation_font_color="#e67e22",
        )

    # Selected snapshot star
    fig.add_trace(go.Scatter(
        x=[x_vals[selected_idx]],
        y=[scores[selected_idx]],
        mode="markers",
        marker=dict(color="gold", size=18, symbol="star", line=dict(width=2, color="#111")),
        name="Snapshot inspecionado",
        hovertemplate=(
            f"<b>Snapshot #{selected_idx}</b><br>"
            f"Score: {scores[selected_idx]:.4f}<extra></extra>"
        ),
    ))

    xaxis_cfg: dict = {}
    if has_ts:
        xaxis_cfg = dict(type="date", tickformat="%d/%m\n%H:%M", tickangle=0,
                         tickfont=dict(size=9))

    fig.update_layout(
        title=dict(
            text="Score de anomalia ao longo do tempo — analisando passado e projetando o futuro",
            font=dict(size=13),
        ),
        xaxis_title="Data / hora do snapshot" if has_ts else "Snapshot (ordem cronologica)",
        yaxis_title="Score de anomalia  (acima do limite = suspeito)",
        xaxis=xaxis_cfg,
        yaxis=dict(range=[y_min, y_max]),
        height=490,
        margin=dict(l=50, r=20, t=80, b=50),
        hovermode="closest",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.06, xanchor="right", x=1,
            bgcolor="rgba(20,20,20,0.85)", bordercolor="#444", borderwidth=1,
        ),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        xaxis_gridcolor="#1a1a1a", yaxis_gridcolor="#222222",
    )
    return fig


def _fig_radar(
    X_test: np.ndarray,
    selected_idx: int,
    y_test: np.ndarray,
    feature_names: list[str],
) -> go.Figure:
    X_normal = X_test[y_test == 0]
    normal_mean = X_normal.mean(axis=0)
    normal_std = X_normal.std(axis=0) + 1e-9

    row_z = np.clip((X_test[selected_idx] - normal_mean) / normal_std, -4, 4)
    cats = [*feature_names, feature_names[0]]
    vals_win = [*row_z.tolist(), row_z[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[0.0] * len(cats), theta=cats,
        fill="toself", fillcolor="rgba(46,204,113,0.10)",
        line=dict(color="#2ecc71", width=2, dash="dot"),
        name="Normal (media)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=vals_win, theta=cats,
        fill="toself", fillcolor="rgba(230,126,34,0.15)",
        line=dict(color="#e67e22", width=2.5),
        name=f"Snapshot #{selected_idx}",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(
            visible=True, range=[-4, 4],
            tickvals=[-3, -1.5, 0, 1.5, 3], tickfont=dict(size=9),
        )),
        showlegend=True,
        title="Features vs. media das janelas saudaveis (desvios-padrao)",
        height=380,
        margin=dict(l=30, r=30, t=55, b=30),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
    )
    return fig


def _fig_score_hist(scores: np.ndarray, selected_idx: int, threshold: float) -> go.Figure:
    percentile = float((scores < scores[selected_idx]).mean() * 100)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=scores, nbinsx=40, marker_color="#3498db", opacity=0.75, name="Scores",
    ))
    fig.add_vline(x=threshold, line_dash="dash", line_color="#e74c3c", line_width=2,
                  annotation_text="Limite", annotation_position="top right",
                  annotation_font_color="#e74c3c")
    fig.add_vline(x=scores[selected_idx], line_color="gold", line_width=2.5,
                  annotation_text=f"Snapshot #{selected_idx} (p{percentile:.0f})",
                  annotation_position="top left", annotation_font_color="gold")
    fig.update_layout(
        title=f"Posicao no ranking — percentil {percentile:.0f}%",
        xaxis_title="Anomaly score",
        yaxis_title="Contagem",
        height=380,
        margin=dict(l=40, r=20, t=55, b=40),
        showlegend=False,
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        xaxis_gridcolor="#2a2a2a", yaxis_gridcolor="#2a2a2a",
    )
    return fig


def _fig_score_over_time_by_bearing(
    scores: np.ndarray,
    meta: pd.DataFrame,
    threshold: float,
) -> go.Figure | None:
    """Multi-bearing score chart (shown only when meta has bearing_id and timestamps)."""
    if "_meta_bearing_id" not in meta.columns or "_meta_timestamp" not in meta.columns:
        return None

    fig = go.Figure()
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]

    for i, bid in enumerate(sorted(meta["_meta_bearing_id"].unique())):
        mask = (meta["_meta_bearing_id"] == bid).values
        ts = pd.to_datetime(meta.loc[mask, "_meta_timestamp"])
        s = scores[mask]
        fig.add_trace(go.Scatter(
            x=ts, y=s, mode="lines+markers",
            marker=dict(size=3),
            line=dict(color=colors[i % len(colors)], width=1.5),
            name=f"Rolamento {bid}",
        ))

    fig.add_hline(y=threshold, line_dash="dash", line_color="#e74c3c", line_width=1.5,
                  annotation_text="Limite", annotation_position="top right",
                  annotation_font_color="#e74c3c")
    fig.update_layout(
        title="Score de anomalia por rolamento — comparativo",
        xaxis_title="Data",
        yaxis_title="Score",
        height=350,
        margin=dict(l=50, r=20, t=55, b=40),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        xaxis_gridcolor="#1a1a1a", yaxis_gridcolor="#222222",
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="right", x=1),
    )
    return fig


# ---------------------------------------------------------------------------
# UI sections
# ---------------------------------------------------------------------------
def _hero(scores: np.ndarray, y_test: np.ndarray, threshold: float) -> None:
    flagged = int((scores >= threshold).sum())
    tp = int(((scores >= threshold) & (y_test == 1)).sum())
    fp = int(((scores >= threshold) & (y_test == 0)).sum())
    if flagged == 0:
        st.markdown(
            '<div class="status-ok">✅ &nbsp; ROLAMENTO SAUDAVEL — nenhuma anomalia detectada</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="status-warn">⚠️ &nbsp; ATENCAO — {tp} snapshots anomalos detectados '
            f"· {fp} falso{'s alarmes' if fp != 1 else ' alarme'} "
            f"· limite: {threshold:.4f}</div>",
            unsafe_allow_html=True,
        )
    st.markdown("<br>", unsafe_allow_html=True)


def _kpi_row(
    scores: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    prediction: dict | None,
    selected_idx: int,
    timestamps: pd.DatetimeIndex,
) -> None:
    n_total = len(scores)
    n_pos = int(y_test.sum())
    n_neg = n_total - n_pos
    tp = int(((scores >= threshold) & (y_test == 1)).sum())
    fp = int(((scores >= threshold) & (y_test == 0)).sum())
    fn = n_pos - tp
    recall = tp / n_pos if n_pos > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    kpi1, kpi2, kpi3, kpi4, pred_col = st.columns([1, 1, 1, 1, 2])

    kpi1.metric(
        "Degradacoes detectadas",
        f"{tp} / {n_pos}",
        f"{recall:.1%} dos snapshots anomalos",
        help="Snapshots marcados como degradados no rotulo que o modelo alertou.",
    )
    fp_rate = fp / n_neg if n_neg > 0 else 0.0
    kpi2.metric(
        "Falsos alarmes",
        fp,
        f"{fp_rate:.1%} dos snapshots saudaveis",
        delta_color="inverse",
        help="Snapshots saudaveis que o modelo incorretamente alertou.",
    )
    kpi3.metric(
        "F1 Score",
        f"{f1:.1%}",
        help="Media harmonica entre deteccao e precisao. 100% = modelo perfeito.",
    )
    kpi4.metric(
        "Nao detectados",
        fn,
        f"{fn / n_pos:.1%} das degradacoes" if n_pos > 0 else "—",
        delta_color="inverse",
        help="Snapshots com rotulo degradado que passaram pelo modelo sem alerta.",
    )

    with pred_col:
        if prediction is None:
            # Show status relative to selected snapshot
            sel_ts = timestamps[selected_idx].strftime("%d/%m/%Y %H:%M") if len(timestamps) == len(scores) else f"#{selected_idx}"
            st.markdown(
                '<div class="ok-card">'
                "<h2>📈 Tendencia estavel</h2>"
                f"<p>Nenhuma projecao de falha no horizonte visivel</p>"
                f"<p>Snapshot atual: {sel_ts}</p>"
                f"<p>Score: {scores[selected_idx]:.4f}</p>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            pred_ts_str = prediction["predicted_ts"].strftime("%d/%m/%Y as %H:%M")
            h = prediction["hours_away"]
            st.markdown(
                '<div class="pred-card">'
                f"<h2>🔮 Falha prevista em {h:.0f}h</h2>"
                f"<p>Projecao: <b>{pred_ts_str}</b></p>"
                f"<p>Baseado na tendencia dos ultimos {prediction['n_trend']} snapshots</p>"
                f"<p style='font-size:0.8rem;color:#aaa;'>Coeficiente angular: {prediction['slope']:.6f}/snapshot</p>"
                "</div>",
                unsafe_allow_html=True,
            )


def _laudo(
    scores: np.ndarray,
    y_test: np.ndarray,
    meta: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    threshold: float,
    prediction: dict | None,
) -> None:
    above = scores >= threshold
    flagged = int(above.sum())

    tp = int(((above) & (y_test == 1)).sum())
    fp = int(((above) & (y_test == 0)).sum())
    fn = int(y_test.sum()) - tp
    recall = tp / max(int(y_test.sum()), 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    has_ts = len(timestamps) == len(scores)

    with st.expander("📋 Laudo automatico — resumo em linguagem simples", expanded=True):
        if flagged == 0:
            st.markdown("**Resultado:** nenhuma anomalia detectada no periodo de monitoramento.")
        else:
            first_anom_idx = int(np.argmax(above))
            first_ts = timestamps[first_anom_idx].strftime("%d/%m/%Y as %H:%M") if has_ts else f"snapshot #{first_anom_idx}"
            max_idx = int(np.argmax(scores))
            max_ts = timestamps[max_idx].strftime("%d/%m/%Y as %H:%M") if has_ts else f"snapshot #{max_idx}"

            st.markdown(
                f"**Resultado:** {flagged} snapshots acima do limite de anomalia detectados."
            )
            st.markdown(
                f"**Primeira anomalia:** {first_ts} "
                f"(score: {scores[first_anom_idx]:.4f})"
            )
            st.markdown(
                f"**Pico maximo:** {max_ts} "
                f"(score: {scores[max_idx]:.4f})"
            )

        st.markdown(
            f"**Degradacoes capturadas:** {tp} de {int(y_test.sum())} (**{recall:.1%}**) · "
            f"**Nao detectadas:** {fn} · "
            f"**Falsos alarmes:** {fp} · "
            f"**F1:** {f1:.1%}"
        )

        if prediction is not None:
            pred_ts_str = prediction["predicted_ts"].strftime("%d/%m/%Y as %H:%M")
            st.markdown(
                f"**Projecao de falha:** {pred_ts_str} "
                f"(em ~{prediction['hours_away']:.0f} horas a partir do ultimo snapshot). "
                "Baseado na regressao linear dos ultimos 25% dos snapshots."
            )

        st.caption(
            "Dataset IMS/NASA Run 2 (2004-02-12 a 2004-02-19) · "
            "Rolamento 1: falha documentada na pista externa (outer race) ao fim do periodo. "
            "Rotulos: primeiros 40% dos snapshots = saudavel (y=0); restante = potencialmente degradado (y=1)."
        )


def _detail_panel(
    X_test: np.ndarray,
    scores: np.ndarray,
    y_test: np.ndarray,
    meta: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    feature_names: list[str],
    sel: int,
    threshold: float,
) -> None:
    has_ts = len(timestamps) == len(scores)
    sel_score = float(scores[sel])
    is_anom = sel_score >= threshold
    badge_color = "red" if is_anom else "green"
    badge_text = "ANOMALIA DETECTADA" if is_anom else "Normal"
    label_text = "degradado (rotulo)" if y_test[sel] else "saudavel (rotulo)"

    ts_str = timestamps[sel].strftime("%d/%m/%Y %H:%M") if has_ts else f"#{sel}"
    bearing_id = int(meta["_meta_bearing_id"].iloc[sel]) if "_meta_bearing_id" in meta.columns else "—"
    snap_idx = int(meta["_meta_snapshot_idx"].iloc[sel]) if "_meta_snapshot_idx" in meta.columns else sel

    st.subheader(f"Snapshot #{sel} de {len(X_test)} — :{badge_color}[{badge_text}]")

    info_cols = st.columns(4)
    info_cols[0].metric("Data / hora", ts_str)
    info_cols[1].metric("Anomaly score", f"{sel_score:.4f}")
    info_cols[2].metric("Rolamento", f"Bearing {bearing_id}")
    info_cols[3].metric("Indice temporal", f"#{snap_idx}")

    st.caption(
        f"**Rotulo real:** {label_text} · "
        f"**Diagnostico do modelo:** {'FALHA' if is_anom else 'normal'} · "
        f"**Limiar:** {threshold:.4f}"
    )

    col_r, col_h = st.columns(2)
    with col_r:
        st.plotly_chart(_fig_radar(X_test, sel, y_test, feature_names), use_container_width=True)
        st.caption(
            "Spikes afastados do centro indicam features anômalas. "
            "O contorno verde e a media dos snapshots saudaveis (z-score 0)."
        )
    with col_h:
        st.plotly_chart(_fig_score_hist(scores, sel, threshold), use_container_width=True)
        st.caption(
            "A linha dourada mostra onde este snapshot esta na distribuicao completa. "
            "Snapshots a direita da linha vermelha sao flagged como anomalos."
        )


def _shap_expander(
    model_name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    sel: int,
) -> None:
    with st.expander("🔍 Por que este snapshot foi flagged? — Explicacao SHAP (on-demand)"):
        if model_name in _SLOW_MODELS:
            st.info(
                f"**{model_name}** usa KernelExplainer (model-agnostic). "
                "Pode levar 15-30 s. IsolationForest usa TreeExplainer (instantaneo)."
            )
        if st.button("Calcular SHAP para este snapshot", key="shap_btn"):
            with st.spinner("Calculando SHAP values…"):
                model = load_model(model_name)
                X_bg = X_test[y_test == 0][:50]
                try:
                    import shap as _shap

                    exp = explain(
                        model,
                        X_test[[sel]],
                        feature_names,
                        X_background=X_bg,
                        bg_size=50,
                        eval_size=None,
                    )
                    _shap.plots.waterfall(exp[0], show=False)
                    st.pyplot(plt.gcf())
                    plt.close("all")
                    st.caption(
                        "Barras vermelhas empurram o score para cima (mais anomalo). "
                        "Barras azuis puxam para baixo (mais normal). "
                        "O ponto de partida E[f(x)] e a media do modelo no conjunto de referencia."
                    )
                except Exception as exc:
                    st.error(f"SHAP falhou: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("🔧 IMS/NASA Bearing Prognostics")
    st.caption(
        "Deteccao de anomalias e projecao de falhas em rolamentos industriais · "
        "Dataset IMS/NASA (University of Cincinnati) · "
        "[github.com/RenanMiqueloti/industrial-anomaly-detection]"
        "(https://github.com/RenanMiqueloti/industrial-anomaly-detection)"
    )

    with st.expander("📖 O que e isso? — Contexto e glossario"):
        st.markdown(
            """
**O problema:** rolamentos industriais degradam lentamente ao longo de semanas.
Este sistema detecta o inicio da degradacao e projeta quando a falha ocorrera,
permitindo manutencao planejada antes da parada nao programada.

**Dataset IMS/NASA Run 2:**
- Periodo: 12 a 19 de fevereiro de 2004 (~7 dias de monitoramento continuo)
- 4 rolamentos monitorados simultaneamente, 20 000 Hz de amostragem
- 1 snapshot de 1 segundo a cada ~10 minutos (984 snapshots totais)
- Rolamento 1: falha documentada na **pista externa (outer race)** ao final do periodo
- Dados reais com timestamps — nao e simulacao laboratorial

**Como funciona:**
1. Cada snapshot (1 s / 20 480 amostras) gera 11 features: RMS, pico, curtose, energia por banda
2. O modelo treina **exclusivamente nos primeiros 40% dos snapshots** (periodo saudavel)
3. Cada snapshot recebe um score de anomalia (maior = mais suspeito)
4. Uma regressao linear nos ultimos 25% projeta quando o score cruzara o limite

**Bandas de frequencia (20 kHz Nyquist):**

| Banda | Frequencia | O que captura |
|-------|-----------|---------------|
| Band 0-500 Hz | Baixa | Desbalanceamento, ressonancias de estrutura |
| Band 500-2000 Hz | Media-baixa | BPFO/BPFI harmonicos fundamentais |
| Band 2-5 kHz | Media-alta | Defeitos incipientes de rolamento |
| Band 5-10 kHz | Alta | Dano avancado, impactos de esfera |

**Modelos disponiveis:** IsolationForest · OC-SVM · LOF · AutoEncoder
(todos treinados **sem rotulos de falha** — aprendizado nao supervisionado)
            """
        )

    # --- Load artifacts ---
    data = load_test_data()
    if data is None:
        st.info(
            "**Artefatos nao encontrados.** Execute o pipeline primeiro:\n\n"
            "```bash\n"
            "make download\nmake features\nmake train\nmake compare\n"
            "```\n\nDepois reinicie o dashboard."
        )
        st.stop()

    X_test, y_test, meta_test, feature_names = data
    n_test = len(X_test)

    # --- Sidebar ---
    with st.sidebar:
        st.header("Configuracoes")
        model_name = st.selectbox("Modelo", list(_MODEL_CLASSES.keys()), index=0)

        # Bearing filter (if multi-bearing data is loaded)
        if "_meta_bearing_id" in meta_test.columns:
            available_bearings = sorted(meta_test["_meta_bearing_id"].unique().tolist())
            if len(available_bearings) > 1:
                selected_bearing = st.selectbox(
                    "Rolamento analisado",
                    options=available_bearings,
                    format_func=lambda b: f"Bearing {b}",
                )
                bear_mask = (meta_test["_meta_bearing_id"] == selected_bearing).values
            else:
                selected_bearing = available_bearings[0]
                bear_mask = np.ones(n_test, dtype=bool)
        else:
            bear_mask = np.ones(n_test, dtype=bool)

        scores_all = compute_scores(model_name, X_test.tobytes(), n_test)
        if scores_all is None:
            st.warning(f"Modelo **{model_name}** nao encontrado. Execute `make compare`.")
            st.stop()

        # Filter to selected bearing
        scores = scores_all[bear_mask]
        y_bear = y_test[bear_mask]
        meta_bear = meta_test.iloc[bear_mask].reset_index(drop=True)
        timestamps = _get_timestamps(meta_bear, len(scores))

        _thr_default = float(np.clip(
            _default_threshold(scores, y_bear, model_name),
            scores.min(), scores.max(),
        ))
        threshold = st.slider(
            "Limite de anomalia",
            min_value=float(scores.min()),
            max_value=float(scores.max()),
            value=_thr_default,
            format="%.4f",
            help="Snapshots com score acima deste valor sao flagged. "
                 "O padrao e o 99o percentil dos snapshots saudaveis (<=1% de falsos alarmes).",
        )
        st.caption(f"Padrao calibrado: **{_thr_default:.4f}** (p99 dos saudaveis)")

        st.divider()
        st.markdown("**Sobre o IMS Run 2:**")
        st.markdown(
            "- 7 dias de monitoramento continuo (fev/2004)\n"
            "- 4 rolamentos simultaneos, 20 kHz\n"
            "- **Bearing 1**: falha na pista externa documentada\n"
            "- Snapshots a cada ~10 min → linha do tempo real\n"
        )
        st.divider()
        st.caption(
            "**Como usar:** ajuste o limite → observe onde as anomalias aparecem → "
            "clique em um ponto para inspecionar o snapshot."
        )

    # --- Dataset banner ---
    n_days = 0.0
    if len(timestamps) >= 2:
        n_days = (timestamps[-1] - timestamps[0]).total_seconds() / 86400
    st.info(
        f"**Dataset IMS/NASA Run 2** · {len(scores)} snapshots (1 s a 20 kHz cada) · "
        f"{n_days:.1f} dias de monitoramento continuo · "
        "4 rolamentos · Bearing 1: falha na pista externa ao fim do periodo",
    )

    # --- Prediction ---
    prediction = _predict_failure(scores, timestamps, threshold)

    # --- Hero ---
    _hero(scores, y_bear, threshold)

    # --- Init selected snapshot ---
    if "selected_idx" not in st.session_state:
        first_anom = np.where((scores >= threshold) & (y_bear == 1))[0]
        st.session_state.selected_idx = int(first_anom[0]) if len(first_anom) else 0
    st.session_state.selected_idx = int(np.clip(st.session_state.selected_idx, 0, len(scores) - 1))

    # --- KPIs + prediction card ---
    _kpi_row(scores, y_bear, threshold, prediction, st.session_state.selected_idx, timestamps)

    # --- Laudo ---
    _laudo(scores, y_bear, meta_bear, timestamps, threshold, prediction)

    st.divider()

    # --- Timeline ---
    st.subheader("📈 Timeline de Anomalia — Passado + Projecao Futura")
    st.caption(
        "Cada ponto = um snapshot (~1 s de vibracoes). "
        "**Verde** = score abaixo do limite. **Vermelho** (diamante) = anomalia detectada. "
        "**Linha laranja solida** = tendencia dos ultimos 25%. "
        "**Linha laranja tracejada** = projecao futura. "
        "**Clique em qualquer ponto** para ver os detalhes do snapshot."
    )

    timeline_fig = _fig_timeline(
        scores, meta_bear, timestamps, y_bear, threshold,
        st.session_state.selected_idx, prediction,
    )
    event = st.plotly_chart(timeline_fig, use_container_width=True, on_select="rerun", key="timeline")

    if event and event.selection and event.selection.points:
        pt = event.selection.points[0]
        raw_cd = pt.get("customdata")
        if raw_cd is not None:
            new_idx = int(raw_cd[0] if isinstance(raw_cd, (list, tuple)) else raw_cd)
            st.session_state.selected_idx = int(np.clip(new_idx, 0, len(scores) - 1))

    st.divider()

    # --- Detail panel ---
    _detail_panel(
        X_test[bear_mask],
        scores,
        y_bear,
        meta_bear,
        timestamps,
        feature_names,
        st.session_state.selected_idx,
        threshold,
    )

    st.divider()

    # --- SHAP ---
    _shap_expander(
        model_name,
        X_test[bear_mask],
        y_bear,
        feature_names,
        st.session_state.selected_idx,
    )

    # --- Multi-bearing comparison (if applicable) ---
    multi_fig = _fig_score_over_time_by_bearing(scores_all, meta_test, threshold)
    if multi_fig is not None:
        st.divider()
        st.subheader("📊 Comparativo entre Rolamentos")
        st.caption(
            "Todos os 4 rolamentos no mesmo grafico. "
            "O Bearing 1 deve mostrar crescimento de score no final do periodo."
        )
        st.plotly_chart(multi_fig, use_container_width=True)


if __name__ == "__main__":
    main()
