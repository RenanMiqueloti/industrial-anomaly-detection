"""Streamlit dashboard — IMS/NASA Bearing Prognostics.

UX priority
-----------
1. Entender em 30 s: status atual + KPIs + card de previsão de falha
2. Auto-diagnóstico em linguagem natural + separabilidade de scores
3. Timeline com timestamps reais + projeção futura
4. Detalhe do snapshot: barra de desvio por feature + histograma de score
5. Explicação SHAP sob demanda

Dataset
-------
IMS/NASA (University of Cincinnati) — Run 2
2004-02-12 a 2004-02-19 | 984 snapshots | 4 rolamentos | 20 kHz
Rolamento 1: falha na pista externa (outer race) ao final do período.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Permite rodar via `streamlit run src/dashboard.py` sem o pacote estar
# instalado — Streamlit Cloud não roda `pip install -e .` por padrão.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # must come before any other matplotlib import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

# torch (via AutoEncoderDetector) and shap (via src.explain) are imported
# lazily inside _get_model_class() and the SHAP button callback — the default
# IsolationForest flow never needs either, and skipping those imports cuts
# several seconds off the cold start on Streamlit Cloud's shared CPU.

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_RESULTS = Path("results")
_DATA_FEATURES = Path("data/features/features.parquet")
_THRESHOLD_JSON = _RESULTS / "threshold.json"

_MODEL_THRESHOLD_KEY = {
    "IsolationForest": "iforest",
    "OC-SVM": "ocsvm",
    "AutoEncoder": "ae",
}
_MODEL_FILES = {
    "IsolationForest": _RESULTS / "iforest_model.joblib",
    "OC-SVM": _RESULTS / "ocsvm_model.joblib",
    "AutoEncoder": _RESULTS / "ae_model.joblib",
}
_SLOW_MODELS = {"OC-SVM", "AutoEncoder"}


def _get_model_class(model_name: str):
    if model_name == "IsolationForest":
        from src.models.iforest import IForestDetector

        return IForestDetector
    if model_name == "OC-SVM":
        from src.models.ocsvm import OCSVMDetector

        return OCSVMDetector
    if model_name == "AutoEncoder":
        from src.models.autoencoder import AutoEncoderDetector

        return AutoEncoderDetector
    raise KeyError(model_name)


_FEATURE_LABELS: dict[str, str] = {
    "rms": "RMS",
    "peak": "Pico",
    "crest_factor": "F. Crista",
    "kurtosis": "Curtose",
    "skewness": "Assimetria",
    "std": "Desv. Padrão",
    "p2p": "Pico-a-Pico",
    "band_0_500": "0–500 Hz",
    "band_500_2000": "500–2k Hz",
    "band_2000_5000": "2–5 kHz",
    "band_5000_10000": "5–10 kHz",
}

_BEARING_COLORS: dict[int, str] = {
    1: "#ef4444",
    2: "#3b82f6",
    3: "#14b8a6",
    4: "#a855f7",
}

# Maps dominant feature → physical failure mode hint.
# Frequências de defeito para Rexnord ZA-2115 a 2000 RPM (literatura, Qiu et al. 2006,
# Bechhoefer et al., Kalikatzarakis 2018): BPFO ≈ 236 Hz, BPFI ≈ 297 Hz, BSF ≈ 140 Hz
# (2×BSF ≈ 280 Hz), FTF ≈ 15 Hz — todas as fundamentais ficam na banda 0–500 Hz.
_FAILURE_HINTS: dict[str, str] = {
    "band_5000_10000": "alta frequência — impactos severos, ressonâncias excitadas por dano avançado",
    "band_2000_5000": "modos estruturais (housing/eixo) excitados por impactos",
    "band_500_2000": "harmônicos das frequências de defeito (2× a 5× BPFO/BPFI)",
    "band_0_500": "frequências fundamentais de defeito — BPFO (~236 Hz), BPFI (~297 Hz), BSF (~140 Hz)",
    "kurtosis": "impactos periódicos impulsivos — fadiga de superfície localizada",
    "crest_factor": "picos de vibração extremos — dano concentrado",
    "rms": "vibração global elevada — degradação disseminada",
    "peak": "valores de pico extremos — impactos severos",
    "p2p": "amplitude pico-a-pico elevada — impactos ou folgas mecânicas",
    "std": "variabilidade do sinal elevada — modulação ou regime variável",
    "skewness": "assimetria na distribuição — dano preferencial em uma direção",
}

st.set_page_config(
    page_title="IMS Bearing Prognostics",
    page_icon="🔧",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }
    h1 { font-size: 1.7rem !important; font-weight: 600 !important; letter-spacing: -0.01em; }
    h2 { font-size: 1.15rem !important; font-weight: 600 !important; margin-top: 0.4rem !important; }
    h3 { font-size: 1.0rem !important; font-weight: 600 !important; }
    [data-testid="stMetricValue"] { font-size: 1.45rem; font-weight: 600; }
    [data-testid="stMetricLabel"] { color:#9ba3af; font-size:0.78rem; text-transform:uppercase;
                                     letter-spacing:0.04em; }
    [data-testid="stMetricDelta"] { font-size:0.78rem; }
    [data-testid="stCaptionContainer"] { color:#9ba3af; }
    hr { margin: 0.8rem 0 !important; border-color: #1f2430 !important; }

    .status-card { background:#13161d; border-left:4px solid #6b7280; border-radius:4px;
                    padding:12px 18px; font-size:0.95rem; color:#e5e7eb; }
    .status-card .label { color:#9ba3af; font-size:0.78rem; text-transform:uppercase;
                          letter-spacing:0.05em; margin-right:8px; }
    .status-card .title { font-weight:600; font-size:1.05rem; }
    .status-card .meta  { color:#9ba3af; font-size:0.85rem; margin-top:4px; }
    .status-ok   { border-left-color:#10b981; }
    .status-warn { border-left-color:#ef4444; }
    .status-recurrent { border-left-color:#f59e0b; }

    .info-card    { background:#13161d; border-left:4px solid #6b7280; border-radius:4px;
                    padding:14px 18px; }
    .info-card .heading { color:#e5e7eb; font-weight:600; font-size:0.95rem; margin:0 0 4px 0; }
    .info-card .value   { color:#e5e7eb; font-size:1.25rem; font-weight:600; margin:2px 0; }
    .info-card .sub     { color:#9ba3af; font-size:0.82rem; margin:2px 0; }
    .pred-card { border-left-color:#f59e0b; }
    .pred-card .heading { color:#f59e0b; }
    .fail-card { border-left-color:#ef4444; }
    .fail-card .heading { color:#ef4444; }
    .recur-card { border-left-color:#f59e0b; }
    .recur-card .heading { color:#f59e0b; }
    .ok-card { border-left-color:#10b981; }
    .ok-card .heading { color:#10b981; }

    .diag-box { background:#13161d; border-left:4px solid #3b82f6; border-radius:4px;
                padding:14px 18px; font-size:0.93rem; line-height:1.6; color:#d1d5db; }
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
    return _get_model_class(model_name).load(path)


@st.cache_data
def compute_scores(model_name: str, X_bytes: bytes, n_rows: int) -> np.ndarray | None:
    model = load_model(model_name)
    if model is None:
        return None
    X = np.frombuffer(X_bytes, dtype=np.float64).reshape(n_rows, -1)
    return model.score(X)


@st.cache_data
def load_healthy_baseline() -> np.ndarray | None:
    """Return feature matrix for all y=0 rows in the full dataset."""
    if not _DATA_FEATURES.exists():
        return None
    df = pd.read_parquet(_DATA_FEATURES)
    feat_cols = [c for c in df.columns if not c.startswith("_meta_")]
    return df.loc[df["_meta_y"] == 0, feat_cols].values


@st.cache_data
def load_full_metadata() -> pd.DataFrame | None:
    """Return per-row metadata (timestamp, bearing_id, y) for the full feature parquet.

    Aligned 1:1 with :func:`compute_full_dataset_scores` so callers can join
    scores with bearing/time without re-reading the heavy feature matrix.
    """
    if not _DATA_FEATURES.exists():
        return None
    df = pd.read_parquet(_DATA_FEATURES)
    cols = [c for c in ("_meta_timestamp", "_meta_bearing_id", "_meta_y") if c in df.columns]
    return df[cols].copy()


_PRECOMPUTED_SCORES = _RESULTS / "full_dataset_scores.parquet"


@st.cache_data
def compute_full_dataset_scores(
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (scores, y, bearing_ids) for the entire feature parquet.

    Reads ``results/full_dataset_scores.parquet`` when it exists (produced by
    ``make precompute-scores``). Falls back to scoring the parquet on the fly
    so deployments that haven't re-run the pipeline keep working.
    """
    score_col = _MODEL_THRESHOLD_KEY.get(model_name)
    if _PRECOMPUTED_SCORES.exists() and score_col is not None:
        df_scores = pd.read_parquet(_PRECOMPUTED_SCORES)
        if score_col in df_scores.columns:
            scores = df_scores[score_col].values.astype(np.float64)
            y = (
                df_scores["_meta_y"].values
                if "_meta_y" in df_scores.columns
                else np.zeros(len(scores), dtype=int)
            )
            bearing_ids = (
                df_scores["_meta_bearing_id"].astype(int).values
                if "_meta_bearing_id" in df_scores.columns
                else np.zeros(len(scores), dtype=int)
            )
            return scores, y, bearing_ids

    if not _DATA_FEATURES.exists():
        return None
    df = pd.read_parquet(_DATA_FEATURES)
    feat_cols = [c for c in df.columns if not c.startswith("_meta_")]
    X = df[feat_cols].values
    y = df["_meta_y"].values
    bearing_ids = (
        df["_meta_bearing_id"].values
        if "_meta_bearing_id" in df.columns
        else np.zeros(len(y), dtype=int)
    )
    model = load_model(model_name)
    if model is None:
        return None
    return model.score(X), y, bearing_ids.astype(int)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _default_threshold(
    scores: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    bearing_id: int | None = None,
) -> float:
    if _THRESHOLD_JSON.exists():
        try:
            data = json.loads(_THRESHOLD_JSON.read_text())
            base_key = _MODEL_THRESHOLD_KEY.get(model_name, model_name.lower())
            for key in [f"{base_key}_b{bearing_id}" if bearing_id else None, base_key]:
                if key and key in data:
                    return float(data[key])
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            logger.warning(
                "Could not read %s for model=%s bearing=%s: %s — falling back to data-derived p99.",
                _THRESHOLD_JSON,
                model_name,
                bearing_id,
                exc,
            )
    normal_mask = y_test == 0
    if normal_mask.any():
        return float(np.percentile(scores[normal_mask], 99))
    return float(np.percentile(scores, 50))


def _get_timestamps(meta: pd.DataFrame, n: int) -> pd.DatetimeIndex:
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
    n = len(scores)
    if n < 20 or len(timestamps) != n:
        return None

    n_trend = max(int(n * trend_frac), 5)
    x = np.arange(n_trend, dtype=float)
    y = scores[n - n_trend :]

    slope, intercept = np.polyfit(x, y, 1)
    if slope <= 1e-10:
        return None

    # Suppress projections on bearings that are not actually degrading:
    #   1. recent capture must be substantive (≥20% of the trend window above
    #      threshold), otherwise the slope is fitting noise drift
    #   2. the regression must explain real variance (R² ≥ 0.3)
    # Without these, any tiny positive slope on a clearly healthy bearing
    # produces a misleading "failure predicted in Xh" card.
    if float(np.mean(y >= threshold)) < 0.20:
        return None

    ss_res = float(np.sum((y - (slope * x + intercept)) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    if r2 < 0.3:
        return None

    t_cross_rel = (threshold - intercept) / slope
    if t_cross_rel <= n_trend:
        return None

    abs_cross = (n - n_trend) + t_cross_rel
    extra = abs_cross - n

    if len(timestamps) >= 2:
        dt = timestamps[-1] - timestamps[-2]
    else:
        return None

    predicted_ts = timestamps[-1] + dt * extra
    hours_away = extra * dt.total_seconds() / 3600

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


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError as exc:
        # roc_auc_score raises ValueError when y_true is degenerate (only one
        # class) or scores contain NaN/inf. Logged so it's traceable instead
        # of disappearing as None on the dashboard.
        logger.debug("roc_auc_score returned ValueError: %s", exc)
        return None


_STATE_FAILURE = "falha"
_STATE_RECURRENT = "recorrente"
_STATE_STABLE = "estavel"

# State-classification thresholds. Tuned against IMS Run 2 ground truth: only
# Bearing 1 has a documented failure — B2/B3/B4 are healthy per the paper. Any
# rule that pushes B2/B3/B4 into "falha" contradicts the dataset.
_RECENT_FRAC = 0.25
_FAIL_RECENT_RATE = 0.60
_FAIL_EXCESS_PCT = 20.0
_RECURRENT_RECENT_RATE = 0.10


def _first_sustained_detection(
    scores: np.ndarray,
    threshold: float,
    window: int = 10,
    frac: float = 0.8,
) -> int | None:
    """Index of the first ``window`` of consecutive snapshots where ``frac`` of them
    are above ``threshold``. Returns the start index of that window, or None.

    Looking for a sustained pattern instead of a single crossing rejects isolated
    spikes — the p99 threshold leaves ~1% of healthy snapshots above by design,
    so a one-off crossing isn't meaningful.
    """
    n = len(scores)
    if n < window:
        return None
    above = (scores >= threshold).astype(np.int64)
    csum = np.cumsum(np.concatenate([[0], above]))
    rates = (csum[window:] - csum[:-window]) / window
    hits = np.where(rates >= frac)[0]
    return int(hits[0]) if len(hits) else None


def _bearing_state(scores: np.ndarray, threshold: float) -> tuple[str, float, float]:
    """Classify a bearing's current state from its score history.

    Uses the trailing ``_RECENT_FRAC`` of the score series — sustained recent
    behaviour is a stronger signal than peak score alone (a single noisy
    snapshot can drive max well above threshold without a real failure).

    Returns
    -------
    state:        one of ``_STATE_FAILURE``, ``_STATE_RECURRENT``, ``_STATE_STABLE``.
    recent_rate:  fraction of recent snapshots above threshold (0..1).
    excess_pct:   max score's percentage excess over threshold.
    """
    if len(scores) == 0:
        return _STATE_STABLE, 0.0, 0.0
    n_recent = max(int(len(scores) * _RECENT_FRAC), 1)
    recent_above = scores[-n_recent:] >= threshold
    recent_rate = float(recent_above.mean())
    excess_pct = (float(scores.max()) - threshold) / max(threshold, 1e-9) * 100.0

    if recent_rate >= _FAIL_RECENT_RATE and excess_pct >= _FAIL_EXCESS_PCT:
        return _STATE_FAILURE, recent_rate, excess_pct
    if recent_rate >= _RECURRENT_RECENT_RATE:
        return _STATE_RECURRENT, recent_rate, excess_pct
    return _STATE_STABLE, recent_rate, excess_pct


def _load_all_thresholds(model_name: str) -> dict[int, float]:
    """Load per-bearing thresholds for the given model. Returns {bearing_id: threshold}.

    Returns an empty dict when the threshold file is missing or unreadable —
    the caller is expected to treat empty as "no per-bearing calibration".
    """
    if not _THRESHOLD_JSON.exists():
        return {}
    try:
        data = json.loads(_THRESHOLD_JSON.read_text())
        base_key = _MODEL_THRESHOLD_KEY.get(model_name, model_name.lower())
        return {
            b: float(data[f"{base_key}_b{b}"]) for b in [1, 2, 3, 4] if f"{base_key}_b{b}" in data
        }
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        logger.warning("Could not load per-bearing thresholds for %s: %s", model_name, exc)
        return {}


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

    # Light tint above threshold — softer than full zones, no big annotations.
    fig.add_hrect(
        y0=threshold,
        y1=y_max,
        fillcolor="rgba(239,68,68,0.06)",
        layer="below",
        line_width=0,
    )

    x_vals = list(timestamps) if has_ts else list(range(n))

    mask_ok = ~above
    if mask_ok.any():
        x_ok = [x_vals[i] for i in range(n) if mask_ok[i]]
        hover_ok = [
            f"{timestamps[i].strftime('%d/%m %H:%M') if has_ts else f'#{i}'} · "
            f"score {scores[i]:.4f}<extra></extra>"
            for i in range(n)
            if mask_ok[i]
        ]
        fig.add_trace(
            go.Scatter(
                x=x_ok,
                y=scores[mask_ok],
                mode="markers",
                marker=dict(color="#6b7280", size=4, opacity=0.7),
                name="Normal",
                customdata=np.where(mask_ok)[0].tolist(),
                hovertemplate=hover_ok,
            )
        )

    if above.any():
        x_ab = [x_vals[i] for i in range(n) if above[i]]
        hover_ab = [
            f"{timestamps[i].strftime('%d/%m %H:%M') if has_ts else f'#{i}'} · "
            f"score {scores[i]:.4f} · anomalia<extra></extra>"
            for i in range(n)
            if above[i]
        ]
        fig.add_trace(
            go.Scatter(
                x=x_ab,
                y=scores[above],
                mode="markers",
                marker=dict(
                    color="#ef4444",
                    size=7,
                    symbol="diamond",
                    opacity=0.95,
                    line=dict(width=0),
                ),
                name="Anomalia",
                customdata=np.where(above)[0].tolist(),
                hovertemplate=hover_ab,
            )
        )

    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#ef4444",
        line_width=1.5,
        annotation_text=f"Limite {threshold:.4f}",
        annotation_position="bottom right",
        annotation_font_color="#9ba3af",
        annotation_font_size=10,
    )

    if above.any():
        first_anom = int(np.argmax(above))
        x_first = x_vals[first_anom]
        fig.add_trace(
            go.Scatter(
                x=[x_first, x_first],
                y=[y_min, y_max],
                mode="lines",
                line=dict(color="#3b82f6", width=1.2, dash="dot"),
                hoverinfo="skip",
                showlegend=False,
                name="",
            )
        )
        fig.add_annotation(
            x=x_first,
            y=y_max * 0.97,
            text=" 1ª detecção",
            showarrow=False,
            font=dict(color="#3b82f6", size=10),
            xanchor="left",
            xref="x",
            yref="y",
        )

    if prediction is not None:
        trend_idx = prediction["trend_x_idx"]
        trend_y = prediction["trend_y_vals"]
        x_trend = [x_vals[i] for i in trend_idx if i < n]
        y_trend = trend_y[: len(x_trend)]
        if len(x_trend) > 1:
            fig.add_trace(
                go.Scatter(
                    x=x_trend,
                    y=y_trend,
                    mode="lines",
                    line=dict(color="#f59e0b", width=2),
                    name="Tendência",
                    hoverinfo="skip",
                )
            )

        proj_idx = prediction["proj_x_idx"]
        proj_y = prediction["proj_y_vals"]
        if has_ts:
            dt = timestamps[-1] - timestamps[-2]
            x_proj = [timestamps[-1] + dt * (i - n + 1) for i in proj_idx]
        else:
            x_proj = list(proj_idx)
        x_proj_valid = x_proj[: len(proj_y)]
        y_proj_clip = np.clip(proj_y, y_min, y_max * 1.5)
        fig.add_trace(
            go.Scatter(
                x=x_proj_valid,
                y=y_proj_clip,
                mode="lines",
                line=dict(color="#f59e0b", width=1.5, dash="dash"),
                name="Projeção",
                hoverinfo="skip",
            )
        )

        x_fail = prediction["predicted_ts"] if has_ts else prediction["abs_cross"]
        fig.add_trace(
            go.Scatter(
                x=[x_fail, x_fail],
                y=[y_min, y_max],
                mode="lines",
                line=dict(color="#f59e0b", width=1, dash="dot"),
                hoverinfo="skip",
                showlegend=False,
                name="",
            )
        )
        fig.add_annotation(
            x=x_fail,
            y=y_max * 0.85,
            text=" Falha prevista",
            showarrow=False,
            font=dict(color="#f59e0b", size=10),
            xanchor="left",
            xref="x",
            yref="y",
        )

    fig.add_trace(
        go.Scatter(
            x=[x_vals[selected_idx]],
            y=[scores[selected_idx]],
            mode="markers",
            marker=dict(
                color="rgba(0,0,0,0)",
                size=14,
                symbol="circle",
                line=dict(width=2, color="#fbbf24"),
            ),
            name="Selecionado",
            hovertemplate=(f"#{selected_idx} · score {scores[selected_idx]:.4f}<extra></extra>"),
        )
    )

    xaxis_cfg: dict = {}
    if has_ts:
        xaxis_cfg = dict(type="date", tickformat="%d/%m\n%H:%M", tickangle=0, tickfont=dict(size=9))

    fig.update_layout(
        xaxis_title="Data / hora" if has_ts else "Snapshot",
        yaxis_title="Score de anomalia",
        xaxis=xaxis_cfg,
        yaxis=dict(range=[y_min, y_max]),
        height=440,
        margin=dict(l=50, r=20, t=40, b=50),
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#e5e7eb", size=11),
        xaxis_gridcolor="#1f2430",
        yaxis_gridcolor="#1f2430",
    )
    return fig


def _fig_feature_bar(
    snapshot: np.ndarray,
    X_healthy: np.ndarray,
    feature_names: list[str],
    selected_idx: int,
) -> go.Figure:
    """Horizontal bar chart of z-scores per feature vs. healthy baseline."""
    mean_h = X_healthy.mean(axis=0)
    std_h = X_healthy.std(axis=0) + 1e-9
    z = (snapshot - mean_h) / std_h

    labels = [_FEATURE_LABELS.get(f, f) for f in feature_names]

    # Sort by absolute z-score descending
    order = np.argsort(np.abs(z))[::-1]
    z_sorted = z[order]
    labels_sorted = [labels[i] for i in order]
    feat_sorted = [feature_names[i] for i in order]

    # Cap display at ±15σ; show actual value in hover/text
    Z_CAP = 15.0
    z_display = np.clip(z_sorted, -Z_CAP, Z_CAP)

    colors = []
    for zi in z_sorted:
        if abs(zi) > 3:
            colors.append("#ef4444")
        elif abs(zi) > 1.5:
            colors.append("#f59e0b")
        else:
            colors.append("#6b7280")

    hover_text = [
        f"<b>{lbl}</b> ({feat}) · z = {zi:+.2f}σ<extra></extra>"
        for lbl, feat, zi in zip(labels_sorted, feat_sorted, z_sorted, strict=True)
    ]
    text_vals = [f"{zi:+.1f}σ" for zi in z_sorted]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=z_display,
            y=labels_sorted,
            orientation="h",
            marker_color=colors,
            text=text_vals,
            textposition="auto",
            customdata=z_sorted,
            hovertemplate=hover_text,
        )
    )

    fig.add_vline(x=0, line_color="#374151", line_width=1)
    fig.add_vline(x=1.5, line_dash="dot", line_color="#f59e0b", line_width=1, opacity=0.4)
    fig.add_vline(x=-1.5, line_dash="dot", line_color="#f59e0b", line_width=1, opacity=0.4)
    fig.add_vline(x=3.0, line_dash="dot", line_color="#ef4444", line_width=1, opacity=0.4)
    fig.add_vline(x=-3.0, line_dash="dot", line_color="#ef4444", line_width=1, opacity=0.4)

    x_range = min(Z_CAP * 1.25, max(4.0, float(np.max(np.abs(z_display))) * 1.25))

    fig.update_layout(
        xaxis_title="z-score vs. baseline saudável (σ)",
        yaxis_title=None,
        xaxis=dict(range=[-x_range, x_range], zeroline=False),
        height=380,
        margin=dict(l=10, r=70, t=20, b=40),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#e5e7eb", size=11),
        xaxis_gridcolor="#1f2430",
        showlegend=False,
    )
    return fig


def _fig_score_hist(scores: np.ndarray, selected_idx: int, threshold: float) -> go.Figure:
    percentile = float((scores < scores[selected_idx]).mean() * 100)
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=scores,
            nbinsx=40,
            marker_color="#3b82f6",
            opacity=0.75,
            name="Scores",
        )
    )
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="#ef4444",
        line_width=1.5,
        annotation_text="Limite",
        annotation_position="top right",
        annotation_font_color="#9ba3af",
        annotation_font_size=10,
    )
    fig.add_vline(
        x=scores[selected_idx],
        line_color="#fbbf24",
        line_width=2,
        annotation_text=f"#{selected_idx} (p{percentile:.0f})",
        annotation_position="top left",
        annotation_font_color="#fbbf24",
        annotation_font_size=10,
    )
    fig.update_layout(
        xaxis_title="Score",
        yaxis_title="Contagem",
        height=380,
        margin=dict(l=40, r=20, t=20, b=40),
        showlegend=False,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#e5e7eb", size=11),
        xaxis_gridcolor="#1f2430",
        yaxis_gridcolor="#1f2430",
    )
    return fig


def _fig_score_distribution(
    scores_h: np.ndarray,
    scores_d: np.ndarray,
    threshold: float,
    bearing_id: int,
    auc: float | None,
) -> go.Figure:
    """Overlapping score distributions: healthy (y=0) vs. degraded (y=1)."""
    auc_str = f"AUC = {auc:.4f}" if auc is not None else "AUC = N/A"
    fig = go.Figure()

    if len(scores_h) > 0:
        fig.add_trace(
            go.Histogram(
                x=scores_h,
                name="Saudável",
                histnorm="probability density",
                marker_color="rgba(59,130,246,0.55)",
                nbinsx=30,
                opacity=0.85,
            )
        )
    if len(scores_d) > 0:
        fig.add_trace(
            go.Histogram(
                x=scores_d,
                name="Degradado",
                histnorm="probability density",
                marker_color="rgba(239,68,68,0.55)",
                nbinsx=30,
                opacity=0.85,
            )
        )

    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="#ef4444",
        line_width=1.5,
        annotation_text=f"Limite {threshold:.4f}",
        annotation_position="top right",
        annotation_font_color="#9ba3af",
        annotation_font_size=10,
    )

    fig.update_layout(
        barmode="overlay",
        title=dict(text=auc_str, font=dict(size=11, color="#9ba3af"), x=0.0, xanchor="left"),
        xaxis_title="Score",
        yaxis_title="Densidade",
        height=300,
        margin=dict(l=50, r=20, t=40, b=40),
        showlegend=True,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#e5e7eb", size=11),
        xaxis_gridcolor="#1f2430",
        yaxis_gridcolor="#1f2430",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.0,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


def _fig_score_over_time_by_bearing(
    scores: np.ndarray,
    meta: pd.DataFrame,
    threshold: float,
    thresholds_by_bearing: dict[int, float] | None = None,
) -> go.Figure | None:
    """Multi-bearing score chart with per-bearing threshold lines."""
    if "_meta_bearing_id" not in meta.columns or "_meta_timestamp" not in meta.columns:
        return None

    fig = go.Figure()

    for bid in sorted(meta["_meta_bearing_id"].unique()):
        mask = (meta["_meta_bearing_id"] == bid).values
        ts = pd.to_datetime(meta.loc[mask, "_meta_timestamp"])
        s = scores[mask]
        color = _BEARING_COLORS.get(int(bid), "#9ca3af")
        fig.add_trace(
            go.Scatter(
                x=ts,
                y=s,
                mode="lines+markers",
                marker=dict(size=3, color=color),
                line=dict(color=color, width=2 if bid == 1 else 1.3),
                name=f"Bearing {bid}",
            )
        )

    if thresholds_by_bearing:
        for bid in sorted(thresholds_by_bearing.keys()):
            thr_val = thresholds_by_bearing[bid]
            color = _BEARING_COLORS.get(bid, "#9ca3af")
            mask = (meta["_meta_bearing_id"] == bid).values
            if mask.any():
                ts_bear = pd.to_datetime(meta.loc[mask, "_meta_timestamp"])
                x0 = ts_bear.min()
                x1 = ts_bear.max()
                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1],
                        y=[thr_val, thr_val],
                        mode="lines",
                        line=dict(color=color, width=1, dash="dot"),
                        showlegend=False,
                        hovertemplate=f"Limite B{bid}: {thr_val:.4f}<extra></extra>",
                        opacity=0.55,
                    )
                )
    else:
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="#ef4444",
            line_width=1.2,
            annotation_text="Limite",
            annotation_position="top right",
            annotation_font_color="#9ba3af",
            annotation_font_size=10,
        )

    fig.update_layout(
        xaxis_title="Data",
        yaxis_title="Score",
        height=340,
        margin=dict(l=50, r=20, t=30, b=40),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#e5e7eb", size=11),
        xaxis_gridcolor="#1f2430",
        yaxis_gridcolor="#1f2430",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# UI sections
# ---------------------------------------------------------------------------
def _hero(
    scores: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    bearing_id: int | None = None,
    timestamps: pd.DatetimeIndex | None = None,
    state: str | None = None,
    recent_rate: float | None = None,
    scores_full: np.ndarray | None = None,
    y_full: np.ndarray | None = None,
    timestamps_full: pd.DatetimeIndex | None = None,
) -> None:
    """Render the page-top status banner.

    ``state`` / ``recent_rate`` come from :func:`_bearing_state` evaluated on the
    bearing's *full* score history (not just the test slice). The test slice
    alone biases towards the end of the run and would push every bearing into
    "falha" — contradicting the IMS ground truth that only B1 fails.

    When ``scores_full`` / ``y_full`` / ``timestamps_full`` are provided, the
    first-detection timestamp and the recall/FP metrics are computed on the
    full bearing history. Otherwise we fall back to the test slice.
    """
    bearing_label = f"Bearing {bearing_id}" if bearing_id else "Rolamento"

    use_full = (
        scores_full is not None
        and y_full is not None
        and timestamps_full is not None
        and len(scores_full) == len(timestamps_full) == len(y_full)
    )
    s_eval = scores_full if use_full else scores
    y_eval = y_full if use_full else y_test
    ts_eval = timestamps_full if use_full else timestamps

    above_eval = s_eval >= threshold
    has_ts = ts_eval is not None and len(ts_eval) == len(s_eval)

    if state is None:
        state, recent_rate, _ = _bearing_state(scores, threshold)

    def _card(cls: str, label: str, title: str, meta: str = "") -> str:
        meta_html = f'<div class="meta">{meta}</div>' if meta else ""
        return (
            f'<div class="status-card {cls}">'
            f'<span class="label">{label}</span>'
            f'<span class="title">{title}</span>'
            f"{meta_html}</div>"
        )

    if state == _STATE_STABLE:
        st.markdown(
            _card(
                "status-ok",
                "Estável",
                f"{bearing_label} sem anomalias significativas",
                f"Taxa recente acima do limite: {recent_rate:.1%}",
            ),
            unsafe_allow_html=True,
        )
    elif state == _STATE_FAILURE:
        first_idx = _first_sustained_detection(s_eval, threshold) if has_ts else None
        if first_idx is not None and has_ts:
            assert ts_eval is not None
            first_ts_str = ts_eval[first_idx].strftime("%d/%m/%Y %H:%M")
            hours_to_end = (ts_eval[-1] - ts_eval[first_idx]).total_seconds() / 3600
            tp = int((above_eval & (y_eval == 1)).sum())
            n_pos = int((y_eval == 1).sum())
            fp = int((above_eval & (y_eval == 0)).sum())
            n_neg = int((y_eval == 0).sum())
            recall = tp / n_pos if n_pos > 0 else None
            fp_rate = fp / n_neg if n_neg > 0 else None
            parts = [
                f"detecção sustentada desde {first_ts_str}",
                f"{hours_to_end:.0f}h até o fim da gravação",
            ]
            if recall is not None:
                parts.append(f"recall {recall:.0%}")
            if fp_rate is not None:
                parts.append(f"FP {fp_rate:.1%}")
            st.markdown(
                _card("status-warn", "Falha em progressão", bearing_label, " · ".join(parts)),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                _card(
                    "status-warn",
                    "Falha em progressão",
                    bearing_label,
                    f"{recent_rate:.0%} dos snapshots recentes acima do limite",
                ),
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            _card(
                "status-recurrent",
                "Anomalias recorrentes",
                bearing_label,
                f"{recent_rate:.0%} dos snapshots recentes acima do limite · "
                "sem falha documentada pelo paper",
            ),
            unsafe_allow_html=True,
        )


def _kpi_row(
    scores: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    prediction: dict | None,
    selected_idx: int,
    timestamps: pd.DatetimeIndex,
    state: str | None = None,
    recent_rate: float | None = None,
    scores_full: np.ndarray | None = None,
    y_full: np.ndarray | None = None,
) -> None:
    """Render KPI row + prediction card.

    Recall / FP / F1 are computed on the full bearing history when
    ``scores_full``/``y_full`` are provided — the test slice alone has no
    healthy labels for Bearing 1 and is essentially vacuous there. Score máx
    uses the test slice (this is what the dashboard zooms into).
    """
    use_full = scores_full is not None and y_full is not None and len(scores_full) == len(y_full)
    s_eval = scores_full if use_full else scores
    y_eval = y_full if use_full else y_test

    n_total = len(s_eval)
    n_pos = int((y_eval == 1).sum())
    n_neg = n_total - n_pos
    above_eval = s_eval >= threshold
    flagged = int(above_eval.sum())
    tp = int((above_eval & (y_eval == 1)).sum())
    fp = int((above_eval & (y_eval == 0)).sum())
    recall = tp / n_pos if n_pos > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fp_rate = fp / n_neg if n_neg > 0 else 0.0
    flagged_rate = flagged / n_total if n_total > 0 else 0.0
    max_score = float(s_eval.max())
    excess_pct = (max_score - threshold) / max(threshold, 1e-9) * 100

    if state is None:
        state, recent_rate, _ = _bearing_state(scores, threshold)

    kpi1, kpi2, kpi3, kpi4, pred_col = st.columns([1, 1, 1, 1, 2])

    metrics_scope = "histórico completo" if use_full else "slice de teste"
    if n_pos > 0:
        kpi1.metric(
            "Degradações detectadas",
            f"{tp} / {n_pos}",
            f"{recall:.1%} dos rotulados",
            help=(
                f"Snapshots rotulados degradados (últimos 60% do período) que o "
                f"modelo flaggou. Calculado no {metrics_scope}."
            ),
        )
        kpi2.metric(
            "Falsos alarmes",
            fp,
            f"{fp_rate:.1%} dos {n_neg} saudáveis",
            delta_color="inverse",
            help=(
                f"Snapshots saudáveis (primeiros 40%) que o modelo flaggou. "
                f"Calculado no {metrics_scope}."
            ),
        )
        kpi3.metric(
            "F1 Score",
            f"{f1:.1%}",
            help=(
                "Média harmônica entre recall e precisão sobre o rótulo "
                "pseudo-supervisionado (40/60). 100% = modelo perfeito nesse rótulo."
            ),
        )
    else:
        # No documented failure for this bearing — recall/precision/F1 are
        # undefined. Show the model's raw alerting behaviour instead.
        kpi1.metric(
            "Snapshots acima do limite",
            f"{flagged} / {n_total}",
            f"{flagged_rate:.1%} do período",
            delta_color="inverse",
            help=(
                "Sem falha documentada para este rolamento — KPIs de recall/F1 "
                "não se aplicam. Mostrando taxa bruta de alerta."
            ),
        )
        recent_rate_kpi = (
            recent_rate if recent_rate is not None else (_bearing_state(scores, threshold)[1])
        )
        kpi2.metric(
            "Taxa recente",
            f"{recent_rate_kpi:.1%}",
            "últimos 25% do período",
            delta_color="inverse" if recent_rate_kpi >= _RECURRENT_RECENT_RATE else "normal",
            help="Fração de snapshots recentes acima do limite — sinal de drift.",
        )
        kpi3.metric(
            "Status",
            {
                _STATE_FAILURE: "Falha",
                _STATE_RECURRENT: "Recorrente",
                _STATE_STABLE: "Estável",
            }.get(state or _STATE_STABLE, "—"),
            help="Classificação automática a partir do histórico completo do rolamento.",
        )

    kpi4.metric(
        "Score máximo",
        f"{max_score:.4f}",
        f"{excess_pct:+.0f}% vs. limite",
        delta_color="inverse" if excess_pct > 0 else "normal",
        help="Score de anomalia mais alto no período. Positivo = acima do limiar de alerta.",
    )

    with pred_col:
        if prediction is not None:
            pred_ts_str = prediction["predicted_ts"].strftime("%d/%m/%Y %H:%M")
            h = prediction["hours_away"]
            st.markdown(
                '<div class="info-card pred-card">'
                '<div class="heading">Falha prevista</div>'
                f'<div class="value">~{h:.0f}h</div>'
                f'<div class="sub">{pred_ts_str}</div>'
                f'<div class="sub">tendência dos últimos {prediction["n_trend"]} snapshots · '
                f"slope {prediction['slope']:.4f}</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        elif state == _STATE_FAILURE:
            detected_line = ""
            above_idx = np.flatnonzero(scores >= threshold)
            if len(above_idx) > 0 and len(timestamps) == len(scores):
                first_ts = timestamps[int(above_idx[0])]
                hours_since = (timestamps[-1] - first_ts).total_seconds() / 3600
                detected_line = f'<div class="sub">Detectado há {hours_since:.0f}h</div>'
            st.markdown(
                '<div class="info-card fail-card">'
                '<div class="heading">Falha em progressão</div>'
                f"{detected_line}"
                f'<div class="sub">{recent_rate:.0%} dos snapshots recentes acima do limite</div>'
                f'<div class="sub">Score máx. {max_score:.4f} ({excess_pct:+.0f}% vs. limite)</div>'
                "</div>",
                unsafe_allow_html=True,
            )
        elif state == _STATE_RECURRENT:
            st.markdown(
                '<div class="info-card recur-card">'
                '<div class="heading">Anomalias recorrentes</div>'
                f'<div class="sub">{recent_rate:.0%} dos snapshots recentes acima do limite</div>'
                f'<div class="sub">Score máx. {max_score:.4f} ({excess_pct:+.0f}% vs. limite)</div>'
                '<div class="sub">Sem falha documentada pelo paper neste rolamento</div>'
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            sel_ts = (
                timestamps[selected_idx].strftime("%d/%m/%Y %H:%M")
                if len(timestamps) == len(scores)
                else f"#{selected_idx}"
            )
            st.markdown(
                '<div class="info-card ok-card">'
                '<div class="heading">Tendência estável</div>'
                f'<div class="sub">{recent_rate:.0%} dos snapshots recentes acima do limite</div>'
                f'<div class="sub">Snapshot atual: {sel_ts}</div>'
                f'<div class="sub">Score: {scores[selected_idx]:.4f}</div>'
                "</div>",
                unsafe_allow_html=True,
            )


def _render_auto_diagnosis(
    scores: np.ndarray,
    timestamps: pd.DatetimeIndex,
    threshold: float,
    bearing_id: int | None,
    X_bear: np.ndarray,
    X_healthy: np.ndarray,
    feature_names: list[str],
    state: str | None = None,
) -> None:
    """Render a natural-language paragraph summarizing the bearing's condition.

    Phrasing adapts to ``state``: a recurrent bearing isn't described as
    "failure detected" — the paper doesn't document a failure there.
    """
    above = scores >= threshold
    has_ts = len(timestamps) == len(scores)

    if not above.any():
        st.markdown(
            '<div class="diag-box">Nenhuma anomalia detectada para este rolamento '
            "no período analisado. Score máximo permanece abaixo do limiar calibrado.</div>",
            unsafe_allow_html=True,
        )
        return

    first_idx = int(np.argmax(above))
    max_idx = int(np.argmax(scores))
    first_score = float(scores[first_idx])
    max_score = float(scores[max_idx])
    excess_pct = (max_score - threshold) / max(threshold, 1e-9) * 100

    mean_h = X_healthy.mean(axis=0)
    std_h = X_healthy.std(axis=0) + 1e-9
    z_worst = (X_bear[max_idx] - mean_h) / std_h
    dom_idx = int(np.argmax(np.abs(z_worst)))
    dom_feat = feature_names[dom_idx]
    dom_label = _FEATURE_LABELS.get(dom_feat, dom_feat)
    dom_z = float(z_worst[dom_idx])
    hint = _FAILURE_HINTS.get(dom_feat, "padrão anômalo detectado")

    bearing_str = f"**Bearing {bearing_id}**" if bearing_id else "**este rolamento**"
    parts: list[str] = []
    is_failure = state == _STATE_FAILURE

    if is_failure:
        if has_ts:
            first_ts_str = timestamps[first_idx].strftime("%d/%m/%Y às %H:%M")
            hours_early = (timestamps[-1] - timestamps[first_idx]).total_seconds() / 3600
            parts.append(
                f"O modelo detectou a primeira anomalia no {bearing_str} em "
                f"**{first_ts_str}**, com **{hours_early:.0f} horas** de antecedência "
                f"em relação ao fim do período monitorado."
            )
        else:
            parts.append(
                f"O modelo detectou a primeira anomalia no {bearing_str} no snapshot **#{first_idx}**."
            )
        if has_ts:
            max_ts_str = timestamps[max_idx].strftime("%d/%m/%Y às %H:%M")
            parts.append(
                f"O score cresceu de **{first_score:.4f}** (1ª detecção) para "
                f"**{max_score:.4f}** no pico (**{max_ts_str}**) — "
                f"**{excess_pct:+.0f}%** acima do limiar de {threshold:.4f}."
            )
        else:
            parts.append(
                f"O score cresceu de **{first_score:.4f}** (1ª detecção) para "
                f"**{max_score:.4f}** no pico — **{excess_pct:+.0f}%** acima do limiar de {threshold:.4f}."
            )
    else:
        # Recurrent or stable-but-some-flags: report the model output without
        # claiming a failure that the paper doesn't document.
        n_flag = int(above.sum())
        flag_rate = n_flag / len(scores)
        parts.append(
            f"O modelo flagga **{n_flag}** snapshots ({flag_rate:.1%} do período) acima do limite "
            f"para o {bearing_str}, com score máximo de **{max_score:.4f}** "
            f"(**{excess_pct:+.0f}%** vs. limiar de {threshold:.4f})."
        )
        parts.append(
            "O paper IMS Run 2 não documenta falha neste rolamento. Picos podem "
            "refletir drift operacional, mudança de regime ou acoplamento mecânico "
            "via eixo com o rolamento que falha (B1)."
        )

    parts.append(
        f"A feature de maior desvio no pico é **{dom_label}** "
        f"(z = **{dom_z:+.1f}σ**), consistente com {hint}."
    )

    st.markdown(
        f'<div class="diag-box">{" ".join(parts)}</div>',
        unsafe_allow_html=True,
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
    X_healthy: np.ndarray | None = None,
    state: str | None = None,
) -> None:
    has_ts = len(timestamps) == len(scores)
    sel_score = float(scores[sel])
    is_anom = sel_score >= threshold
    # For non-failure bearings, "ANOMALIA DETECTADA" overstates a score that
    # crossed the p99 limit — by design ~1% of healthy snapshots will. Use the
    # softer "ACIMA DO LIMITE" so the language stays honest. Same for the
    # caption's "Diagnóstico do modelo" line.
    documented_failure = state == _STATE_FAILURE
    if is_anom:
        badge_color = "red" if documented_failure else "orange"
        badge_text = "ANOMALIA DETECTADA" if documented_failure else "ACIMA DO LIMITE"
        diag_text = "FALHA" if documented_failure else "score acima do limite"
    else:
        badge_color = "green"
        badge_text = "Normal"
        diag_text = "normal"
    label_text = "degradado (rótulo)" if y_test[sel] else "saudável (rótulo)"

    ts_str = timestamps[sel].strftime("%d/%m/%Y %H:%M") if has_ts else f"#{sel}"
    bearing_id = (
        int(meta["_meta_bearing_id"].iloc[sel]) if "_meta_bearing_id" in meta.columns else "—"
    )
    snap_idx = (
        int(meta["_meta_snapshot_idx"].iloc[sel]) if "_meta_snapshot_idx" in meta.columns else sel
    )

    st.subheader(f"Snapshot #{sel} de {len(X_test)} — :{badge_color}[{badge_text}]")

    info_cols = st.columns(4)
    info_cols[0].metric("Data / hora", ts_str)
    info_cols[1].metric("Anomaly score", f"{sel_score:.4f}")
    info_cols[2].metric("Rolamento", f"Bearing {bearing_id}")
    info_cols[3].metric("Índice temporal", f"#{snap_idx}")

    st.caption(
        f"**Rótulo real:** {label_text} · "
        f"**Diagnóstico do modelo:** {diag_text} · "
        f"**Limiar:** {threshold:.4f}"
    )

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Desvio por feature**")
        if X_healthy is not None and len(X_healthy) > 0:
            st.plotly_chart(
                _fig_feature_bar(X_test[sel], X_healthy, feature_names, sel),
                use_container_width=True,
            )
            st.caption("Vermelho ≥ 3σ · amber ≥ 1.5σ · capado em ±15σ.")
        else:
            st.info("Baseline saudável não disponível.")

    with col_r:
        st.markdown("**Posição na distribuição**")
        st.plotly_chart(_fig_score_hist(scores, sel, threshold), use_container_width=True)
        st.caption("Linha amber marca o snapshot atual; vermelho = limite.")


def _shap_expander(
    model_name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    sel: int,
    X_healthy: np.ndarray | None = None,
) -> None:
    with st.expander("Explicabilidade SHAP — por que este snapshot?"):
        if model_name in _SLOW_MODELS:
            st.info(
                f"**{model_name}** usa KernelExplainer (model-agnostic). "
                "Pode levar 15–30 s. IsolationForest usa TreeExplainer (instantâneo)."
            )
        if st.button("Calcular SHAP para este snapshot", key="shap_btn"):
            with st.spinner("Calculando SHAP values…"):
                model = load_model(model_name)
                # Prefer the global healthy baseline; fall back to test-set y=0 rows
                if X_healthy is not None and len(X_healthy) >= 10:
                    X_bg = X_healthy[:50]
                else:
                    X_bg = X_test[y_test == 0][:50]
                try:
                    import shap as _shap

                    from src.explain import explain

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
                        "Barras vermelhas empurram o score para cima (mais anômalo). "
                        "Barras azuis puxam para baixo (mais normal). "
                        "O ponto de partida E[f(x)] é a média do modelo no conjunto de referência."
                    )
                except Exception as exc:
                    st.error(f"SHAP falhou: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("Detecção preditiva de falhas em rolamentos")
    st.caption(
        "Modelo não supervisionado treinado só em dados saudáveis. "
        "Dataset IMS/NASA Run 2 · 12–19 fev 2004 · 984 snapshots (1 s @ 20 kHz, a cada 10 min) "
        "× 4 rolamentos · "
        "[código](https://github.com/RenanMiqueloti/industrial-anomaly-detection)"
    )

    # --- Load artifacts ---
    data = load_test_data()
    if data is None:
        st.info(
            "**Artefatos não encontrados.** Execute o pipeline primeiro:\n\n"
            "```bash\n"
            "make download\nmake features\nmake train\nmake compare\n"
            "```\n\nDepois reinicie o dashboard."
        )
        st.stop()

    X_test, y_test, meta_test, feature_names = data
    n_test = len(X_test)

    # Load global healthy baseline (y=0 rows from full dataset)
    X_healthy = load_healthy_baseline()

    # --- Sidebar ---
    with st.sidebar:
        st.header("Configurações")
        model_name = st.selectbox("Modelo", list(_MODEL_FILES.keys()), index=0)

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
            selected_bearing = None
            bear_mask = np.ones(n_test, dtype=bool)

        scores_all = compute_scores(model_name, X_test.tobytes(), n_test)
        if scores_all is None:
            st.warning(f"Modelo **{model_name}** não encontrado. Execute `make compare`.")
            st.stop()

        scores = scores_all[bear_mask]
        y_bear = y_test[bear_mask]
        meta_bear = meta_test.iloc[bear_mask].reset_index(drop=True)
        timestamps = _get_timestamps(meta_bear, len(scores))

        _thr_default = float(
            np.clip(
                _default_threshold(
                    scores,
                    y_bear,
                    model_name,
                    bearing_id=selected_bearing
                    if "_meta_bearing_id" in meta_test.columns
                    else None,
                ),
                scores.min(),
                scores.max(),
            )
        )
        threshold = st.slider(
            "Limite de anomalia",
            min_value=float(scores.min()),
            max_value=float(scores.max()),
            value=_thr_default,
            format="%.4f",
            help="Snapshots com score acima deste valor são flagged. "
            "O padrão é o 99º percentil dos snapshots saudáveis de calibração "
            "(primeiros 40% do período — limita FP a ≤1% nesse subconjunto).",
        )
        st.caption(f"Padrão calibrado: **{_thr_default:.4f}** (p99 dos saudáveis)")

        st.divider()
        st.caption(
            "Ajuste o limite, observe onde as anomalias aparecem, clique num ponto "
            "da timeline para inspecionar."
        )

    # Reset selected snapshot when the user switches bearing
    if st.session_state.get("_last_bearing") != selected_bearing:
        st.session_state["_last_bearing"] = selected_bearing
        st.session_state.pop("selected_idx", None)

    # --- Bearing state from FULL history (not just test slice) ---
    # The test slice is the chronological tail of the run — using it alone
    # biases every bearing toward "falha". Full-history scoring gives the
    # honest current state.
    full_data = compute_full_dataset_scores(model_name)
    full_meta = load_full_metadata()
    bearing_state_val: str | None = None
    recent_rate_val: float | None = None
    scores_full_bear_sorted: np.ndarray | None = None
    y_full_bear_sorted: np.ndarray | None = None
    ts_full_bear_sorted: pd.DatetimeIndex | None = None

    if full_data is not None and selected_bearing is not None:
        full_scores_all, full_y_all, full_bids_all = full_data
        bear_full_mask_state = full_bids_all == selected_bearing
        scores_full_for_state = full_scores_all[bear_full_mask_state]
        if full_meta is not None and "_meta_timestamp" in full_meta.columns:
            ts_bear_unsorted = pd.to_datetime(
                full_meta.loc[bear_full_mask_state, "_meta_timestamp"].values
            )
            order = np.argsort(ts_bear_unsorted.values)
            scores_full_bear_sorted = scores_full_for_state[order]
            y_full_bear_sorted = full_y_all[bear_full_mask_state][order]
            ts_full_bear_sorted = pd.DatetimeIndex(ts_bear_unsorted.values[order])
        else:
            scores_full_bear_sorted = scores_full_for_state
            y_full_bear_sorted = full_y_all[bear_full_mask_state]
        bearing_state_val, recent_rate_val, _ = _bearing_state(scores_full_bear_sorted, threshold)

    # --- Prediction ---
    # Only run the linear-extrapolation failure projection when the bearing is
    # actually in failure state. Otherwise a noisy upward trend on a healthy
    # bearing can produce a "Falha prevista em Xh" card — exactly the kind
    # of overclaim the three-tier state classifier exists to prevent.
    if bearing_state_val == _STATE_FAILURE:
        prediction = _predict_failure(scores, timestamps, threshold)
    else:
        prediction = None

    # --- Hero (status + key facts) ---
    _hero(
        scores,
        y_bear,
        threshold,
        bearing_id=selected_bearing,
        timestamps=timestamps,
        state=bearing_state_val,
        recent_rate=recent_rate_val,
        scores_full=scores_full_bear_sorted,
        y_full=y_full_bear_sorted,
        timestamps_full=ts_full_bear_sorted,
    )

    # --- Init selected snapshot ---
    if "selected_idx" not in st.session_state:
        first_anom = np.where((scores >= threshold) & (y_bear == 1))[0]
        st.session_state.selected_idx = int(first_anom[0]) if len(first_anom) else 0
    st.session_state.selected_idx = int(np.clip(st.session_state.selected_idx, 0, len(scores) - 1))

    # --- KPIs + prediction card ---
    _kpi_row(
        scores,
        y_bear,
        threshold,
        prediction,
        st.session_state.selected_idx,
        timestamps,
        state=bearing_state_val,
        recent_rate=recent_rate_val,
        scores_full=scores_full_bear_sorted,
        y_full=y_full_bear_sorted,
    )

    st.divider()

    # --- Auto-diagnosis + Separability chart (reuses full history computed above) ---
    if (
        scores_full_bear_sorted is not None
        and y_full_bear_sorted is not None
        and X_healthy is not None
        and selected_bearing is not None
    ):
        scores_h_dist = scores_full_bear_sorted[y_full_bear_sorted == 0]
        scores_d_dist = scores_full_bear_sorted[y_full_bear_sorted == 1]
        auc = _safe_auc(y_full_bear_sorted, scores_full_bear_sorted)

        diag_col, sep_col = st.columns([2, 3])

        with diag_col:
            st.subheader("Auto-diagnóstico")
            _render_auto_diagnosis(
                scores=scores,
                timestamps=timestamps,
                threshold=threshold,
                bearing_id=selected_bearing,
                X_bear=X_test[bear_mask],
                X_healthy=X_healthy,
                feature_names=feature_names,
                state=bearing_state_val,
            )
            if auc is not None:
                st.caption(
                    f"AUC no dataset completo (train+test): **{auc:.4f}**. "
                    "Valores acima de 0,80 indicam forte separabilidade entre saudável e degradado."
                )
            else:
                st.caption(
                    "AUC indisponível: este rolamento não tem falha documentada "
                    "no paper, então a métrica de separabilidade não se aplica."
                )

        with sep_col:
            st.subheader("Separabilidade")
            st.plotly_chart(
                _fig_score_distribution(
                    scores_h_dist,
                    scores_d_dist,
                    threshold,
                    bearing_id=selected_bearing,
                    auc=auc,
                ),
                use_container_width=True,
            )
            st.caption(
                "Distribuição dos scores no dataset completo. "
                "Menos sobreposição = melhor separação."
            )

        st.divider()

    # --- Multi-bearing overview ---
    thresholds_by_bearing = _load_all_thresholds(model_name)
    multi_fig = _fig_score_over_time_by_bearing(
        scores_all,
        meta_test,
        threshold,
        thresholds_by_bearing=thresholds_by_bearing if thresholds_by_bearing else None,
    )
    if multi_fig is not None:
        st.subheader("Visão geral · 4 rolamentos")
        st.caption(
            "Linhas pontilhadas = limiar p99 por rolamento. "
            "Apenas Bearing 1 tem falha documentada pelo paper (pista externa)."
        )
        st.plotly_chart(multi_fig, use_container_width=True)
        st.divider()

    _bearing_label = f"Bearing {selected_bearing}" if selected_bearing else "Rolamento"
    st.subheader(f"Timeline · {_bearing_label}")
    st.caption("Cada ponto = 1 snapshot (1 s a 20 kHz). Clique para inspecionar.")

    timeline_fig = _fig_timeline(
        scores,
        meta_bear,
        timestamps,
        y_bear,
        threshold,
        st.session_state.selected_idx,
        prediction,
    )
    event = st.plotly_chart(
        timeline_fig, use_container_width=True, on_select="rerun", key="timeline"
    )

    # Streamlit's PlotlyState stubs don't expose `.selection` even though the
    # runtime object carries it when `on_select="rerun"` is set.
    if event and event.selection and event.selection.points:  # type: ignore[attr-defined]
        pt = event.selection.points[0]  # type: ignore[attr-defined]
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
        X_healthy=X_healthy,
        state=bearing_state_val,
    )

    # --- SHAP ---
    _shap_expander(
        model_name,
        X_test[bear_mask],
        y_bear,
        feature_names,
        st.session_state.selected_idx,
        X_healthy=X_healthy,
    )

    # --- Context / methodology ---
    st.divider()
    with st.expander("Pipeline e metodologia"):
        st.markdown(
            """
**Extração de features (por snapshot):**

Cada snapshot = 1 segundo de vibração a 20 kHz (20 480 amostras). São extraídas 11 features:

| Grupo | Features |
|-------|---------|
| Domínio do tempo | RMS · Pico · Fator de crista · Curtose · Assimetria · Desvio-padrão · Pico-a-pico |
| Energia espectral | 0–500 Hz · 500–2k Hz · 2–5 kHz · 5–10 kHz |

**Bandas de frequência — o que cada uma captura:**

Frequências de defeito do rolamento Rexnord ZA-2115 a 2000 RPM (Qiu et al. 2006,
Kalikatzarakis 2018): BPFO ≈ 236 Hz, BPFI ≈ 297 Hz, BSF ≈ 140 Hz (2×BSF ≈ 280 Hz),
FTF ≈ 15 Hz. Todas as fundamentais caem na banda 0–500 Hz.

| Banda | O que indica |
|-------|-------------|
| 0–500 Hz | Fundamentais — BPFO/BPFI/BSF, frequência de rotação, desbalanceamento |
| 500–2k Hz | Harmônicos das frequências de defeito (2× a 5× BPFO/BPFI) |
| 2–5 kHz | Modos ressonantes estruturais (housing/eixo) excitados por impactos |
| 5–10 kHz | Alta frequência — impactos severos, ressonâncias excitadas por dano avançado |

**Pipeline de detecção:**

1. **Treino sem rótulos** — modelo ajustado exclusivamente nos primeiros 40% dos snapshots (período saudável)
2. **Scoring** — cada snapshot recebe um score de anomalia proporcional ao desvio do comportamento de treino
3. **Limiar por rolamento** — p99 dos snapshots de **calibração** (primeiros 40% saudáveis) de cada bearing. Limita FP a ≤ 1% **nesse subconjunto** — a taxa medida no histórico completo pode ser maior em B2/B3/B4 por drift natural dos sinais ao longo da semana
4. **Projeção** — regressão linear nos últimos 25% dos scores projeta o cruzamento do limiar no tempo

**Decisões de design:**
- Split treino/teste **temporal** (70/30) — preserva ordem cronológica, evita data leakage
- Rótulos binários: y=0 (primeiros 40% = saudável), y=1 (restante = potencialmente degradado)
- AUC calculado sobre o dataset completo (train+test) — split temporal coloca todo o período degradado no teste
            """
        )


if __name__ == "__main__":
    main()
