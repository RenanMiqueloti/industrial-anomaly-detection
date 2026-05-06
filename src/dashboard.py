"""Streamlit dashboard for industrial-anomaly-detection.

Usage
-----
    streamlit run src/dashboard.py

Loads the test split saved by ``make train`` plus the model selected in the
sidebar, shows the feature-vector "signal" of the selected window with
above-threshold windows highlighted, the score distribution, and a local
SHAP waterfall for the selected window.

If artifacts are missing (no ``make data features train`` run yet), the app
shows an info banner with instructions rather than crashing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.explain import explain
from src.models.autoencoder import AutoEncoderDetector
from src.models.iforest import IForestDetector
from src.models.lof import LOFDetector
from src.models.ocsvm import OCSVMDetector

# ---------------------------------------------------------------------------
# Paths (relative to repo root; `streamlit run` is always called from there)
# ---------------------------------------------------------------------------
_RESULTS = Path("results")
_DATA_FEATURES = Path("data/features/features.parquet")

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

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Industrial Anomaly Detection",
    page_icon="🔧",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_test_data() -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]] | None:
    """Return (X_test, y_test, meta_test, feature_names) or None if artifacts absent."""
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
        meta_test = pd.DataFrame({"_meta_class": ["unknown"] * len(X_test)})

    return X_test, y_test, meta_test, feature_names


@st.cache_resource
def load_model(model_name: str):
    """Load and return the fitted detector for *model_name*.

    Falls back gracefully if the model file doesn't exist (only IForest is
    guaranteed from ``make train``; the others require ``make compare``).
    """
    path = _MODEL_FILES.get(model_name)
    if path is None or not path.exists():
        return None
    cls = _MODEL_CLASSES[model_name]
    return cls.load(path)


@st.cache_data
def compute_scores(model_name: str, X_bytes: bytes, n_rows: int) -> np.ndarray | None:
    """Score the test set with the selected model.

    ``X_bytes`` is ``X_test.tobytes()`` — passed as bytes so Streamlit can
    hash it correctly (np.ndarray is not natively hashable by st.cache_data).
    """
    model = load_model(model_name)
    if model is None:
        return None
    X = np.frombuffer(X_bytes, dtype=np.float64).reshape(n_rows, -1)
    return model.score(X)


# ---------------------------------------------------------------------------
# Helper: feature-vector bar plot (proxy for the "signal" panel)
# ---------------------------------------------------------------------------
def _plot_window_features(
    X: np.ndarray, idx: int, feature_names: list[str], score: float, threshold: float
) -> plt.Figure:
    row = X[idx]
    color = "#d62728" if score >= threshold else "#1f77b4"
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(range(len(row)), row, color=color, alpha=0.8)
    ax.set_xticks(range(len(row)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
    ax.set_title(
        f"Window {idx} — score={score:.4f} ({'ANOMALY' if score >= threshold else 'normal'})"
    )
    ax.set_ylabel("Feature value (scaled)")
    fig.tight_layout()
    return fig


def _plot_score_histogram(scores: np.ndarray, threshold: float, selected_idx: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(scores, bins=40, alpha=0.7, color="#1f77b4", label="Test scores")
    ax.axvline(threshold, color="#d62728", lw=2, linestyle="--", label=f"Threshold {threshold:.3f}")
    ax.axvline(
        scores[selected_idx],
        color="#ff7f0e",
        lw=1.5,
        linestyle=":",
        label=f"Selected window ({scores[selected_idx]:.3f})",
    )
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Count")
    ax.set_title("Score distribution (test set)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("🔧 Industrial Anomaly Detection")
    st.markdown(
        "Source: [RenanMiqueloti/industrial-anomaly-detection]"
        "(https://github.com/RenanMiqueloti/industrial-anomaly-detection)"
    )

    # --- Load artifacts ---
    data = load_test_data()
    if data is None:
        st.info(
            "**Artifacts not found.** Run the pipeline first:\n\n"
            "```bash\n"
            "make data        # download CWRU bearing data\n"
            "make features    # extract features\n"
            "make train       # fit IsolationForest, save test split\n"
            "make compare     # train all 4 models\n"
            "```\n\n"
            "Then restart the dashboard."
        )
        st.stop()

    X_test, y_test, meta_test, feature_names = data
    n_test = len(X_test)

    # --- Sidebar ---
    with st.sidebar:
        st.header("Controls")

        model_name = st.selectbox(
            "Model",
            list(_MODEL_CLASSES.keys()),
            index=0,
        )

        scores = compute_scores(model_name, X_test.tobytes(), n_test)
        if scores is None:
            st.warning(
                f"Model file not found for **{model_name}**. "
                "Run `make compare` to train all models."
            )
            st.stop()

        score_min, score_max = float(scores.min()), float(scores.max())
        score_median = float(np.median(scores))
        threshold = st.slider(
            "Score threshold",
            min_value=score_min,
            max_value=score_max,
            value=score_median,
            format="%.4f",
        )

        available_classes = sorted(meta_test["_meta_class"].unique().tolist())
        class_filter = st.multiselect(
            "Fault class filter",
            options=available_classes,
            default=available_classes,
        )

        # Apply class filter
        if class_filter:
            mask = meta_test["_meta_class"].isin(class_filter).values
            X_visible = X_test[mask]
            scores_visible = scores[mask]
        else:
            X_visible = X_test
            scores_visible = scores
            mask = np.ones(n_test, dtype=bool)

        visible_indices = np.where(mask)[0]
        n_visible = len(X_visible)

        selected_local = st.number_input(
            "Window index (within filter)",
            min_value=0,
            max_value=max(n_visible - 1, 0),
            value=0,
            step=1,
        )

    selected_global = int(visible_indices[selected_local]) if n_visible > 0 else 0
    selected_score = float(scores[selected_global])
    selected_class = meta_test["_meta_class"].iloc[selected_global]

    # --- Main layout ---
    col_left, col_right = st.columns([6, 4])

    with col_left:
        st.subheader("Window features")
        st.caption(
            f"Index {selected_global} · class: **{selected_class}** · "
            f"ground-truth label: {'faulty' if y_test[selected_global] else 'normal'}"
        )
        fig_win = _plot_window_features(
            X_test, selected_global, feature_names, selected_score, threshold
        )
        st.pyplot(fig_win)
        plt.close(fig_win)

        n_above = int((scores_visible >= threshold).sum())
        st.metric(
            label="Windows above threshold (filtered view)",
            value=n_above,
            delta=f"{n_above / max(n_visible, 1) * 100:.1f}% flagged",
        )

    with col_right:
        st.subheader("Score distribution")
        fig_hist = _plot_score_histogram(scores, threshold, selected_global)
        st.pyplot(fig_hist)
        plt.close(fig_hist)

    # --- SHAP waterfall (full width) ---
    st.subheader("Local SHAP explanation")
    st.caption(f"Window {selected_global} — {model_name}")

    with st.spinner("Computing SHAP values for the selected window…"):
        model = load_model(model_name)
        X_single = X_test[[selected_global]]
        # Use a small background for speed (only this one window is explained)
        X_bg = X_test[y_test == 0][:50]  # healthy background
        try:
            exp_single = explain(
                model,
                X_single,
                feature_names,
                X_background=X_bg,
                bg_size=50,
                eval_size=None,
            )
            import shap as _shap

            fig_shap, _ax_shap = plt.subplots(figsize=(10, 4))
            _shap.plots.waterfall(exp_single[0], show=False)
            fig_shap = plt.gcf()
            st.pyplot(fig_shap)
            plt.close("all")
        except Exception as exc:
            st.warning(f"SHAP explanation failed: {exc}")


if __name__ == "__main__":
    main()
