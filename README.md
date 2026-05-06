# industrial-anomaly-detection

![CI](https://github.com/RenanMiqueloti/industrial-anomaly-detection/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)

**Unsupervised anomaly detection on industrial vibration time-series.** Compares Isolation Forest, One-Class SVM, Local Outlier Factor and a small AutoEncoder on the [CWRU bearing dataset](https://engineering.case.edu/bearingdatacenter), with handcrafted features (RMS, FFT band energy, kurtosis), SHAP explanations, and bootstrap confidence intervals on the metrics.

> **Status:** Sprint 1 done — IsolationForest baseline with bootstrap CI. Feature extraction module is in place and tested against signal-processing identities. See [PLANO.md](PLANO.md) for the next sprints.

---

## Why this matters

In industrial predictive maintenance, **labelled fault data is rare** — by the time a bearing fails enough times to be labelled, you are already losing money. Unsupervised models trained only on healthy data can flag anomalies before failure, with no labelled rollout cost.

The CWRU dataset is the de facto benchmark for bearing diagnostics: drive-end accelerometer at 12 kHz, four classes (healthy, inner race, outer race, ball), several fault diameters and motor loads. It's small, public, and well-instrumented — perfect for honest model comparison.

---

## Architecture

```mermaid
graph LR
    A([raw .mat<br/>CWRU]) --> B
    B[ingest<br/>load + window] --> C
    C[features<br/>RMS · FFT bands · kurtosis] --> D
    D[scale<br/>RobustScaler] --> E
    E[fit<br/>healthy windows only] --> F
    F[score<br/>anomaly score] --> G
    G[evaluate<br/>ROC-AUC · F1 · IC bootstrap]
    F --> H
    H[explain<br/>SHAP TreeExplainer]
    H --> I([per-fault<br/>feature attribution])

    style C fill:#1e293b,color:#e2e8f0
    style E fill:#1e293b,color:#e2e8f0
    style G fill:#1e293b,color:#e2e8f0
    style H fill:#1e293b,color:#e2e8f0
```

---

## Models compared

| Model | Why it might win | Why it might lose |
|---|---|---|
| **Isolation Forest** | Robust to high-dimensional, low-sample regimes; no kernel tuning. | Axis-aligned splits miss interactions. |
| **One-Class SVM (RBF)** | Captures non-linear boundaries with the right kernel. | Sensitive to ν / γ; expensive on large training sets. |
| **Local Outlier Factor** | Local density makes it good for clustered failure modes. | Doesn't generalize to unseen test points without `novelty=True`. |
| **AutoEncoder (PyTorch)** | Reconstruction error encodes complex non-linear normality. | Easily overfits with small healthy sets; needs early stopping. |

Each is fit **only on healthy windows** and evaluated on a held-out mix of healthy + faulty windows.

---

## Features

Implemented in [`src/features.py`](src/features.py), tested against physical truths in [`tests/test_features.py`](tests/test_features.py):

**Time-domain** (7 statistics): `rms`, `peak`, `crest_factor`, `kurtosis`, `skewness`, `std`, `p2p`.

**Frequency-domain** (band energy via Welch's PSD, default bands `0–500 / 500–1500 / 1500–3000 / 3000–6000` Hz). Bands map roughly onto the BPFO/BPFI/BSF/FTF families when motor load is held constant.

```python
from src.features import extract_all

feats = extract_all(window, fs=12_000)  # → dict[str, float]
```

---

## Quick start

```bash
git clone https://github.com/RenanMiqueloti/industrial-anomaly-detection.git
cd industrial-anomaly-detection
make install           # pip install -e ".[dev]"
make test              # pytest -v --cov=src tests/
```

Sprint 1 targets are now functional:

```bash
make data        # clone CWRU mirror → data/raw/
make features    # extract features → data/features/features.parquet
make train       # fit IsolationForest on healthy windows → results/iforest_model.joblib
make eval        # bootstrap CI → results/iforest_metrics.json + results/figures/iforest_roc.png
```

The `explain` and `dashboard` targets land in upcoming sprints — see [PLANO.md](PLANO.md).

---

## Reproducing the baseline

```bash
make install
make data
make features
make train
make eval
```

### Baseline metrics (IsolationForest, CWRU subset)

| Metric | Mean | 95% CI low | 95% CI high |
|--------|------|------------|-------------|
| ROC-AUC | _[fill after `make eval`]_ | — | — |
| F1 | _[fill after `make eval`]_ | — | — |

> _[a preencher após download manual da CWRU em `data/raw/` se `make data` falhar]_

---

## Layout

```
industrial-anomaly-detection/
├── src/
│   ├── __init__.py
│   ├── features.py           # time-domain + FFT band energy
│   ├── ingest.py             # load_cwru + window generator
│   ├── dataset.py            # build_feature_matrix → parquet
│   ├── evaluate.py           # bootstrap_ci + plot_roc
│   ├── cli.py                # download | features | train | eval
│   └── models/
│       ├── __init__.py
│       └── iforest.py        # IForestDetector (fit/score/save/load)
├── tests/
│   ├── test_features.py      # 8 physical-truth assertions (Sprint 0)
│   ├── test_ingest.py        # windowing + synthetic .mat loading
│   ├── test_dataset.py       # feature matrix construction
│   ├── test_iforest.py       # fit/score/save/load
│   ├── test_evaluate.py      # bootstrap CI + ROC plot
│   └── test_pipeline.py      # end-to-end synthetic pipeline
├── data/
│   ├── raw/                  # CWRU .mat files (gitignored)
│   └── features/             # parquet feature matrix (gitignored)
├── results/
│   ├── figures/              # ROC plots (gitignored)
│   ├── iforest_model.joblib  # trained model (gitignored)
│   └── iforest_metrics.json  # bootstrap CI results (gitignored)
├── .github/workflows/ci.yml
├── pyproject.toml
├── Makefile
├── PLANO.md
├── LICENSE                   # MIT
└── README.md
```

---

## Design decisions

**Handcrafted features instead of "deep features over the raw waveform".**
On bearing vibration, RMS + crest factor + spectral band energy carry most of the signal. Deep learning needs labelled examples per fault mode; CWRU has ~10⁴ windows total. With <10⁵ samples the bias of physically-motivated features beats the variance of a learned representation. Papers from 2018–2023 keep showing this on small industrial datasets — handcrafted + tree-based ensembles outperform end-to-end CNNs unless the dataset is on the order of millions of windows.

**Unsupervised, not classification.**
Predictive maintenance hits a label cliff: by the time a bearing fails often enough to be labelled, it's too late. Training only on healthy data and flagging deviations is the only protocol that scales to a fleet of unlabelled machines.

**Bootstrap CI on every reported metric.**
Single ROC-AUC numbers without confidence intervals are noise on small datasets. Every figure ships a 95% bootstrap CI computed from 1000 resamples — the goal is reproducibility, not point estimates.

**SHAP for per-prediction explanations.**
For Isolation Forest, `TreeExplainer` gives exact Shapley values in O(TLD²). For the AutoEncoder, KernelExplainer falls back to model-agnostic SHAP. Both ship with the same API for downstream consumers (`shap.summary_plot(...)`).

**Streamlit dashboard, not a notebook.**
Notebooks are for exploration; a small Streamlit app is what stakeholders actually open. The dashboard target in the Makefile is the deliverable an operations team would consume.
