# PLANO.md — sprint roadmap

Sprints are deliverable units, not time boxes. Each one ends with a green CI run, updated docs and (where relevant) a regenerated figure.

---

## Sprint 0 — scaffold *(complete)*

- [x] `pyproject.toml`, `Makefile`, MIT `LICENSE`, `.gitignore`, `.github/workflows/ci.yml`
- [x] `src/features.py` — `time_domain_features`, `fft_band_energy`, `extract_all`, `TimeDomainFeatures` dataclass
- [x] `tests/test_features.py` — physical-truth assertions:
  - RMS of a sine of amplitude A is `A / sqrt(2)`
  - Crest factor of a sine is `sqrt(2)`
  - Kurtosis of `N(0, 1)` is approximately 0 at large N
  - Energy of a 1 kHz sine concentrates in the 500–1500 Hz band
  - 1-D / non-empty input validation
  - Custom bands override the defaults
- [x] README with mermaid architecture diagram, model table and design decisions

---

## Sprint 1 — ingest + IsolationForest + evaluate

**Goal:** end-to-end pipeline that reads CWRU `.mat`, extracts features, fits Isolation Forest on healthy windows and reports ROC-AUC + F1 with bootstrap 95% CI.

- [ ] `src/ingest.py`
  - `load_cwru(root)` → DataFrame with `signal`, `class`, `fault_diameter`, `load_hp`, `rpm`
  - `window(signal, length, hop)` → iterator of fixed-length non-overlapping windows
- [ ] `src/dataset.py`
  - `build_feature_matrix(root, window_len=2048, hop=2048)` → `(X, y, meta)` parquet
- [ ] `src/models/iforest.py`
  - thin wrapper exposing `.fit(X_healthy)` and `.score(X)` (higher = more anomalous)
- [ ] `src/evaluate.py`
  - `bootstrap_ci(scores, y_true, n_resamples=1000, seed=42)` → `{"roc_auc": (mean, low, high), ...}`
- [ ] `make data` downloads the dataset (or fetches a cached mirror)
- [ ] `make features` produces `data/features/cwru.parquet`
- [ ] `make train` writes `results/models/iforest.pkl`
- [ ] `make eval` writes `results/figures/iforest_roc.png` + `results/iforest_metrics.json`
- [ ] `tests/test_pipeline.py` — runs end-to-end on a 100-window synthetic fixture, asserts ROC-AUC > 0.7

**Definition of done:** `make data features train eval` works on a fresh clone; CI runs green on `main`.

---

## Sprint 2 — One-Class SVM, LOF, AutoEncoder

**Goal:** three more models behind a common `BaseDetector` interface, plus a comparison report.

- [ ] `src/models/base.py` — abstract `BaseDetector` with `fit` / `score`
- [ ] `src/models/ocsvm.py` — wrapper over `sklearn.svm.OneClassSVM` with sensible defaults (RBF, ν=0.1, γ='scale')
- [ ] `src/models/lof.py` — wrapper over `sklearn.neighbors.LocalOutlierFactor(novelty=True)`
- [ ] `src/models/autoencoder.py`
  - PyTorch MLP autoencoder (input → 16 → 8 → 16 → input), early stopping
  - Reconstruction-error scoring
- [ ] `src/compare.py` — runs all four detectors with shared CV split, writes `results/comparison.parquet`
- [ ] `results/figures/model_comparison.png` — bar chart with bootstrap CI error bars
- [ ] `tests/test_models.py` — each detector fits and scores on the synthetic fixture

**Definition of done:** comparison figure committed (rendered, not regenerated in CI), README updated with the actual numbers (no placeholders).

---

## Sprint 3 — SHAP explainability

**Goal:** per-prediction explanations for the best-performing model.

- [ ] `src/explain.py`
  - `TreeExplainer` for Isolation Forest
  - `KernelExplainer` fallback for the AutoEncoder
  - common API: `explain(model, X, feature_names) → shap.Explanation`
- [ ] `make explain` writes `results/figures/shap_summary.png` + `results/figures/shap_per_fault_<class>.png` (one per fault mode)
- [ ] `tests/test_explain.py` — asserts that, on a synthetic dataset where only `rms` differs, SHAP attributes the anomaly to `rms` (sanity check)

**Definition of done:** README has a screenshot and a one-paragraph interpretation of which features drive each fault class.

---

## Sprint 4 — Streamlit dashboard + Docker

**Goal:** reproducible deployment and a UI an operations team would actually open.

- [ ] `src/dashboard.py`
  - Sidebar: pick model, threshold, fault class
  - Main: time-series plot, anomaly score histogram, SHAP per-window plot
- [ ] `Dockerfile` (`python:3.12-slim`, non-root, healthcheck on `/_stcore/health`)
- [ ] `docker-compose.yml` — single service `dashboard`, env-driven config
- [ ] `make dashboard` → `streamlit run src/dashboard.py`
- [ ] README "Production deployment" section with `docker compose up`

**Definition of done:** dashboard URL screenshot in README; container starts with `docker compose up` on a clean machine.

---

## Sprint 5 *(optional)* — FastAPI streaming + drift detection

**Goal:** push beyond the offline-evaluation baseline.

- [ ] `src/api.py` — FastAPI with `POST /score` (raw signal → anomaly score) and `WS /stream` (windowed signals → live scores)
- [ ] `src/drift.py` — Population Stability Index over the feature matrix, alert when PSI > 0.2 on any feature
- [ ] `tests/test_api.py` — TestClient covers both endpoints with synthetic payloads

**Definition of done:** API stays optional behind an extra dependency group; the core analysis path remains pip-installable without FastAPI.
