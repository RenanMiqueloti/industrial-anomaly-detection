# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Three-tier bearing state classifier** in `dashboard.py` (`_bearing_state`)
  driven by the fraction of recent snapshots above threshold:
  `falha` (≥ 60% recent + ≥ 20% excess), `recorrente` (≥ 10%), `estavel`
  (< 10%). Hero, KPI panel, prediction card, detail panel and
  auto-diagnosis all branch on this. Replaces the single
  `excess_pct >= 20` rule that fired on 3 of 4 bearings in IMS Run 2.
- Per-bearing thresholds for OC-SVM and AutoEncoder in `threshold.json`
  (`ocsvm_bN`, `ae_bN`), computed by `compare.py` from the healthy
  training slice — the dashboard's per-bearing slider now works
  consistently across all three models instead of falling back to
  test-set p99 for non-IForest models.
- `documented_failure_bearings` parameter on `build_ims_features` and a
  module-level `IMS_RUN2_FAILURE_BEARINGS = (1,)` constant that encodes
  the IMS Run 2 ground truth.
- Multi-stage `Dockerfile` and `Dockerfile.api` to drop pip toolchain from
  the runtime image (-150 to -300 MB).
- `IAD_MODEL_PATH`, `IAD_THRESHOLD_PATH`, `IAD_RESULTS_DIR` environment
  variables to override hardcoded artifact paths in the API and pipeline.
- `pip-audit` job in CI for dependency vulnerability scanning
  (`continue-on-error: true` while triaging existing advisories).
- Smoke tests for pure helper functions in `dashboard.py`
  (`_safe_auc`, `_default_threshold`, `_load_all_thresholds`,
  `_get_timestamps`, `_predict_failure`).
- Issue and pull-request templates under `.github/`.
- `docs/MODEL_CARD.md` (Mitchell et al. 2019 template) and
  `docs/DATASET.md` (Gebru et al. 2021 datasheet template).
- `scripts/generate_synthetic_dataset.py` + `make demo` for offline runs
  without Kaggle credentials.
- Pre-commit hooks (`ruff`, large-file guard, EOL fixer).
- Dependabot configuration for pip + github-actions + docker.
- Coverage threshold of 85% enforced in CI.

### Fixed
- **`dataset.py` labels contradicted the IMS Run 2 ground truth.** The old
  rule labelled the last 60% of every bearing as `y=1` regardless of
  whether the paper documented a failure there. The paper only documents
  Bearing 1 (outer race). Mislabelling B2/B3/B4 as degraded inflated false
  positives, corrupted recall/F1/AUC on those bearings (IForest test-set
  AUC was 0.41 — anti-correlated with truth), and pushed the dashboard's
  "Falha em progressão" card onto 3 of 4 bearings. After the fix
  `iforest_metrics.json` AUC moves from 0.41 → 0.93.
- **Dashboard's bearing selector lied about state.** The dropdown labelled
  every non-B1 option as `✓ Saudável`, but the model raises sustained
  alerts in B2/B4 (drift / shaft coupling). Selector now shows just the
  bearing number; state is conveyed by the badge after selection.
- **"Falha prevista em Xh" could fire on healthy bearings.** The linear
  extrapolation in `_predict_failure` ran unconditionally. Now gated on
  `_bearing_state == falha` — the projection is only computed for
  bearings that are actually in failure.
- **`_detail_panel` always rendered "ANOMALIA DETECTADA"** for any score
  above threshold. For non-failure bearings this overstates a snapshot
  that crossed the design noise floor; now reads "ACIMA DO LIMITE"
  (orange) when the bearing has no documented failure.
- **`_kpi_row` showed `recall = 0%` and `F1 = 0%`** on B2/B3/B4. With no
  positive class in the ground truth these metrics are undefined, not
  zero. Panel now shows raw alerting behaviour (flagged count, recent
  rate, state) when `n_pos == 0`.
- **Multi-bearing chart title hard-coded "Bearing 1 diverge dos demais"**
  — true under IForest on Run 2, but stuck around when switching models.
  Replaced with a model-neutral description.
- **Data leakage in `compare.py`**: the 4-model comparison was deriving its
  training set via stratified random split, while `cli.py train` uses a
  temporal split by timestamp. The leakage allowed OC-SVM/LOF/AutoEncoder
  to train on windows that were in IsolationForest's test set. Now
  `compare.py` consumes `X_train_healthy.npy` saved by `cli train`.
- **API ignored per-bearing thresholds**: the `/score` endpoint always
  used the global `iforest` key from `threshold.json`, contradicting the
  README/MODEL_CARD claim of "≤ 1% false alarms per bearing". Added
  optional `bearing_id` to `ScoreRequest`; routing falls back to the
  global threshold for unknown bearings.
- **API depended on dict iteration order for feature vectors**: scoring
  silently fed the model an out-of-order feature vector if `extract_all`
  ever changed the dict layout. `threshold.json` now carries
  `feature_order`; the API reorders by name lookup.
- **WebSocket `/stream` dropped on malformed payloads**: now returns an
  `{"error": ..., "detail": ...}` envelope and keeps the socket open.
- **Stale CWRU references** in `evaluate.py` (figure suptitle visible to
  users), `features.py`, `explain.py`, `autoencoder.py`, and
  `tests/test_pipeline.py`.
- **`make download` target broken**: README referenced it but the Makefile
  defined `data:` only. Renamed `data` → `download` to match docs.
- Bare `except Exception: pass` blocks in `dashboard.py` now narrow to
  `(json.JSONDecodeError, ValueError, OSError)` and log via `logging`.

### Removed
- Dead `build_feature_matrix` shim in `dataset.py` (raised
  `NotImplementedError`, no callers).
- Unverifiable `data.nasa.gov/...` URL in `ingest.py` and `docs/DATASET.md`.
- `PLANO.md` — described pre-IMS-migration sprints; git log is canonical.

### Changed
- `Dockerfile` adds `IAD_RESULTS_DIR` and switches to a slim runtime stage.
- CORS middleware on the API allows any origin for the demo (tighten before
  any deploy with auth).
