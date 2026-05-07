# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
