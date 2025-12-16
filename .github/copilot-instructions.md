<!-- Copied/created by GitHub Copilot assistant -->
# Copilot / AI Agent Instructions — End-to-End MLOps Insurance Pricing

Purpose: give AI coding agents the essential, actionable knowledge to be immediately productive in this repository.

- **Big picture**: This is an MLOps pipeline skeleton for insurance pricing. Key artifacts and responsibilities are separated by folder:
  - `data/` — raw and processed datasets. Example: `data/raw/uk_insurance_synthetic.csv` is the canonical sample input.
  - `src/` — code split by concern: `training/`, `deployment/`, `monitoring/`, `governance/`. These are currently scaffolds; new code should be placed in the appropriate subpackage.
  - `mlruns/` — MLflow experiment backend store (experiment runs and artifacts). Treat this as the local MLflow tracking DB.
  - `models/` — serialized models and model artifacts produced by training.
  - `tests/` — test cases (currently empty); add unit and integration tests here alongside production modules.

- **Why this structure**: separation of concerns for MLOps lifecycle — training, model storage, deployment, monitoring, and governance live in separate directories to keep experiment code, serving code, and observability/governance rules isolated.

- **Developer workflows & commands (discoverable & tested in this repo)**:
  - Activate an existing venv (if present): `source venv/bin/activate` or inspect `venv_broken_1764812438/bin/python3.11` if you must reproduce that environment.
  - If no venv exists, create one and install dependencies (there is no `requirements.txt` in the repo):
    ```bash
    python -m venv venv
    source venv/bin/activate
    # then install packages used by the project (add a requirements.txt when you capture them)
    ```
  - View MLflow experiments stored in this repo: `mlflow ui --backend-store-uri mlruns/` and open `http://127.0.0.1:5000`.
  - Inspect local model artifacts: look under `models/` and `mlruns/` for model files, metrics, and params.

- **Project-specific conventions & patterns** (do not assume defaults from other projects):
  - Directory-first layout: put runnable training code under `src/training/` and serving/API code under `src/deployment/` (not top-level scripts).
  - `mlruns/` is used as the canonical, checked-in store for experiments in this repository — prefer relative paths to it when running MLflow locally.
  - There is no `requirements.txt` or `pyproject.toml` in the repo root. Capture dependencies into `requirements.txt` before sharing reproducible runs.

- **Integration points & cross-component communication**:
  - MLflow (local `mlruns/`) — training writes experiments here; deployment and monitoring should read model artifact URIs from `mlruns` or `models/`.
  - Data ingress is file-based: `data/raw/` → `data/processed/` pipelines. Expect CSV inputs (see `uk_insurance_synthetic.csv`).
  - CI: `.github/workflows/` exists but is empty; assume PR-based CI is intended — add workflow YAML when enabling automated tests/builds.

- **Where to look first when changing behaviour**:
  - Add training code: `src/training/` (create `train.py` or a package submodule). Example pattern: write a `train(...)` callable that logs to MLflow.
  - Add serving API: `src/deployment/` (FastAPI / Uvicorn are already present in the virtual env packages; prefer an ASGI app file `src/deployment/app.py`).
  - Add monitoring checks / dashboards: `src/monitoring/` and use `mlruns/` metrics or `models/` artifacts.

- **Concrete examples for common tasks**:
  - Start MLflow UI: `mlflow ui --backend-store-uri mlruns/`
  - Inspect installed packages from provided venv: `venv_broken_1764812438/bin/python -m pip freeze | sed -n '1,120p'`
  - Run tests (after adding them): `pytest -q tests/`

If anything in this file is unclear or you'd like more detail (for example: expected training script signatures, preferred logging/MLflow conventions, or CI rules), tell me which area to expand and I will iterate.

Files referenced: `data/raw/uk_insurance_synthetic.csv`, `mlruns/`, `models/`, `src/training/`, `src/deployment/`, `src/monitoring/`, `.github/workflows/`.
