"""Static tests for Dockerfile and docker-compose.yml structure."""

from __future__ import annotations

from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).parent.parent


def test_dockerfile_structure() -> None:
    """Dockerfile contains the mandatory directives for a secure, functional image."""
    dockerfile = (_REPO_ROOT / "Dockerfile").read_text()

    assert "FROM python:3.12-slim" in dockerfile, "Base image must be python:3.12-slim"
    assert "USER appuser" in dockerfile, "Must run as non-root user 'appuser'"
    assert "EXPOSE 8501" in dockerfile, "Must expose Streamlit default port 8501"
    assert "HEALTHCHECK" in dockerfile, "Must define a HEALTHCHECK"
    assert "localhost:8501/_stcore/health" in dockerfile, (
        "Healthcheck must target Streamlit health endpoint"
    )
    assert "src/dashboard.py" in dockerfile, "CMD must reference src/dashboard.py"
    assert "streamlit" in dockerfile, "CMD must use streamlit to launch the dashboard"


def test_compose_yaml_parses() -> None:
    """docker-compose.yml is valid YAML with the expected dashboard service configuration."""
    compose_path = _REPO_ROOT / "docker-compose.yml"
    compose = yaml.safe_load(compose_path.read_text())

    assert "services" in compose, "Must have a 'services' key"
    assert "dashboard" in compose["services"], "Must define a 'dashboard' service"

    dashboard = compose["services"]["dashboard"]
    ports = dashboard.get("ports", [])
    port_strings = [str(p) for p in ports]
    assert any("8501" in p for p in port_strings), (
        f"dashboard service must map port 8501, got: {ports}"
    )

    volumes = dashboard.get("volumes", [])
    assert len(volumes) >= 1, "dashboard service should mount at least one volume"


def test_dockerignore_excludes_sensitive_paths() -> None:
    """.dockerignore excludes .venv, .git, .env and __pycache__."""
    dockerignore = (_REPO_ROOT / ".dockerignore").read_text()
    for entry in (".venv", ".git", ".env", "__pycache__"):
        assert entry in dockerignore, f".dockerignore must exclude '{entry}'"
