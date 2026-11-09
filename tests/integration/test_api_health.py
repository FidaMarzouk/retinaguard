from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from retinaguard import __version__
from retinaguard.serving.api import create_app

pytestmark = pytest.mark.integration


def test_health_without_models(tmp_path: Path) -> None:
    app = create_app(tmp_path / "no-models", f"sqlite:///{(tmp_path / 'db.sqlite').as_posix()}")
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "version": __version__,
            "models_loaded": False,
            "database": True,
        }
        # no models: predictions are refused, never guessed
        image = {"image": ("x.jpg", b"\xff\xd8", "image/jpeg")}
        assert client.post("/predict", files=image).status_code == 503
        assert client.get("/models").status_code == 503
