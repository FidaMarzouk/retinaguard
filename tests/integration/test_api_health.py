import pytest
from fastapi.testclient import TestClient

from retinaguard import __version__
from retinaguard.serving.api import app

pytestmark = pytest.mark.integration


def test_health() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__, "models_loaded": False}
