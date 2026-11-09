import base64
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from retinaguard.serving.api import create_app
from retinaguard.serving.store import PredictionStore, metadata

pytestmark = pytest.mark.integration

ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "idrid"  # IDRiD, CC BY 4.0
DECISIONS = {"REFER", "NO_REFER", "DEFER_TO_HUMAN", "RETAKE"}


@pytest.fixture
def client(tmp_path: Path, models_dir: Path) -> Iterator[TestClient]:
    url = f"sqlite:///{(tmp_path / 'db.sqlite').as_posix()}"
    with TestClient(create_app(models_dir, url, git_sha="abc123")) as test_client:
        yield test_client


def _image(name: str = "IDRiD_016.jpg") -> dict[str, tuple[str, bytes, str]]:
    return {"image": (name, (FIXTURES / name).read_bytes(), "image/jpeg")}


def test_health_and_models(client: TestClient) -> None:
    assert client.get("/health").json()["models_loaded"] is True
    info = client.get("/models").json()
    assert set(info["versions"]) == {"quality", "grader", "segmentation"}
    assert info["lesions_available"] is True
    assert info["thresholds"]["lambda_low"] == 0.05


def test_predict_is_stored_and_retrievable(client: TestClient) -> None:
    response = client.post("/predict", files=_image(), data={"site": "idrid", "lesions": "true"})
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] in DECISIONS
    assert "not a medical device" in body["disclaimer"]
    record = client.get(f"/predictions/{body['prediction_id']}").json()
    assert record["site"] == "idrid"
    assert record["git_sha"] == "abc123"
    assert record["decision"] == body["decision"]
    assert len(record["input_sha256"]) == 64
    if body["decision"] != "RETAKE":
        assert base64.b64decode(body["overlay_png_base64"])[:4] == b"\x89PNG"
        assert record["p_referable_calibrated"] == pytest.approx(body["p_referable"])


def test_batch_and_listing(client: TestClient) -> None:
    files = [
        ("images", (name, (FIXTURES / name).read_bytes(), "image/jpeg"))
        for name in ("IDRiD_016.jpg", "IDRiD_138.jpg", "IDRiD_139.jpg")
    ]
    response = client.post("/predict/batch", files=files)
    assert response.status_code == 200
    assert len(response.json()) == 3
    listed = client.get("/predictions", params={"limit": 10}).json()
    assert len(listed) == 3
    decision = listed[0]["decision"]
    filtered = client.get("/predictions", params={"decision": decision}).json()
    assert {row["decision"] for row in filtered} == {decision}
    assert client.get("/predictions", params={"decision": "MAYBE"}).status_code == 422


def test_bad_uploads(client: TestClient) -> None:
    garbage = client.post("/predict", files={"image": ("x.jpg", b"not an image", "image/jpeg")})
    assert garbage.status_code == 200
    assert (garbage.json()["decision"], garbage.json()["reason"]) == ("DEFER_TO_HUMAN", "error")
    record = client.get(f"/predictions/{garbage.json()['prediction_id']}").json()
    assert "could not decode" in record["error"]
    empty = client.post("/predict", files={"image": ("x.jpg", b"", "image/jpeg")})
    assert empty.status_code == 400
    assert client.get("/predictions/no-such-id").status_code == 404


def test_audit_log_records_predictions(tmp_path: Path, models_dir: Path) -> None:
    url = f"sqlite:///{(tmp_path / 'db.sqlite').as_posix()}"
    with TestClient(create_app(models_dir, url)) as test_client:
        prediction_id = test_client.post("/predict", files=_image()).json()["prediction_id"]
    events = PredictionStore(url).events(prediction_id)
    assert [e["event"] for e in events] == ["prediction"]


def test_migrations_match_the_tables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    url = f"sqlite:///{(tmp_path / 'migrated.sqlite').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", url)
    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
    with create_engine(url).connect() as connection:
        differences = compare_metadata(MigrationContext.configure(connection), metadata)
    assert differences == []
    command.downgrade(Config(str(ROOT / "alembic.ini")), "base")
