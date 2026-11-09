import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.integration


def _review() -> None:
    from retinaguard.app.main import review_page

    review_page()


def _sites() -> None:
    from retinaguard.app.main import sites_page

    sites_page()


def _health() -> None:
    from retinaguard.app.main import health_page

    health_page()


def _card() -> None:
    from retinaguard.app.main import model_card_page

    model_card_page()


def _predict() -> None:
    from retinaguard.app.main import predict_page

    predict_page()


@pytest.mark.parametrize("page", [_predict, _review, _sites, _health, _card])
def test_pages_render_without_the_api(page, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")  # nothing listens there
    app = AppTest.from_function(page, default_timeout=60).run()
    assert not app.exception
    if page in (_predict, _review, _health):
        assert any("did not answer" in e.value for e in app.error)
