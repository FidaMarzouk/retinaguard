import pytest

pytestmark = pytest.mark.regression


@pytest.mark.skip(reason="Model regression gate added in Phase 6 (PLAN.md §12).")
def test_model_auroc_gate() -> None:
    raise AssertionError
