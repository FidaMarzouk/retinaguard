import pytest

from retinaguard.pipeline.definitions import defs

pytestmark = pytest.mark.integration


def test_code_location_loads() -> None:
    # Resolving the repository fails if asset/check definitions are inconsistent.
    defs.get_repository_def()
