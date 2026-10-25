import retinaguard


def test_version_is_set() -> None:
    assert retinaguard.__version__ != ""


def test_disclaimer_says_not_a_medical_device() -> None:
    assert "not a medical device" in retinaguard.DISCLAIMER
