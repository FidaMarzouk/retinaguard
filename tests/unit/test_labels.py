import pytest

from retinaguard.data.labels import is_referable, parse_icdr


@pytest.mark.parametrize(("raw", "grade"), [("0", 0), (" 4 ", 4), (2, 2)])
def test_parse_icdr(raw: object, grade: int) -> None:
    assert parse_icdr(raw) == grade


@pytest.mark.parametrize("raw", ["5", "-1", "moderate", "", "2.5"])
def test_parse_icdr_rejects_invalid(raw: str) -> None:
    with pytest.raises(ValueError, match="ICDR grade"):
        parse_icdr(raw)


def test_referable_threshold() -> None:
    assert [is_referable(g) for g in range(5)] == [False, False, True, True, True]
