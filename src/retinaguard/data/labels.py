"""Label harmonisation to ICDR 0-4 plus binary ``referable``."""

LABEL_MAP_VERSION = "v1"
ICDR_GRADES = (0, 1, 2, 3, 4)
REFERABLE_MIN_GRADE = 2


def parse_icdr(raw: object) -> int:
    """Parse an ICDR grade that a source already reports on the 0-4 scale."""
    try:
        grade = int(str(raw).strip())
    except ValueError:
        raise ValueError(f"not an ICDR grade: {raw!r}") from None
    if grade not in ICDR_GRADES:
        raise ValueError(f"ICDR grade out of range 0-4: {raw!r}")
    return grade


def is_referable(grade: int) -> bool:
    return grade >= REFERABLE_MIN_GRADE
