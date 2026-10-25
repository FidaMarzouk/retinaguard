import io
import zipfile
from pathlib import Path

import pytest

from retinaguard.data.archive import ConcatenatedParts, find_parts, iter_members, open_split_zip


def _split(data: bytes, directory: Path, stem: str, part_size: int) -> list[Path]:
    paths = []
    for i, start in enumerate(range(0, len(data), part_size), start=1):
        path = directory / f"{stem}.{i:03d}"
        path.write_bytes(data[start : start + part_size])
        paths.append(path)
    return paths


def test_concatenated_reads_across_part_boundaries(tmp_path: Path) -> None:
    data = bytes(range(256)) * 10
    parts = _split(data, tmp_path, "blob.bin", 333)

    f = ConcatenatedParts(parts)
    assert f.read() == data
    f.seek(330)
    assert f.read(10) == data[330:340]  # spans parts 1 and 2
    f.seek(-5, io.SEEK_END)
    assert f.read() == data[-5:]
    f.seek(2, io.SEEK_CUR)
    assert f.read() == b""


def test_split_zip_round_trip(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    members = {f"train/{i}_left.jpeg": bytes([i]) * (500 + i) for i in range(20)}
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)
        zf.writestr("train/readme.txt", b"not an image")
    _split(buffer.getvalue(), tmp_path, "train.zip", 1024)

    with open_split_zip(tmp_path / "train.zip.001") as archive:
        names = [info.filename for info in iter_members(archive, ".jpeg")]
        assert names == list(members)
        assert archive.read("train/7_left.jpeg") == members["train/7_left.jpeg"]


def test_find_parts_rejects_gaps(tmp_path: Path) -> None:
    for i in (1, 2, 4):
        (tmp_path / f"train.zip.{i:03d}").write_bytes(b"x")
    with pytest.raises(ValueError, match="not contiguous"):
        find_parts(tmp_path / "train.zip.001")


def test_find_parts_ignores_other_archives(tmp_path: Path) -> None:
    for name in ("train.zip.001", "train.zip.002", "test.zip.001"):
        (tmp_path / name).write_bytes(b"x")
    assert [p.name for p in find_parts(tmp_path / "train.zip.001")] == [
        "train.zip.001",
        "train.zip.002",
    ]


def test_find_parts_requires_numbered_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="numbered"):
        find_parts(tmp_path / "train.zip")
