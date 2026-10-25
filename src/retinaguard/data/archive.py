"""Read split zip archives (``train.zip.001``, ``train.zip.002``, ...) without extracting them.

Kaggle ships the EyePACS images as one ~35 GB zip cut into byte-range parts, more than a Kaggle
session can write to disk. ``ConcatenatedParts`` presents the parts as one seekable file, so
``zipfile`` can read members straight from the read-only input mount.
"""

import io
import re
import zipfile
from collections.abc import Iterator, Sequence
from pathlib import Path

_PART_SUFFIX = re.compile(r"\.(\d{3})$")


class ConcatenatedParts(io.RawIOBase):
    """Read-only, seekable view of several files laid end to end."""

    def __init__(self, paths: Sequence[Path]) -> None:
        if not paths:
            raise ValueError("no archive parts given")
        self._files = [path.open("rb") for path in paths]
        self._starts: list[int] = []
        total = 0
        for path in paths:
            self._starts.append(total)
            total += path.stat().st_size
        self._size = total
        self._pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._size}[whence]
        self._pos = max(0, base + offset)
        return self._pos

    def readinto(self, buffer: memoryview) -> int:  # type: ignore[override]
        view = memoryview(buffer).cast("B")
        written = 0
        while written < len(view) and self._pos < self._size:
            part = self._part_at(self._pos)
            fh = self._files[part]
            fh.seek(self._pos - self._starts[part])
            part_end = self._starts[part + 1] if part + 1 < len(self._starts) else self._size
            want = min(len(view) - written, part_end - self._pos)
            chunk = fh.read(want)
            if not chunk:
                break
            view[written : written + len(chunk)] = chunk
            written += len(chunk)
            self._pos += len(chunk)
        return written

    def close(self) -> None:
        for fh in self._files:
            fh.close()
        super().close()

    def _part_at(self, pos: int) -> int:
        part = 0
        while part + 1 < len(self._starts) and self._starts[part + 1] <= pos:
            part += 1
        return part


def find_parts(first_part: Path) -> list[Path]:
    """All numbered siblings of ``name.zip.001``, in order; checks none are missing."""
    match = _PART_SUFFIX.search(first_part.name)
    if match is None:
        raise ValueError(f"not a numbered archive part: {first_part.name}")
    stem = first_part.name[: match.start()]
    parts = sorted(first_part.parent.glob(f"{stem}.[0-9][0-9][0-9]"))
    numbers = [int(p.suffix[1:]) for p in parts]
    if numbers != list(range(1, len(parts) + 1)):
        raise ValueError(f"archive parts not contiguous from 001: {[p.name for p in parts]}")
    return parts


def open_split_zip(first_part: Path) -> zipfile.ZipFile:
    raw = ConcatenatedParts(find_parts(first_part))
    return zipfile.ZipFile(io.BufferedReader(raw, buffer_size=1 << 20))


def iter_members(archive: zipfile.ZipFile, suffix: str) -> Iterator[zipfile.ZipInfo]:
    for info in archive.infolist():
        if not info.is_dir() and info.filename.lower().endswith(suffix):
            yield info
