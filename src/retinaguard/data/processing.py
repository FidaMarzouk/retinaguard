"""Batch preprocessing of raw images into sharded zip archives plus per-image records.

Processed images are stored in zip shards of a few thousand JPEGs rather than as individual
files: Kaggle notebook outputs and datasets handle a few dozen large files far better than tens
of thousands of small ones. A shard is written to a temporary name and renamed when complete,
so a rerun after an interrupted session skips finished shards and redoes only the partial one.

``processed_path`` in the manifest is ``"<shard>.zip::<member>"``, relative to the processed
data root.
"""

import os
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from retinaguard.data.hashing import phash, sha256_hex
from retinaguard.data.preprocess import (
    DEFAULT_SIZE,
    PREPROCESS_VERSION,
    Image,
    decode_rgb,
    encode_jpeg,
    preprocess,
)
from retinaguard.data.schemas import ImageManifestSchema

SHARD_SIZE = 2000
PATH_SEPARATOR = "::"


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    sha256: str
    width: int | None = None
    height: int | None = None
    phash: str | None = None
    fundus_cx: float | None = None
    fundus_cy: float | None = None
    fundus_radius: float | None = None
    processed_path: str | None = None
    preprocess_version: str = PREPROCESS_VERSION
    error: str | None = None


def process_bytes(
    image_id: str, raw: bytes, size: int = DEFAULT_SIZE
) -> tuple[ImageRecord, bytes | None]:
    """Preprocess one raw image. Never raises: failures are returned in ``ImageRecord.error``."""
    sha = sha256_hex(raw)
    try:
        img = decode_rgb(raw)
    except ValueError as exc:
        return ImageRecord(image_id=image_id, sha256=sha, error=f"decode: {exc}"), None
    height, width = img.shape[:2]
    try:
        out, circle = preprocess(img, size)
    except ValueError as exc:  # includes FundusNotFoundError
        record = ImageRecord(image_id, sha, width, height, error=f"{type(exc).__name__}: {exc}")
        return record, None
    record = ImageRecord(
        image_id=image_id,
        sha256=sha,
        width=width,
        height=height,
        phash=phash(out),
        fundus_cx=circle.cx,
        fundus_cy=circle.cy,
        fundus_radius=circle.radius,
    )
    return record, encode_jpeg(out)


def shard_ids(image_ids: Iterable[str], shard_size: int = SHARD_SIZE) -> list[list[str]]:
    """Deterministic shards: sorted ids cut into consecutive chunks."""
    ordered = sorted(image_ids)
    return [ordered[i : i + shard_size] for i in range(0, len(ordered), shard_size)]


def member_name(image_id: str) -> str:
    return image_id.split(":", 1)[1] + ".jpg"


def processed_path(shard_file: str, member: str) -> str:
    return f"{shard_file}{PATH_SEPARATOR}{member}"


def split_processed_path(path: str) -> tuple[str, str]:
    shard_file, sep, member = path.partition(PATH_SEPARATOR)
    if not sep or not member:
        raise ValueError(f"not a sharded processed path: {path!r}")
    return shard_file, member


def attach_records(labels: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    """Join preprocessing records onto label rows and validate the result.

    Every labelled image must have exactly one record; a missing one means the run is incomplete.
    """
    missing = sorted(set(labels["image_id"]) - set(records["image_id"]))
    if missing:
        raise ValueError(f"{len(missing)} labelled images have no record, e.g. {missing[:3]}")
    columns = ["width", "height", "sha256", "phash", "processed_path", "preprocess_version"]
    merged = labels.merge(
        records[["image_id", *columns]], on="image_id", how="left", validate="one_to_one"
    )
    merged = merged.astype({"width": "Int64", "height": "Int64"})
    return ImageManifestSchema.validate(merged)


def build_shard(
    shard_dir: Path,
    shard_file: str,
    image_ids: Sequence[str],
    read_raw: Callable[[str], bytes],
    size: int = DEFAULT_SIZE,
) -> pd.DataFrame:
    """Process ``image_ids`` into ``shard_dir/shard_file`` and return their records.

    Records are also saved next to the shard as ``<shard>.records.parquet``. If both files already
    exist the shard is complete and its saved records are returned without reprocessing.
    """
    shard_path = shard_dir / shard_file
    records_path = shard_dir / f"{shard_file}.records.parquet"
    if shard_path.exists() and records_path.exists():
        return pd.read_parquet(records_path)

    shard_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = shard_path.with_suffix(".tmp")
    records = []
    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for image_id in image_ids:
            record, jpeg = process_bytes(image_id, read_raw(image_id), size)
            if jpeg is not None:
                member = member_name(image_id)
                archive.writestr(member, jpeg)
                record = ImageRecord(
                    **{**asdict(record), "processed_path": processed_path(shard_file, member)}
                )
            records.append(asdict(record))
    frame = pd.DataFrame(records, columns=list(ImageRecord.__dataclass_fields__))
    frame.to_parquet(records_path.with_suffix(".tmp"), index=False)
    os.replace(tmp_path, shard_path)
    os.replace(records_path.with_suffix(".tmp"), records_path)
    return frame


class ProcessedImages:
    """Read processed images by ``image_id`` from shard folders, keeping each shard open.

    ``processed_paths`` maps image ids to ``processed_path``; ``roots`` maps each shard file name
    to the folder holding it, so shards of several datasets can be read together.
    """

    def __init__(self, processed_paths: Mapping[str, str], roots: Mapping[str, Path]) -> None:
        self._paths = processed_paths
        self._roots = roots
        self._open: dict[str, zipfile.ZipFile] = {}

    def read(self, image_id: str) -> Image:
        shard_file, member = split_processed_path(self._paths[image_id])
        if shard_file not in self._open:
            self._open[shard_file] = zipfile.ZipFile(self._roots[shard_file] / shard_file)
        return decode_rgb(self._open[shard_file].read(member))

    def close(self) -> None:
        for archive in self._open.values():
            archive.close()
        self._open.clear()


def shard_roots(*folders: Path) -> dict[str, Path]:
    """Shard file name → folder, for every ``*.zip`` in ``folders``."""
    return {p.name: folder for folder in folders for p in sorted(folder.glob("*.zip"))}
