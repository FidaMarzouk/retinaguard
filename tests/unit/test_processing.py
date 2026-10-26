import zipfile
from pathlib import Path

import pandas as pd
import pandera.errors
import pytest
from fundus_fixtures import synthetic_fundus

from retinaguard.data.preprocess import PREPROCESS_VERSION, decode_rgb, encode_jpeg
from retinaguard.data.processing import (
    attach_records,
    build_shard,
    member_name,
    process_bytes,
    processed_path,
    shard_ids,
    split_processed_path,
)
from retinaguard.data.sources import eyepacs

GOOD = encode_jpeg(synthetic_fundus())
BLANK = encode_jpeg(synthetic_fundus(radius=5))  # no plausible fundus


def test_process_bytes_success() -> None:
    record, jpeg = process_bytes("eyepacs:1_left", GOOD)
    assert jpeg is not None
    assert decode_rgb(jpeg).shape == (512, 512, 3)
    assert (record.width, record.height) == (800, 600)
    assert record.phash is not None
    assert len(record.phash) == 16
    assert record.fundus_radius == pytest.approx(250, abs=3)
    assert record.error is None
    assert record.preprocess_version == PREPROCESS_VERSION


@pytest.mark.parametrize(
    ("raw", "error_start"),
    [(b"not a jpeg", "decode"), (BLANK, "FundusNotFoundError")],
    ids=["garbage", "no-fundus"],
)
def test_process_bytes_records_failures(raw: bytes, error_start: str) -> None:
    record, jpeg = process_bytes("eyepacs:1_left", raw)
    assert jpeg is None
    assert record.error is not None
    assert record.error.startswith(error_start)
    assert record.phash is None
    assert len(record.sha256) == 64


def test_shard_ids_sorted_and_chunked() -> None:
    assert shard_ids(["c", "a", "e", "b", "d"], shard_size=2) == [["a", "b"], ["c", "d"], ["e"]]


def test_processed_path_round_trip() -> None:
    path = processed_path("eyepacs_000.zip", member_name("eyepacs:10_left"))
    assert path == "eyepacs_000.zip::10_left.jpg"
    assert split_processed_path(path) == ("eyepacs_000.zip", "10_left.jpg")
    with pytest.raises(ValueError, match="sharded"):
        split_processed_path("eyepacs_000.zip")


def test_build_shard_and_resume(tmp_path: Path) -> None:
    raw = {"eyepacs:1_left": GOOD, "eyepacs:1_right": BLANK, "eyepacs:2_left": GOOD}
    reads: list[str] = []

    def read(image_id: str) -> bytes:
        reads.append(image_id)
        return raw[image_id]

    records = build_shard(tmp_path, "eyepacs_000.zip", sorted(raw), read)

    assert records["image_id"].tolist() == sorted(raw)
    assert records["error"].notna().tolist() == [False, True, False]
    with zipfile.ZipFile(tmp_path / "eyepacs_000.zip") as archive:
        assert sorted(archive.namelist()) == ["1_left.jpg", "2_left.jpg"]
    assert records.loc[0, "processed_path"] == "eyepacs_000.zip::1_left.jpg"
    assert not list(tmp_path.glob("*.tmp"))

    # A rerun finds the finished shard and does not read any image again.
    reads.clear()
    again = build_shard(tmp_path, "eyepacs_000.zip", sorted(raw), read)
    assert reads == []
    pd.testing.assert_frame_equal(again, records)


def test_build_shard_redoes_partial_shard(tmp_path: Path) -> None:
    (tmp_path / "eyepacs_000.zip").write_bytes(b"left over from a killed session")
    records = build_shard(tmp_path, "eyepacs_000.zip", ["eyepacs:1_left"], lambda _: GOOD)
    assert records["error"].isna().all()
    with zipfile.ZipFile(tmp_path / "eyepacs_000.zip") as archive:
        assert archive.namelist() == ["1_left.jpg"]


def _labels() -> pd.DataFrame:
    labels = pd.DataFrame({"image": ["1_left", "1_right"], "level": ["0", "3"]}, dtype=str)
    return eyepacs.rows_from_labels(labels)


def test_attach_records(tmp_path: Path) -> None:
    raw = {"eyepacs:1_left": GOOD, "eyepacs:1_right": BLANK}
    records = build_shard(tmp_path, "eyepacs_000.zip", sorted(raw), raw.__getitem__)

    manifest = attach_records(_labels(), records)

    assert manifest.loc[0, "processed_path"] == "eyepacs_000.zip::1_left.jpg"
    assert manifest["processed_path"].isna().tolist() == [False, True]
    assert manifest["phash"].isna().tolist() == [False, True]
    assert manifest["width"].tolist() == [800, 800]


def test_attach_records_requires_every_label(tmp_path: Path) -> None:
    records = build_shard(tmp_path, "s.zip", ["eyepacs:1_left"], lambda _: GOOD)
    with pytest.raises(ValueError, match="no record"):
        attach_records(_labels(), records)


def test_manifest_rejects_path_without_hash(tmp_path: Path) -> None:
    raw = {"eyepacs:1_left": GOOD, "eyepacs:1_right": GOOD}
    records = build_shard(tmp_path, "s.zip", sorted(raw), raw.__getitem__)
    records.loc[0, "phash"] = None
    with pytest.raises(pandera.errors.SchemaError, match="processed_iff_hashed"):
        attach_records(_labels(), records)
