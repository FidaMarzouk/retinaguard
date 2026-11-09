"""Lesion segmentation data: IDRiD and DDR images with pixel masks, cropped and packed into a shard.

Lesion types (channel order): microaneurysms (MA), haemorrhages (HE), hard exudates (EX), soft
exudates (SE). Each image is cropped to its fundus disc like the grading images, then resized
to ``SEG_SIZE`` (higher than the 512 px grading size, so that 512 px training crops keep small
lesions visible). Its four masks get the same crop and are packed into one PNG, bit ``i`` for
lesion ``i``.

A shard is a zip with ``images/<id>.jpg`` and ``masks/<id>.png``, plus ``seg_index.parquet``
(``image_id``, ``site``, ``split``, lesion pixel counts). Splits: IDRiD keeps its official test
set and holds every fifth training image out for validation; DDR keeps train / val / test.

Sources (raw layout as released):

- IDRiD ``A. Segmentation`` (CC BY 4.0): ``1. Original Images/{a. Training Set,b. Testing Set}``
  and ``2. All Segmentation Groundtruths/<set>/{1. Microaneurysms,...}/IDRiD_01_MA.tif``; a
  missing mask file means no lesion of that type.
- DDR ``lesion_segmentation`` (CC BY-NC-SA 4.0): ``images/{train,val,test}/<name>.jpg`` and
  ``annotations/{train,val,tet}/{MA,HE,EX,SE}/<name>.tif`` (the release spells test ``tet``).
"""

import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd

from retinaguard.data.preprocess import (
    Image,
    crop_to_fundus,
    decode_rgb,
    encode_jpeg,
    find_fundus,
    fundus_mask,
    surround_colour,
)

LESIONS = ("MA", "HE", "EX", "SE")
LESION_NAMES = {
    "MA": "microaneurysms",
    "HE": "haemorrhages",
    "EX": "hard exudates",
    "SE": "soft exudates",
}
SEG_SIZE = 1024
# A downsampled mask pixel is a lesion when at least this share of its area was. Below one
# half, so microaneurysms a few raw pixels wide survive the resize.
MASK_COVERAGE = 0.25
INDEX_NAME = "seg_index.parquet"
IDRID_FOLDERS = {
    "MA": "1. Microaneurysms",
    "HE": "2. Haemorrhages",
    "EX": "3. Hard Exudates",
    "SE": "4. Soft Exudates",
}
IDRID_SETS = {"train": "a. Training Set", "test": "b. Testing Set"}
DDR_ANNOTATION_SPLITS = {"train": "train", "val": "val", "test": "tet"}
IDRID_VAL_EVERY = 5

Masks = npt.NDArray[np.bool_]  # (4, H, W)


@dataclass(frozen=True)
class RawPair:
    image_id: str
    site: str
    split: str
    image_path: Path
    mask_paths: dict[str, Path | None]  # None: no lesion of that type


def pack(masks: Masks) -> npt.NDArray[np.uint8]:
    """(4, H, W) booleans to one (H, W) uint8, bit i for lesion i."""
    packed = np.zeros(masks.shape[1:], dtype=np.uint8)
    for i, mask in enumerate(masks):
        packed |= mask.astype(np.uint8) << i
    return packed


def unpack(packed: npt.NDArray[np.uint8]) -> Masks:
    return np.stack([(packed >> i) & 1 for i in range(len(LESIONS))]).astype(bool)


def _crop_channel(
    mask: npt.NDArray[np.float32], circle_crop: tuple[int, int, int]
) -> npt.NDArray[np.float32]:
    side, left, top = circle_crop
    out = np.zeros((side, side), dtype=mask.dtype)
    x0, y0 = max(left, 0), max(top, 0)
    x1, y1 = min(left + side, mask.shape[1]), min(top + side, mask.shape[0])
    out[y0 - top : y1 - top, x0 - left : x1 - left] = mask[y0:y1, x0:x1]
    return out


def prepare_pair(img: Image, masks: Masks, size: int = SEG_SIZE) -> tuple[Image, Masks]:
    """Crop an image and its masks to the fundus and resize both to ``size``."""
    if masks.shape[1:] != img.shape[:2]:
        raise ValueError(f"mask shape {masks.shape[1:]} differs from image {img.shape[:2]}")
    circle = find_fundus(img)
    if surround_colour(img) is not None:
        img = np.where(fundus_mask(img)[..., None], img, 0).astype(np.uint8)
    crop = crop_to_fundus(img, circle)
    interpolation = cv2.INTER_AREA if crop.shape[0] >= size else cv2.INTER_CUBIC
    image = np.asarray(cv2.resize(crop, (size, size), interpolation=interpolation), np.uint8)
    box = (
        round(2 * circle.radius),
        round(circle.cx - circle.radius),
        round(circle.cy - circle.radius),
    )
    resized = []
    for mask in masks:
        cropped = _crop_channel(mask.astype(np.float32), box)
        area = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
        resized.append(area >= MASK_COVERAGE)
    return image, np.stack(resized)


def read_mask(path: Path | None, shape: tuple[int, int]) -> npt.NDArray[np.bool_]:
    if path is None:
        return np.zeros(shape, dtype=bool)
    mask = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"cannot read mask {path}")
    if mask.ndim == 3:
        mask = mask.max(axis=2)
    if mask.shape != shape:
        raise ValueError(f"{path}: mask shape {mask.shape} differs from image {shape}")
    return np.asarray(mask > 0)


def idrid_pairs(root: Path) -> list[RawPair]:
    """IDRiD segmentation images from ``A. Segmentation``."""
    pairs = []
    for split, folder in IDRID_SETS.items():
        images = sorted((root / "1. Original Images" / folder).glob("IDRiD_*.jpg"))
        if not images:
            raise FileNotFoundError(f"no IDRiD images in {root / '1. Original Images' / folder}")
        held_out = {p.stem for p in images[IDRID_VAL_EVERY - 1 :: IDRID_VAL_EVERY]}
        truth = root / "2. All Segmentation Groundtruths" / folder
        for path in images:
            masks: dict[str, Path | None] = {}
            for lesion, sub in IDRID_FOLDERS.items():
                candidate = truth / sub / f"{path.stem}_{lesion}.tif"
                masks[lesion] = candidate if candidate.exists() else None
            this_split = "val" if split == "train" and path.stem in held_out else split
            pairs.append(RawPair(f"idrid_seg:{path.stem}", "idrid", this_split, path, masks))
    return pairs


def ddr_pairs(root: Path) -> list[RawPair]:
    """DDR images from ``lesion_segmentation``; every image has all four mask files."""
    pairs = []
    for split, annotation_split in DDR_ANNOTATION_SPLITS.items():
        images = sorted((root / "images" / split).glob("*.jpg"))
        if not images:
            raise FileNotFoundError(f"no DDR images in {root / 'images' / split}")
        for path in images:
            masks: dict[str, Path | None] = {}
            for lesion in LESIONS:
                candidate = root / "annotations" / annotation_split / lesion / f"{path.stem}.tif"
                if not candidate.exists():
                    raise FileNotFoundError(f"missing DDR mask {candidate}")
                masks[lesion] = candidate
            pairs.append(RawPair(f"ddr_seg:{path.stem}", "ddr", split, path, masks))
    return pairs


def build_shard(pairs: Iterable[RawPair], out_zip: Path, size: int = SEG_SIZE) -> pd.DataFrame:
    """Write prepared images and packed masks to ``out_zip``; returns the index rows."""
    rows = []
    seen: set[str] = set()
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_STORED) as archive:
        for pair in pairs:
            if pair.image_id in seen:
                raise ValueError(f"repeated image id {pair.image_id}")
            seen.add(pair.image_id)
            img = decode_rgb(pair.image_path.read_bytes())
            shape = img.shape[:2]
            masks = np.stack([read_mask(pair.mask_paths[lesion], shape) for lesion in LESIONS])
            image, resized = prepare_pair(img, masks, size)
            name = pair.image_id.replace(":", "__")
            archive.writestr(f"images/{name}.jpg", encode_jpeg(image))
            ok, png = cv2.imencode(".png", pack(resized))
            if not ok:
                raise ValueError(f"cannot encode mask of {pair.image_id}")
            archive.writestr(f"masks/{name}.png", png.tobytes())
            rows.append(
                {
                    "image_id": pair.image_id,
                    "site": pair.site,
                    "split": pair.split,
                    "shard": out_zip.name,
                    **{
                        f"pixels_{lesion}": int(m.sum())
                        for lesion, m in zip(LESIONS, resized, strict=True)
                    },
                }
            )
    return pd.DataFrame(rows)


def read_index(shard_dirs: Iterable[Path]) -> pd.DataFrame:
    """All ``seg_index.parquet`` files found in ``shard_dirs``, with the folder of each shard."""
    frames = []
    for folder in shard_dirs:
        for path in sorted(Path(folder).glob(f"**/{INDEX_NAME}")):
            frames.append(pd.read_parquet(path).assign(folder=str(path.parent)))
    if not frames:
        raise FileNotFoundError(f"no {INDEX_NAME} in {list(shard_dirs)}")
    index = pd.concat(frames, ignore_index=True)
    if index["image_id"].duplicated().any():
        raise ValueError("segmentation image ids repeat across shards")
    return index


class SegShards:
    """Reads prepared images and masks of index rows; one open zip per shard file."""

    def __init__(self, index: pd.DataFrame) -> None:
        self._where = {
            str(row.image_id): Path(str(row.folder)) / str(row.shard)
            for row in index.itertuples(index=False)
        }
        self._open: dict[Path, zipfile.ZipFile] = {}

    def read(self, image_id: str) -> tuple[Image, Masks]:
        path = self._where[image_id]
        archive = self._open.get(path)
        if archive is None:
            archive = self._open[path] = zipfile.ZipFile(path)
        name = image_id.replace(":", "__")
        image = decode_rgb(archive.read(f"images/{name}.jpg"))
        packed = cv2.imdecode(np.frombuffer(archive.read(f"masks/{name}.png"), np.uint8), 0)
        return image, unpack(np.asarray(packed, dtype=np.uint8))

    def __iter__(self) -> Iterator[str]:
        return iter(self._where)


def write_index(index: pd.DataFrame, folder: Path) -> Path:
    path = folder / INDEX_NAME
    index.to_parquet(path, index=False)
    return path


def main(argv: list[str] | None = None) -> None:
    """Prepare one site: ``python -m retinaguard.data.segmentation idrid "A. Segmentation" out``."""
    import argparse

    parser = argparse.ArgumentParser(description="Prepare lesion segmentation images and masks")
    parser.add_argument("site", choices=["idrid", "ddr"])
    parser.add_argument(
        "root", type=Path, help="IDRiD 'A. Segmentation' or DDR 'lesion_segmentation'"
    )
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--size", type=int, default=SEG_SIZE)
    args = parser.parse_args(argv)

    pairs = idrid_pairs(args.root) if args.site == "idrid" else ddr_pairs(args.root)
    index = build_shard(pairs, args.out_dir / f"{args.site}_seg.zip", args.size)
    write_index(index, args.out_dir)
    print(index.groupby("split").size().to_string())
    print(index.filter(like="pixels_").gt(0).groupby(index["split"]).sum().to_string())


if __name__ == "__main__":
    main()
