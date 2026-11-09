"""Frozen-backbone embeddings of every processed image, for linear probes and drift monitoring.

    python -m retinaguard.train.extract_embeddings --model dinov2_vitb \\
        --manifest manifest.parquet --shard-dirs DIR [DIR ...] --out-dir embeddings/dinov2_vitb

Uses the pretrained weights of a model config (``configs/model/``) without any task training.
Images are fed as in grader training (512 px, ImageNet scaling), by default without local mean
normalisation, which a backbone pretrained on natural images has never seen. Writes
``embeddings.npy`` (float16, one row per image), ``index.parquet`` (``image_id`` in the same
order) and ``meta.json``. All splits are included, train too, since a probe is fitted on it.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from retinaguard import __version__
from retinaguard.config import load_config, timm_kwargs
from retinaguard.data.datasets import FundusDataset
from retinaguard.models.grader import Grader
from retinaguard.train.train_grader import resolve_device


def embedding_rows(manifest: pd.DataFrame) -> pd.DataFrame:
    """Every processed image, in a stable order."""
    rows = manifest[manifest["processed_path"].notna()]
    return rows.sort_values("image_id").reset_index(drop=True)


@torch.no_grad()
def extract(
    backbone: nn.Module,
    rows: pd.DataFrame,
    shard_dirs: Sequence[Path],
    *,
    image_size: int,
    local_mean_norm: bool,
    batch_size: int = 64,
    num_workers: int = 4,
    device: torch.device | None = None,
) -> npt.NDArray[np.float16]:
    """Pooled backbone features of ``rows``, in row order."""
    device = device or next(backbone.parameters()).device
    backbone = backbone.to(device).eval()
    dataset = FundusDataset(rows, shard_dirs, size=image_size, local_mean_norm=local_mean_norm)
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=device.type == "cuda"
    )
    chunks = []
    for done, (images, _, _) in enumerate(loader, start=1):
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            features = backbone(images.to(device))
        chunks.append(features.float().cpu().numpy().astype(np.float16))
        if done % 100 == 0:
            print(f"{done * batch_size:,}/{len(dataset):,} images", flush=True)
    return np.concatenate(chunks)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="model config name, e.g. dinov2_vitb")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--local-mean-norm", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    model_cfg = load_config("train/grader", [f"model={args.model}"]).model
    backbone = Grader(
        model_cfg.timm_name, pretrained=True, timm_kwargs=timm_kwargs(model_cfg)
    ).backbone
    rows = embedding_rows(pd.read_parquet(args.manifest))
    embeddings = extract(
        backbone,
        rows,
        args.shard_dirs,
        image_size=args.image_size,
        local_mean_norm=args.local_mean_norm,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=resolve_device(args.device),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "embeddings.npy", embeddings)
    rows[["image_id"]].to_parquet(args.out_dir / "index.parquet", index=False)
    meta = {
        "model": args.model,
        "timm_name": model_cfg.timm_name,
        "image_size": args.image_size,
        "local_mean_norm": args.local_mean_norm,
        "retinaguard_version": __version__,
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{len(rows):,} embeddings of size {embeddings.shape[1]} written to {args.out_dir}")


if __name__ == "__main__":
    main()
