"""Run a trained grader over every image evaluation and later calibration need.

    python -m retinaguard.train.predict --checkpoint runs/grader-convnext_tiny/best.pt \\
        --manifest manifest.parquet --shard-dirs DIR [DIR ...] --out-dir predictions/

Writes ``predictions.parquet`` (grade and referable probabilities per image) and
``embeddings.npy`` (pooled backbone features, float16, same row order), for the ``val``,
``calib``, ``test_internal`` and ``test_external`` splits. Quality rejects and ungradable images
are included; evaluation decides which rows to count.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from retinaguard.data.datasets import FundusDataset
from retinaguard.models.grader import N_GRADES, Grader
from retinaguard.train.train_grader import resolve_device

PREDICTED_SPLITS = ("val", "calib", "test_internal", "test_external")
PROBABILITY_COLUMNS = [f"p_grade_{g}" for g in range(N_GRADES)]


def load_grader(checkpoint: Path, device: torch.device) -> tuple[Grader, dict[str, Any]]:
    """Model and training config from a ``best.pt`` written by ``train_grader``."""
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    config: dict[str, Any] = state["config"]
    model = Grader(
        config["model"]["timm_name"],
        pretrained=False,
        timm_kwargs=config["model"].get("timm_kwargs"),
    )
    model.load_state_dict(state["model"])
    return model.to(device).eval(), config


def prediction_rows(manifest: pd.DataFrame) -> pd.DataFrame:
    """Processed images of the predicted splits, in a stable order."""
    rows = manifest[manifest["split"].isin(PREDICTED_SPLITS) & manifest["processed_path"].notna()]
    return rows.sort_values("image_id").reset_index(drop=True)


@torch.no_grad()
def predict_images(
    model: Grader,
    rows: pd.DataFrame,
    shard_dirs: Sequence[Path],
    *,
    image_size: int,
    local_mean_norm: bool,
    batch_size: int = 64,
    num_workers: int = 4,
    device: torch.device | None = None,
) -> tuple[pd.DataFrame, npt.NDArray[np.float16]]:
    """Per-image probabilities (``image_id``, ``p_grade_0..4``, ``p_referable``) and embeddings."""
    device = device or next(model.parameters()).device
    dataset = FundusDataset(rows, shard_dirs, size=image_size, local_mean_norm=local_mean_norm)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    amp = device.type == "cuda"
    grade_probs, referable_probs, embeddings = [], [], []
    for done, (images, _, _) in enumerate(loader, start=1):
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            output = model(images.to(device))
        grade_probs.append(output.grade_logits.float().softmax(1).cpu().numpy())
        referable_probs.append(output.referable_logit.float().sigmoid().cpu().numpy())
        embeddings.append(output.embedding.float().cpu().numpy().astype(np.float16))
        if done % 50 == 0:
            print(f"{done * batch_size:,}/{len(dataset):,} images", flush=True)
    predictions = pd.DataFrame(np.concatenate(grade_probs), columns=PROBABILITY_COLUMNS)
    predictions.insert(0, "image_id", dataset.image_ids)
    predictions["p_referable"] = np.concatenate(referable_probs)
    return predictions, np.concatenate(embeddings)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    model, config = load_grader(args.checkpoint, device)
    rows = prediction_rows(pd.read_parquet(args.manifest))
    predictions, embeddings = predict_images(
        model,
        rows,
        args.shard_dirs,
        image_size=config["data"]["image_size"],
        local_mean_norm=config["data"]["local_mean_norm"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(args.out_dir / "predictions.parquet", index=False)
    np.save(args.out_dir / "embeddings.npy", embeddings)
    print(f"{len(predictions):,} predictions written to {args.out_dir}")


if __name__ == "__main__":
    main()
