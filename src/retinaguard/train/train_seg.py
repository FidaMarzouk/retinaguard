"""Train the lesion segmentation U-Net, selecting on site-balanced validation AUPR.

    python -m retinaguard.train.train_seg "data.shard_dirs=[...]"
    python -m retinaguard.train.train_seg "data.train_sites=[ddr]" run_name=seg-ddr_only ...

Trains on random crops (``data.crop_size``) of the ``train`` images of ``data.train_sites``,
part of them centred on a lesion pixel so rare lesions are seen. Every ``train.val_every``
epochs the model segments the full ``val`` images of the training sites; the score is the mean
over those sites of the mean AUPR over lesion types, so a small site counts as much as a large
one, and a held-out site is never used for selection. After training, the best epoch segments
every image not used for training (``stats.npz``: per-image pixel counts for ``evaluate_seg``).
Resumable like the grader: ``last.pt`` after every epoch, ``best.pt`` for the best validation.
"""

import math
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

import albumentations as A
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler

from retinaguard import __version__
from retinaguard.config import load_config
from retinaguard.data.datasets import IMAGENET_MEAN, IMAGENET_STD
from retinaguard.data.segmentation import LESIONS, SegShards, read_index
from retinaguard.models.segmentation import (
    PixelStats,
    SegLoss,
    aupr,
    concat_stats,
    dice,
    image_stats,
    seg_model,
)
from retinaguard.tracking import RunLogger, default_output_dir
from retinaguard.train.train_grader import (
    _init_worker,
    _save_atomic,
    _seed_everything,
    flatten,
    resolve_device,
    warmup_cosine,
)

LOG_EVERY_STEPS = 50


def _worker_context(num_workers: int) -> str | None:
    """Start loader workers fresh ("spawn") rather than forked: a fork made after OpenCV's
    threads have run (e.g. preparing the lesion set in the same notebook) can hang on its
    first image decode."""
    return "spawn" if num_workers > 0 else None


def seg_augment() -> A.Compose:
    """Flips, right-angle rotations and mild photometric changes; no elastic distortion, which
    would move lesion boundaries."""
    return A.Compose(
        [
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        ]
    )


def lesion_crop(
    masks: npt.NDArray[np.bool_], size: int, rng: np.random.Generator, on_lesion: bool
) -> tuple[int, int]:
    """Top-left corner of a ``size`` crop: centred on a random lesion pixel when ``on_lesion``
    and the image has one, else uniform."""
    height, width = masks.shape[1:]
    if on_lesion:
        ys, xs = np.nonzero(masks.any(axis=0))
        if len(ys):
            k = rng.integers(len(ys))
            top = int(np.clip(ys[k] - size // 2, 0, height - size))
            left = int(np.clip(xs[k] - size // 2, 0, width - size))
            return top, left
    return int(rng.integers(0, height - size + 1)), int(rng.integers(0, width - size + 1))


def to_tensors(
    image: npt.NDArray[np.uint8], masks: npt.NDArray[np.bool_]
) -> tuple[torch.Tensor, torch.Tensor]:
    scaled = (image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return (
        torch.from_numpy(np.ascontiguousarray(scaled.transpose(2, 0, 1))),
        torch.from_numpy(masks.astype(np.float32)),
    )


class SegDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Index rows as (image CHW, masks LHW): random crops when ``crop_size`` is set, else the
    full prepared image."""

    def __init__(
        self,
        rows: pd.DataFrame,
        *,
        crop_size: int | None = None,
        lesion_crop_fraction: float = 0.0,
        augment: A.Compose | None = None,
    ) -> None:
        self.rows = rows.reset_index(drop=True)
        self.image_ids: list[str] = self.rows["image_id"].tolist()
        self._crop_size = crop_size
        self._lesion_crop_fraction = lesion_crop_fraction
        self._augment = augment
        self._reader: SegShards | None = None
        self._reader_pid = -1

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._reader is None or self._reader_pid != os.getpid():
            self._reader = SegShards(self.rows)
            self._reader_pid = os.getpid()
        image, masks = self._reader.read(self.image_ids[index])
        if self._crop_size is not None:
            rng = np.random.default_rng(np.random.randint(2**31))
            on_lesion = rng.random() < self._lesion_crop_fraction
            top, left = lesion_crop(masks, self._crop_size, rng, on_lesion)
            image = image[top : top + self._crop_size, left : left + self._crop_size]
            masks = masks[:, top : top + self._crop_size, left : left + self._crop_size]
        if self._augment is not None:
            out = self._augment(image=image, mask=masks.transpose(1, 2, 0).astype(np.uint8))
            image, masks = out["image"], out["mask"].transpose(2, 0, 1).astype(bool)
        return to_tensors(image, masks)


@torch.no_grad()
def segment(
    model: nn.Module, rows: pd.DataFrame, device: torch.device, amp: bool, num_workers: int = 0
) -> PixelStats:
    """Pixel counts of the full images of ``rows``, in row order."""
    model.eval()
    loader = DataLoader(
        SegDataset(rows),
        batch_size=1,
        num_workers=num_workers,
        multiprocessing_context=_worker_context(num_workers),
    )
    parts = []
    for images, masks in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = model(images.to(device))
        probs = logits.float().sigmoid()[0].cpu().numpy()
        parts.append(image_stats(probs, masks[0].numpy().astype(bool)))
    return concat_stats(parts)


def site_scores(rows: pd.DataFrame, stats: PixelStats) -> dict[str, float]:
    """``val_aupr_<site>_<lesion>``, ``val_dice_<site>`` and ``val_score`` (mean over sites of
    the mean AUPR over lesions)."""
    metrics: dict[str, float] = {}
    site_means = []
    for site in sorted(rows["site"].unique()):
        part = stats.take(np.flatnonzero(rows["site"].to_numpy() == site))
        values = aupr(part)
        metrics |= {
            f"val_aupr_{site}_{lesion}": float(v) for lesion, v in zip(LESIONS, values, strict=True)
        }
        metrics[f"val_dice_{site}"] = float(np.nanmean(dice(part)))
        site_means.append(float(np.nanmean(values)))
    metrics["val_score"] = float(np.mean(site_means))
    return metrics


def _rows(cfg: DictConfig) -> pd.DataFrame:
    index = read_index([Path(d) for d in cfg.data.shard_dirs]).sort_values("image_id")
    if cfg.data.limit_per_split is not None:
        index = index.groupby(["site", "split"], sort=False).head(cfg.data.limit_per_split)
    return index.reset_index(drop=True)


def train(cfg: DictConfig) -> Path:
    """Train (or resume) the run described by ``cfg``; returns the run directory."""
    output_dir = Path(cfg.output_dir) if cfg.output_dir else default_output_dir() / "runs"
    run_dir: Path = output_dir / cfg.run_name
    logger = RunLogger(run_dir)
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    config = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(config, dict)
    OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)
    logger.log_params(flatten(cast(dict[str, Any], config)))
    logger.set_tags({"retinaguard_version": __version__, "task": "segmentation"})

    _seed_everything(cfg.seed)
    device = resolve_device(cfg.train.device)
    amp = bool(cfg.train.amp) and device.type == "cuda"
    index = _rows(cfg)
    sites = list(cfg.data.train_sites)
    train_rows = index[(index["split"] == "train") & index["site"].isin(sites)]
    val_rows = index[(index["split"] == "val") & index["site"].isin(sites)].reset_index(drop=True)
    if train_rows.empty or val_rows.empty:
        raise ValueError(f"no train or val images for sites {sites}")
    dataset = SegDataset(
        train_rows,
        crop_size=cfg.data.crop_size,
        lesion_crop_fraction=cfg.data.lesion_crop_fraction,
        augment=seg_augment(),
    )
    print(f"device {device}, amp {amp}; train {len(train_rows)}, val {len(val_rows)} images")

    generator = torch.Generator()
    sampler = RandomSampler(
        dataset, replacement=True, num_samples=cfg.train.samples_per_epoch, generator=generator
    )
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.train.num_workers > 0,
        worker_init_fn=_init_worker,
        multiprocessing_context=_worker_context(cfg.train.num_workers),
        drop_last=True,
    )
    model = seg_model(cfg.model.encoder, pretrained=cfg.model.pretrained).to(device)
    criterion = SegLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    steps_per_epoch = len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        warmup_cosine(
            cfg.train.warmup_epochs * steps_per_epoch, cfg.train.epochs * steps_per_epoch
        ),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    start_epoch, best_score, stale, finished = 0, -math.inf, 0, False
    if last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best_score = state["epoch"] + 1, state["best_score"]
        stale, finished = state["stale"], state["finished"]
        print(f"resuming after epoch {state['epoch']} (best val score {best_score:.4f})")

    for epoch in range(start_epoch, cfg.train.epochs):
        if finished:
            break
        started = time.perf_counter()
        generator.manual_seed(cfg.seed + epoch)
        model.train()
        losses = []
        for step, (images, masks) in enumerate(loader, start=1):
            images, masks = (
                images.to(device, non_blocking=True),
                masks.to(device, non_blocking=True),
            )
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits = model(images)
            loss = criterion(logits, masks)
            optimizer.zero_grad(set_to_none=True)
            torch.autograd.backward(scaler.scale(loss))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach()))
            if step == 1 and epoch == start_epoch:
                print(f"first batch after {time.perf_counter() - started:.0f} s", flush=True)
            if step % LOG_EVERY_STEPS == 0:
                print(
                    f"epoch {epoch} step {step}/{steps_per_epoch} loss {np.mean(losses):.4f}",
                    flush=True,
                )

        metrics = {"train_loss": float(np.mean(losses)), "lr": float(scheduler.get_last_lr()[0])}
        last_epoch = epoch == cfg.train.epochs - 1
        if (epoch + 1) % cfg.train.val_every == 0 or last_epoch:
            metrics |= site_scores(val_rows, segment(model, val_rows, device, amp))
            current = metrics["val_score"]
            if current > best_score or not best_path.exists():
                best_score, stale = current, 0
                _save_atomic(
                    {
                        "model": model.state_dict(),
                        "config": config,
                        "epoch": epoch,
                        "val_score": current,
                    },
                    best_path,
                )
            else:
                stale += 1
            finished = stale >= cfg.train.patience
        metrics["epoch_minutes"] = (time.perf_counter() - started) / 60
        logger.log_metrics(metrics, step=epoch)
        _save_atomic(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_score": best_score,
                "stale": stale,
                "finished": finished,
            },
            last_path,
        )
        shown = {k: v for k, v in metrics.items() if "_aupr_" not in k}
        print(
            f"epoch {epoch}: "
            + ", ".join(f"{k} {v:.4f}" for k, v in shown.items())
            + (" (early stop)" if finished else ""),
            flush=True,
        )
    return run_dir


def predict_run(cfg: DictConfig, run_dir: Path) -> pd.DataFrame:
    """Pixel counts of the best epoch on every image not used for training, written to
    ``run_dir/stats.npz``; returns the rows segmented."""
    device = resolve_device(cfg.train.device)
    amp = bool(cfg.train.amp) and device.type == "cuda"
    state = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model = seg_model(cfg.model.encoder, pretrained=False)
    model.load_state_dict(state["model"])
    model.to(device)
    index = _rows(cfg)
    trained_on = (index["split"] == "train") & index["site"].isin(list(cfg.data.train_sites))
    rows = index[~trained_on].reset_index(drop=True)
    stats = segment(model, rows, device, amp, num_workers=cfg.train.num_workers)
    np.savez_compressed(
        run_dir / "stats.npz",
        image_id=np.asarray(rows["image_id"], dtype=str),
        site=np.asarray(rows["site"], dtype=str),
        split=np.asarray(rows["split"], dtype=str),
        tp=stats.tp,
        fp=stats.fp,
        fn=stats.fn,
        positive_hist=stats.positive_hist,
        negative_hist=stats.negative_hist,
    )
    return rows


def main(argv: list[str] | None = None) -> None:
    cfg = load_config("train/segmentation", sys.argv[1:] if argv is None else argv)
    run_dir = train(cfg)
    rows = predict_run(cfg, run_dir)
    print(f"{len(rows):,} images segmented; counts in {run_dir / 'stats.npz'}")


if __name__ == "__main__":
    main()
