"""Train the image quality model on EyeQ labels, selecting epochs on ``val`` macro-F1.

    python -m retinaguard.train.train_quality "data.shard_dirs=[...]"

EyeQ labels exist for EyePACS images only; the patient-level EyePACS splits are reused, so the
model trains on ``train``, is selected on ``val`` and is measured on ``test_internal``. The loss
is cross-entropy weighted by inverse class frequency. After training, the best epoch predicts
every processed image of the predicted splits (``predictions.parquet`` in the run directory).
Resumable like the grader: ``last.pt`` after every epoch, ``best.pt`` for the best epoch.
"""

import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from retinaguard import __version__
from retinaguard.config import load_config
from retinaguard.data.datasets import QualityDataset, quality_augment, select_quality_rows
from retinaguard.models.quality import QUALITY_COLUMNS, class_weights, macro_f1, quality_model
from retinaguard.tracking import RunLogger, default_output_dir
from retinaguard.train.predict import prediction_rows
from retinaguard.train.train_grader import (
    _init_worker,
    _save_atomic,
    _seed_everything,
    flatten,
    resolve_device,
    warmup_cosine,
)

LOG_EVERY_STEPS = 100


@torch.no_grad()
def predict_probs(
    model: nn.Module, loader: DataLoader[Any], device: torch.device, amp: bool
) -> npt.NDArray[np.float32]:
    """Class probabilities (N, 3) in loader order."""
    model.eval()
    probs = []
    for images, _, _ in loader:
        images = images.to(device, memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = model(images)
        probs.append(logits.float().softmax(1).cpu().numpy())
    return np.concatenate(probs)


def predict_quality(
    model: nn.Module,
    rows: pd.DataFrame,
    shard_dirs: Sequence[Path],
    *,
    image_size: int,
    batch_size: int = 64,
    num_workers: int = 4,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """``image_id`` and ``p_good``, ``p_usable``, ``p_reject`` for ``rows``."""
    device = device or next(model.parameters()).device
    dataset = QualityDataset(rows, shard_dirs, size=image_size)
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=device.type == "cuda"
    )
    probs = predict_probs(model.to(device), loader, device, device.type == "cuda")
    predictions = pd.DataFrame(probs, columns=QUALITY_COLUMNS)
    predictions.insert(0, "image_id", dataset.image_ids)
    return predictions


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
    logger.set_tags({"retinaguard_version": __version__, "task": "quality"})

    _seed_everything(cfg.seed)
    device = resolve_device(cfg.train.device)
    amp = bool(cfg.train.amp) and device.type == "cuda"
    manifest = pd.read_parquet(cfg.data.manifest)
    shard_dirs = [Path(d) for d in cfg.data.shard_dirs]
    datasets = {
        split: QualityDataset(
            select_quality_rows(manifest, [split], limit_per_split=cfg.data.limit_per_split),
            shard_dirs,
            size=cfg.data.image_size,
            augment=quality_augment() if split == "train" else None,
        )
        for split in ("train", "val")
    }
    print(f"device {device}, amp {amp}; images: { {s: len(d) for s, d in datasets.items()} }")

    shuffle_generator = torch.Generator()
    loader_args: dict[str, Any] = {
        "batch_size": cfg.train.batch_size,
        "num_workers": cfg.train.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": cfg.train.num_workers > 0,
        "worker_init_fn": _init_worker,
    }
    train_loader = DataLoader(
        datasets["train"], shuffle=True, generator=shuffle_generator, drop_last=True, **loader_args
    )
    val_loader = DataLoader(datasets["val"], shuffle=False, **loader_args)

    weights = torch.from_numpy(class_weights(datasets["train"].labels)).to(device)
    model = quality_model(
        cfg.model.timm_name,
        pretrained=cfg.model.pretrained,
        drop_path_rate=cfg.model.drop_path_rate,
    ).to(device)
    model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        warmup_cosine(
            cfg.train.warmup_epochs * steps_per_epoch, cfg.train.epochs * steps_per_epoch
        ),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    start_epoch, best_f1, stale, finished = 0, -math.inf, 0, False
    if last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best_f1 = state["epoch"] + 1, state["best_f1"]
        stale, finished = state["stale"], state["finished"]
        print(f"resuming after epoch {state['epoch']} (best val macro-F1 {best_f1:.4f})")

    for epoch in range(start_epoch, cfg.train.epochs):
        if finished:
            break
        started = time.perf_counter()
        shuffle_generator.manual_seed(cfg.seed + epoch)
        model.train()
        losses = []
        for step, (images, labels, _) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits = model(images)
            loss = F.cross_entropy(logits.float(), labels.to(device), weight=weights)
            optimizer.zero_grad(set_to_none=True)
            torch.autograd.backward(scaler.scale(loss))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach()))
            if step % LOG_EVERY_STEPS == 0:
                print(f"epoch {epoch} step {step}/{steps_per_epoch} loss {np.mean(losses):.4f}")

        probs = predict_probs(model, val_loader, device, amp)
        val_labels = datasets["val"].labels
        val_loss = float(
            F.cross_entropy(
                torch.from_numpy(np.log(np.clip(probs, 1e-7, 1))),
                torch.from_numpy(val_labels),
                weight=weights.cpu(),
            )
        )
        metrics = {
            "val_macro_f1": macro_f1(val_labels, probs.argmax(1)),
            "val_recall_reject": float(np.mean(probs.argmax(1)[val_labels == 2] == 2)),
            "train_loss": float(np.mean(losses)),
            "val_loss": val_loss,
            "lr": float(scheduler.get_last_lr()[0]),
            "epoch_minutes": (time.perf_counter() - started) / 60,
        }
        logger.log_metrics(metrics, step=epoch)

        current = metrics["val_macro_f1"]
        if current > best_f1 or not best_path.exists():
            best_f1, stale = current, 0
            _save_atomic(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_macro_f1": current,
                },
                best_path,
            )
        else:
            stale += 1
        finished = stale >= cfg.train.patience
        _save_atomic(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_f1": best_f1,
                "stale": stale,
                "finished": finished,
            },
            last_path,
        )
        print(
            f"epoch {epoch}: "
            + ", ".join(f"{k} {v:.4f}" for k, v in metrics.items())
            + (" (early stop)" if finished else ""),
            flush=True,
        )
    return run_dir


def predict_run(cfg: DictConfig, run_dir: Path) -> pd.DataFrame:
    """Predictions of the best epoch for every processed image of the predicted splits,
    written to ``run_dir/predictions.parquet``."""
    device = resolve_device(cfg.train.device)
    state = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model = quality_model(cfg.model.timm_name, pretrained=False)
    model.load_state_dict(state["model"])
    rows = prediction_rows(pd.read_parquet(cfg.data.manifest))
    if cfg.data.limit_per_split is not None:
        rows = rows.groupby("split", sort=False).head(cfg.data.limit_per_split)
    predictions = predict_quality(
        model,
        rows,
        [Path(d) for d in cfg.data.shard_dirs],
        image_size=cfg.data.image_size,
        num_workers=cfg.train.num_workers,
        device=device,
    )
    predictions.to_parquet(run_dir / "predictions.parquet", index=False)
    return predictions


def main(argv: list[str] | None = None) -> None:
    cfg = load_config("train/quality", sys.argv[1:] if argv is None else argv)
    run_dir = train(cfg)
    predictions = predict_run(cfg, run_dir)
    print(f"{len(predictions):,} predictions written to {run_dir / 'predictions.parquet'}")


if __name__ == "__main__":
    main()
