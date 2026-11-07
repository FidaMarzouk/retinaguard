"""Train the DR grader on EyePACS ``train``, selecting epochs on ``val`` referable AUROC.

    python -m retinaguard.train.train_grader model=efficientnet_b3 "data.shard_dirs=[...]"

The run directory holds ``last.pt`` (full training state, replaced atomically after every epoch)
and ``best.pt`` (weights of the best validation epoch). Running again with the same run
directory continues after the last finished epoch, so a stopped Kaggle session loses at most one
epoch. Metrics go to ``metrics.jsonl`` via ``RunLogger``.
"""

import math
import os
import random
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Sampler

from retinaguard import __version__
from retinaguard.config import load_config
from retinaguard.data.datasets import (
    FundusDataset,
    grade_balancing_sampler,
    select_rows,
    train_augment,
)
from retinaguard.models.grader import Grader, grader_loss
from retinaguard.models.metrics import auroc, quadratic_kappa
from retinaguard.tracking import RunLogger, default_output_dir

IMBALANCE_MODES = ("none", "sampler", "focal")
LOG_EVERY_STEPS = 100


def warmup_cosine(warmup_steps: int, total_steps: int) -> Callable[[int], float]:
    """Learning-rate factor: linear warmup to 1, then cosine decay to 0 at ``total_steps``."""

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return factor


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def flatten(values: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Nested config as dotted keys, for ``RunLogger.log_params``."""
    flat: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, Mapping):
            flat.update(flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _init_worker(worker_id: int) -> None:
    cv2.setNumThreads(1)  # workers already run in parallel; avoid oversubscribing the CPUs
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    info = torch.utils.data.get_worker_info()
    augment = getattr(info.dataset, "_augment", None) if info is not None else None
    if augment is not None:
        augment.set_random_seed(seed)  # otherwise forked workers repeat the same augmentations


def _save_atomic(state: Mapping[str, Any], path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(dict(state), tmp)
    os.replace(tmp, path)


@torch.no_grad()
def predict(
    model: Grader, loader: DataLoader[Any], device: torch.device, amp: bool
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], float]:
    """Grade probabilities (N, 5), referable probabilities (N,) and the mean loss."""
    model.eval()
    grade_probs, referable_probs, losses = [], [], []
    for images, grades, referable in loader:
        images = images.to(device, memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            output = model(images)
        losses.append(float(grader_loss(output, grades.to(device), referable.float().to(device))))
        grade_probs.append(output.grade_logits.float().softmax(1).cpu().numpy())
        referable_probs.append(output.referable_logit.float().sigmoid().cpu().numpy())
    return np.concatenate(grade_probs), np.concatenate(referable_probs), float(np.mean(losses))


def validation_metrics(
    grades: npt.NDArray[np.integer[Any]],
    grade_probs: npt.NDArray[np.float32],
    referable_probs: npt.NDArray[np.float32],
) -> dict[str, float]:
    referable = grades >= 2
    return {
        "val_referable_auroc": auroc(referable, referable_probs),
        "val_referable_auroc_from_grades": auroc(referable, grade_probs[:, 2:].sum(1)),
        "val_quadratic_kappa": quadratic_kappa(grades, grade_probs.argmax(1)),
    }


def train(cfg: DictConfig) -> Path:
    """Train (or resume) the run described by ``cfg``; returns the run directory."""
    if cfg.train.imbalance not in IMBALANCE_MODES:
        raise ValueError(f"train.imbalance must be one of {IMBALANCE_MODES}")
    output_dir = Path(cfg.output_dir) if cfg.output_dir else default_output_dir() / "runs"
    run_dir: Path = output_dir / cfg.run_name
    logger = RunLogger(run_dir)
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    config = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(config, dict)
    OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)
    logger.log_params(flatten(cast(dict[str, Any], config)))
    logger.set_tags({"retinaguard_version": __version__, "task": "grader"})

    _seed_everything(cfg.seed)
    device = resolve_device(cfg.train.device)
    amp = bool(cfg.train.amp) and device.type == "cuda"
    manifest = pd.read_parquet(cfg.data.manifest)
    shard_dirs = [Path(d) for d in cfg.data.shard_dirs]
    rows = {
        split: select_rows(
            manifest,
            [split],
            exclude_quality_reject=cfg.data.exclude_quality_reject,
            limit_per_split=cfg.data.limit_per_split,
        )
        for split in ("train", "val")
    }
    datasets = {
        split: FundusDataset(
            split_rows,
            shard_dirs,
            size=cfg.data.image_size,
            local_mean_norm=cfg.data.local_mean_norm,
            augment=train_augment() if split == "train" else None,
        )
        for split, split_rows in rows.items()
    }
    print(f"device {device}, amp {amp}; images: { {s: len(d) for s, d in datasets.items()} }")

    shuffle_generator = torch.Generator()
    sampler: Sampler[int] | None = None
    if cfg.train.imbalance == "sampler":
        sampler = grade_balancing_sampler(datasets["train"].grades, cfg.seed)
    loader_args: dict[str, Any] = {
        "batch_size": cfg.train.batch_size,
        "num_workers": cfg.train.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": cfg.train.num_workers > 0,
        "worker_init_fn": _init_worker,
    }
    train_loader = DataLoader(
        datasets["train"],
        sampler=sampler,
        shuffle=sampler is None,
        generator=shuffle_generator,
        drop_last=True,
        **loader_args,
    )
    val_loader = DataLoader(datasets["val"], shuffle=False, **loader_args)

    model = Grader(
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
    focal_gamma = cfg.train.focal_gamma if cfg.train.imbalance == "focal" else None

    start_epoch, best_auroc, stale, finished = 0, -math.inf, 0, False
    if last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best_auroc = state["epoch"] + 1, state["best_auroc"]
        stale, finished = state["stale"], state["finished"]
        print(f"resuming after epoch {state['epoch']} (best val AUROC {best_auroc:.4f})")

    for epoch in range(start_epoch, cfg.train.epochs):
        if finished:
            break
        started = time.perf_counter()
        shuffle_generator.manual_seed(cfg.seed + epoch)
        if isinstance(sampler, torch.utils.data.WeightedRandomSampler):
            sampler.generator = torch.Generator().manual_seed(cfg.seed + epoch)
        model.train()
        losses = []
        for step, (images, grades, referable) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                output = model(images)
                loss = grader_loss(
                    output,
                    grades.to(device, non_blocking=True),
                    referable.float().to(device, non_blocking=True),
                    referable_weight=cfg.train.referable_loss_weight,
                    focal_gamma=focal_gamma,
                )
            optimizer.zero_grad(set_to_none=True)
            torch.autograd.backward(scaler.scale(loss))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach()))
            if step % LOG_EVERY_STEPS == 0:
                print(f"epoch {epoch} step {step}/{steps_per_epoch} loss {np.mean(losses):.4f}")

        grade_probs, referable_probs, val_loss = predict(model, val_loader, device, amp)
        metrics = validation_metrics(datasets["val"].grades, grade_probs, referable_probs)
        metrics |= {
            "train_loss": float(np.mean(losses)),
            "val_loss": val_loss,
            "lr": float(scheduler.get_last_lr()[0]),
            "epoch_minutes": (time.perf_counter() - started) / 60,
        }
        logger.log_metrics(metrics, step=epoch)

        current = metrics["val_referable_auroc"]
        if current > best_auroc or not best_path.exists():
            best_auroc, stale = current, 0
            _save_atomic(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_referable_auroc": current,
                },
                best_path,
            )
        else:
            stale += 1
        finished = stale >= cfg.train.patience  # early stop; raising train.epochs extends a run
        _save_atomic(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_auroc": best_auroc,
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


def main(argv: list[str] | None = None) -> None:
    train(load_config("train/grader", sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    main()
