"""CPU inference latency of grader backbones, for the model comparison.

    python -m retinaguard.train.benchmark convnext_tiny efficientnet_b3 dinov2_vitb

For each model config: parameter count, median latency of one 512 px image (batch 1, the
serving case) and throughput at batch 16, in PyTorch eager float32 on CPU. Weights are random:
speed does not depend on them, and no download is needed.
"""

import argparse
import statistics
import time

import pandas as pd
import torch

from retinaguard.config import load_config, timm_kwargs
from retinaguard.models.grader import Grader


@torch.inference_mode()
def time_forward(
    model: torch.nn.Module, batch: torch.Tensor, repeats: int, warmup: int = 2
) -> float:
    """Median seconds per forward pass."""
    for _ in range(warmup):
        model(batch)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        model(batch)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def benchmark(model_name: str, image_size: int = 512, repeats: int = 10) -> dict[str, float | str]:
    cfg = load_config("train/grader", [f"model={model_name}"]).model
    model = Grader(cfg.timm_name, pretrained=False, timm_kwargs=timm_kwargs(cfg)).eval()
    single = time_forward(model, torch.randn(1, 3, image_size, image_size), repeats)
    batch = time_forward(model, torch.randn(16, 3, image_size, image_size), max(2, repeats // 4))
    return {
        "model": model_name,
        "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
        "latency_ms": single * 1000,
        "images_per_s_batch16": 16 / batch,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("models", nargs="+")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args(argv)

    print(f"torch {torch.__version__}, {torch.get_num_threads()} CPU threads")
    table = pd.DataFrame([benchmark(m, args.image_size, args.repeats) for m in args.models])
    print(table.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
