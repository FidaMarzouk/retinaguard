"""Export the quality model, grader and lesion segmentation model to ONNX for CPU serving.

    python -m retinaguard.export.onnx_export --out-dir outputs/models \\
        --quality outputs/runs/quality-efficientnet_b0/best.pt \\
        --grader outputs/runs/grader-convnext_tiny-no_lmn/best.pt \\
        --segmentation outputs/runs/seg-pooled/best.pt \\
        --policy-dir data/interim/predictions/grader-convnext_tiny-no_lmn/policy

Each model is exported with a dynamic batch size (the image size is fixed), then checked
against PyTorch on smooth random images (largest absolute difference of every output must stay
below ``PARITY_TOLERANCE``).
Pixel-level noise is not used: it drives logits to hundreds, where float rounding alone
exceeds the tolerance although real fundus images agree to about 1e-4. Writes
``<name>.onnx`` per model and ``models.json``: for each, its input size, input preprocessing,
output names, source checkpoint, SHA-256 and the parity result. ``--policy-dir`` copies the
fitted ``policy.json`` and ``ood.npz`` alongside. Serving reads only this folder.
"""

import argparse
import hashlib
import json
import shutil
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from torch import nn

from retinaguard.data.datasets import IMAGENET_MEAN, IMAGENET_STD
from retinaguard.data.segmentation import SEG_SIZE
from retinaguard.models.quality import QUALITY_CLASSES, quality_model
from retinaguard.models.segmentation import seg_model
from retinaguard.train.predict import load_grader

PARITY_TOLERANCE = 1e-3
OPSET = 18
MAX_BATCH = 64


@dataclass(frozen=True)
class ExportedModel:
    name: str
    file: str
    image_size: int
    local_mean_norm: bool
    outputs: list[str]
    checkpoint: str
    sha256: str
    max_abs_diff: float


class GraderOutputs(nn.Module):
    """The grader with plain tensor outputs (ONNX cannot return a NamedTuple)."""

    def __init__(self, grader: nn.Module) -> None:
        super().__init__()
        self.grader = grader

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.grader(images)
        return out.grade_logits, out.referable_logit, out.embedding


def example_images(batch: int, image_size: int, seed: int = 0) -> torch.Tensor:
    """Smooth random colour images scaled like model inputs (ImageNet statistics)."""
    generator = torch.Generator().manual_seed(seed)
    coarse = torch.rand(batch, 3, 16, 16, generator=generator)
    pixels = F.interpolate(coarse, size=(image_size, image_size), mode="bilinear")
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (pixels - mean) / std


def export_onnx(
    model: nn.Module, image_size: int, outputs: Sequence[str], path: Path, batch: int = 2
) -> float:
    """Write ``model`` to ``path`` and return the largest output difference to PyTorch."""
    model = model.eval()
    example = example_images(batch, image_size)
    batch_dim = torch.export.Dim("batch", min=1, max=MAX_BATCH)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (example,),
        str(path),
        input_names=["images"],
        output_names=list(outputs),
        dynamic_shapes=({0: batch_dim},),
        opset_version=OPSET,
        dynamo=True,
        verbose=False,
        external_data=False,  # one self-contained file per model (all are far below 2 GB)
    )
    with torch.no_grad():
        expected = model(example)
    expected = expected if isinstance(expected, tuple) else (expected,)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(list(outputs), {"images": example.numpy()})
    return max(float(np.abs(a - e.numpy()).max()) for a, e in zip(actual, expected, strict=True))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(
    name: str,
    path: Path,
    checkpoint: Path,
    image_size: int,
    local_mean_norm: bool,
    outputs: Sequence[str],
    diff: float,
) -> ExportedModel:
    if not diff < PARITY_TOLERANCE:
        raise ValueError(f"{name}: ONNX differs from PyTorch by {diff:.2e}")
    return ExportedModel(
        name,
        path.name,
        image_size,
        local_mean_norm,
        list(outputs),
        checkpoint.as_posix(),
        _sha256(path),
        diff,
    )


def export_quality(checkpoint: Path, out_dir: Path) -> ExportedModel:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config: dict[str, Any] = state["config"]
    model = quality_model(config["model"]["timm_name"], pretrained=False)
    model.load_state_dict(state["model"])
    size, outputs = config["data"]["image_size"], ["quality_logits"]
    path = out_dir / "quality.onnx"
    diff = export_onnx(model, size, outputs, path)
    return _record("quality", path, checkpoint, size, False, outputs, diff)


def export_grader(checkpoint: Path, out_dir: Path) -> ExportedModel:
    grader, config = load_grader(checkpoint, torch.device("cpu"))
    size = config["data"]["image_size"]
    outputs = ["grade_logits", "referable_logit", "embedding"]
    path = out_dir / "grader.onnx"
    diff = export_onnx(GraderOutputs(grader), size, outputs, path)
    lmn = bool(config["data"]["local_mean_norm"])
    return _record("grader", path, checkpoint, size, lmn, outputs, diff)


def export_segmentation(
    checkpoint: Path, out_dir: Path, image_size: int = SEG_SIZE
) -> ExportedModel:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = seg_model(state["config"]["model"]["encoder"], pretrained=False)
    model.load_state_dict(state["model"])
    outputs = ["lesion_logits"]
    path = out_dir / "segmentation.onnx"
    diff = export_onnx(model, image_size, outputs, path)
    return _record("segmentation", path, checkpoint, image_size, False, outputs, diff)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--quality", type=Path)
    parser.add_argument("--grader", type=Path)
    parser.add_argument("--segmentation", type=Path)
    parser.add_argument("--segmentation-size", type=int, default=SEG_SIZE)
    parser.add_argument(
        "--policy-dir", type=Path, help="calibrate/fit_ood output: policy.json and ood.npz"
    )
    args = parser.parse_args(argv)

    exported = []
    if args.quality:
        exported.append(export_quality(args.quality, args.out_dir))
    if args.grader:
        exported.append(export_grader(args.grader, args.out_dir))
    if args.segmentation:
        exported.append(
            export_segmentation(args.segmentation, args.out_dir, args.segmentation_size)
        )
    if not exported and not args.policy_dir:
        parser.error("give at least one checkpoint or --policy-dir")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.policy_dir:
        for name in ("policy.json", "ood.npz"):
            if (args.policy_dir / name).exists():
                shutil.copyfile(args.policy_dir / name, args.out_dir / name)
    manifest_path = args.out_dir / "models.json"
    models = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    models |= {m.name: asdict(m) for m in exported}
    models["quality_classes"] = list(QUALITY_CLASSES)
    manifest_path.write_text(json.dumps(models, indent=2), encoding="utf-8")
    for m in exported:
        print(f"{m.name}: {m.file}, {m.image_size} px, max |ONNX - PyTorch| {m.max_abs_diff:.1e}")


if __name__ == "__main__":
    main()
