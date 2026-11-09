import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from retinaguard.export.onnx_export import PARITY_TOLERANCE, _record, export_onnx, main
from retinaguard.models.grader import Grader
from retinaguard.models.quality import quality_model
from retinaguard.models.segmentation import seg_model


def _checkpoint(path: Path, model: torch.nn.Module, config: dict[str, object]) -> Path:
    torch.save({"model": model.state_dict(), "config": config}, path)
    return path


def test_export_all_models(tmp_path: Path) -> None:
    quality = _checkpoint(
        tmp_path / "quality.pt",
        quality_model("test_efficientnet", pretrained=False),
        {"model": {"timm_name": "test_efficientnet"}, "data": {"image_size": 64}},
    )
    grader = _checkpoint(
        tmp_path / "grader.pt",
        Grader("test_convnext", pretrained=False),
        {
            "model": {"timm_name": "test_convnext"},
            "data": {"image_size": 64, "local_mean_norm": False},
        },
    )
    segmentation = _checkpoint(
        tmp_path / "seg.pt",
        seg_model("resnet18", pretrained=False),
        {"model": {"encoder": "resnet18"}},
    )
    out = tmp_path / "models"
    main(
        [
            f"--out-dir={out}",
            f"--quality={quality}",
            f"--grader={grader}",
            f"--segmentation={segmentation}",
        ]
    )
    models = json.loads((out / "models.json").read_text())
    assert models["quality_classes"] == ["good", "usable", "reject"]
    for name in ("quality", "grader", "segmentation"):
        assert models[name]["max_abs_diff"] < PARITY_TOLERANCE
        assert (out / models[name]["file"]).exists()
    session = ort.InferenceSession(str(out / "grader.onnx"), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"images": np.zeros((3, 3, 64, 64), np.float32)})
    assert [o.shape for o in outputs][:2] == [(3, 5), (3,)]  # batch size is dynamic


def test_export_refuses_a_model_that_does_not_match(tmp_path: Path) -> None:
    class RandomOutput(torch.nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return torch.rand(images.shape[0], 3)  # differs between PyTorch and ONNX

    diff = export_onnx(RandomOutput(), 32, ["out"], tmp_path / "random.onnx")
    assert diff > PARITY_TOLERANCE
    with pytest.raises(ValueError, match="differs from PyTorch"):
        _record("random", tmp_path / "random.onnx", tmp_path / "x.pt", 32, False, ["out"], diff)
