import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from retinaguard.config import load_config
from retinaguard.models.grader import Grader, focal_cross_entropy, grader_loss
from retinaguard.models.metrics import auroc, quadratic_kappa
from retinaguard.train.train_grader import flatten, validation_metrics, warmup_cosine

TINY_BACKBONE = "test_convnext"  # timm's few-thousand-parameter test model


def test_grader_heads() -> None:
    model = Grader(TINY_BACKBONE, pretrained=False)
    output = model(torch.randn(2, 3, 64, 64))
    assert output.grade_logits.shape == (2, 5)
    assert output.referable_logit.shape == (2,)
    assert output.embedding.shape == (2, model.backbone.num_features)


def test_losses() -> None:
    logits = torch.randn(8, 5)
    target = torch.randint(0, 5, (8,))
    assert focal_cross_entropy(logits, target, gamma=0).item() == pytest.approx(
        F.cross_entropy(logits, target).item(), rel=1e-5
    )
    assert focal_cross_entropy(logits, target, gamma=2) < F.cross_entropy(logits, target)
    model = Grader(TINY_BACKBONE, pretrained=False)
    output = model(torch.randn(4, 3, 64, 64))
    loss = grader_loss(output, torch.tensor([0, 1, 2, 4]), torch.tensor([0.0, 0, 1, 1]))
    loss.backward()
    assert math.isfinite(loss.item())


def test_warmup_cosine() -> None:
    factor = warmup_cosine(warmup_steps=10, total_steps=110)
    assert factor(0) == pytest.approx(0.1)
    assert factor(9) == pytest.approx(1.0)
    assert factor(60) == pytest.approx(0.5)
    assert factor(110) == pytest.approx(0.0)


def test_metrics() -> None:
    assert auroc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    assert math.isnan(auroc([1, 1], [0.2, 0.3]))
    assert quadratic_kappa([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]) == pytest.approx(1.0)
    grades = np.array([0, 0, 2, 4])
    probs = np.eye(5, dtype=np.float32)[[0, 1, 2, 4]]
    metrics = validation_metrics(grades, probs, np.array([0.1, 0.2, 0.9, 0.8], np.float32))
    assert metrics["val_referable_auroc"] == 1.0
    assert metrics["val_referable_auroc_from_grades"] == 1.0


def test_config_composes() -> None:
    cfg = load_config("train/grader")
    assert cfg.model.name == "convnext_tiny"
    assert cfg.run_name == "grader-convnext_tiny"
    assert cfg.data.image_size == 512
    other = load_config("train/grader", ["model=efficientnet_b3", "train.imbalance=focal"])
    assert other.model.timm_name.startswith("efficientnet_b3")
    assert other.run_name == "grader-efficientnet_b3"
    assert flatten({"a": {"b": 1}, "c": 2}) == {"a.b": 1, "c": 2}
