"""DR grader: a timm backbone with two heads, the 5-class ICDR grade and binary referable DR."""

from typing import NamedTuple, cast

import timm
import torch
import torch.nn.functional as F
from torch import nn

N_GRADES = 5


class GraderOutput(NamedTuple):
    grade_logits: torch.Tensor  # (N, 5)
    referable_logit: torch.Tensor  # (N,)
    embedding: torch.Tensor  # (N, D) pooled backbone features, for OOD and drift


class Grader(nn.Module):
    def __init__(self, timm_name: str, *, pretrained: bool, drop_path_rate: float = 0.0) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            timm_name, pretrained=pretrained, num_classes=0, drop_path_rate=drop_path_rate
        )
        features = cast(int, self.backbone.num_features)
        self.grade_head = nn.Linear(features, N_GRADES)
        self.referable_head = nn.Linear(features, 1)

    def forward(self, images: torch.Tensor) -> GraderOutput:
        embedding = self.backbone(images)
        return GraderOutput(
            self.grade_head(embedding), self.referable_head(embedding).squeeze(1), embedding
        )


def focal_cross_entropy(
    logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0
) -> torch.Tensor:
    """Cross-entropy down-weighting well-classified examples by (1 - p_true) ** gamma."""
    log_p = F.log_softmax(logits.float(), dim=1).gather(1, target[:, None]).squeeze(1)
    return (-((1 - log_p.exp()) ** gamma) * log_p).mean()


def grader_loss(
    output: GraderOutput,
    grades: torch.Tensor,
    referable: torch.Tensor,
    *,
    referable_weight: float = 1.0,
    focal_gamma: float | None = None,
) -> torch.Tensor:
    """Grade loss (cross-entropy, or focal when ``focal_gamma`` is set) plus weighted BCE on
    the referable head."""
    if focal_gamma is None:
        grade_loss = F.cross_entropy(output.grade_logits.float(), grades)
    else:
        grade_loss = focal_cross_entropy(output.grade_logits, grades, focal_gamma)
    referable_loss = F.binary_cross_entropy_with_logits(output.referable_logit.float(), referable)
    return grade_loss + referable_weight * referable_loss
