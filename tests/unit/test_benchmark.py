import torch

from retinaguard.train.benchmark import time_forward


def test_time_forward_is_positive() -> None:
    model = torch.nn.Conv2d(3, 4, 3)
    assert time_forward(model, torch.zeros(1, 3, 16, 16), repeats=3) > 0
