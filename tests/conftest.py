import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from retinaguard.data.preprocess import decode_rgb, encode_jpeg, preprocess
from retinaguard.data.processing import member_name, processed_path
from retinaguard.data.segmentation import IDRID_FOLDERS, build_shard, idrid_pairs, write_index

# Four IDRiD disease grading images (Porwal et al., doi:10.21227/H25W98, CC BY 4.0) and their
# grades.
IDRID_FIXTURES = {"IDRiD_005": 4, "IDRiD_016": 2, "IDRiD_138": 0, "IDRiD_139": 0}
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "idrid"


@pytest.fixture(scope="session")
def fixture_shard(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, pd.DataFrame]:
    """A shard of 64 processed images (16 ids per IDRiD fixture) and a manifest giving 48 to
    ``train`` and 16 to ``val``, every grade present in both."""
    shard_dir = tmp_path_factory.mktemp("shards")
    processed = {
        name: encode_jpeg(preprocess(decode_rgb((FIXTURE_DIR / f"{name}.jpg").read_bytes()))[0])
        for name in IDRID_FIXTURES
    }
    rows = []
    with zipfile.ZipFile(shard_dir / "fixture_000.zip", "w") as archive:
        for copy in range(16):
            for name, grade in IDRID_FIXTURES.items():
                image_id = f"idrid:{name}_{copy:02d}"
                archive.writestr(member_name(image_id), processed[name])
                rows.append(
                    {
                        "image_id": image_id,
                        "split": "val" if copy >= 12 else "train",
                        "processed_path": processed_path("fixture_000.zip", member_name(image_id)),
                        "dr_grade": grade,
                        "referable": grade >= 2,
                        "quality_label": None,
                    }
                )
    manifest = pd.DataFrame(rows).astype({"dr_grade": "Int64", "referable": "boolean"})
    return shard_dir, manifest


def _write_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(
        path.suffix, cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 else img
    )
    assert ok
    path.write_bytes(data.tobytes())


def write_raw_seg(root: Path, n_train: int = 10, n_test: int = 4) -> None:
    """A fake IDRiD ``A. Segmentation`` folder: orange discs with a few lesion blobs, every
    lesion type in every image except soft exudates (odd images only, no mask file otherwise)."""
    rng = np.random.default_rng(0)
    # numbering continues into the test set, as in the real release
    sets = (("a. Training Set", 1, n_train), ("b. Testing Set", n_train + 1, n_train + n_test))
    for folder, first, last in sets:
        for i in range(first, last + 1):
            stem = f"IDRiD_{i:02d}"
            img = np.zeros((300, 400, 3), np.uint8)
            cv2.circle(img, (200, 150), 140, (180, 90, 40), -1)
            for k, (lesion, sub) in enumerate(IDRID_FOLDERS.items()):
                if lesion == "SE" and i % 2 == 0:
                    continue
                mask = np.zeros((300, 400), np.uint8)
                for _ in range(3):
                    x, y = rng.integers(120, 280), rng.integers(70, 230)
                    cv2.circle(mask, (int(x), int(y)), 4 + 2 * k, 76, -1)
                    cv2.circle(img, (int(x), int(y)), 4 + 2 * k, (90 + 40 * k, 40, 160), -1)
                truth = root / "2. All Segmentation Groundtruths" / folder / sub
                _write_image(truth / f"{stem}_{lesion}.tif", mask)
            _write_image(root / "1. Original Images" / folder / f"{stem}.jpg", img)


@pytest.fixture(scope="session")
def seg_shard(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Prepared synthetic lesion shard (128 px) of a fake IDRiD release, folder with its index."""
    root = tmp_path_factory.mktemp("raw_seg")
    write_raw_seg(root)
    out = tmp_path_factory.mktemp("seg_shard")
    write_index(build_shard(idrid_pairs(root), out / "idrid_seg.zip", size=128), out)
    return out


@pytest.fixture(scope="session")
def models_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Tiny exported models (random weights) with a fitted-looking policy, as serving loads."""
    import json

    import torch

    from retinaguard.export.onnx_export import main as export_main
    from retinaguard.models.grader import Grader
    from retinaguard.models.ood import Mahalanobis
    from retinaguard.models.quality import quality_model
    from retinaguard.models.segmentation import seg_model

    root = tmp_path_factory.mktemp("models")
    grader = Grader("test_convnext", pretrained=False)
    checkpoints = {
        "quality": (
            quality_model("test_efficientnet", pretrained=False),
            {"model": {"timm_name": "test_efficientnet"}, "data": {"image_size": 64}},
        ),
        "grader": (
            grader,
            {
                "model": {"timm_name": "test_convnext"},
                "data": {"image_size": 64, "local_mean_norm": False},
            },
        ),
        "segmentation": (
            seg_model("resnet18", pretrained=False),
            {"model": {"encoder": "resnet18"}},
        ),
    }
    for name, (model, config) in checkpoints.items():
        torch.save({"model": model.state_dict(), "config": config}, root / f"{name}.pt")
    policy_dir = root / "policy"
    policy_dir.mkdir()
    dim = int(grader.backbone.num_features)
    rng = np.random.default_rng(0)
    Mahalanobis.fit(rng.normal(size=(200, dim)), rng.integers(0, 5, 200)).save(
        policy_dir / "ood.npz"
    )
    policy = {
        "lambda_low": 0.05,
        "lambda_high": 0.6,
        "temperature_referable": 1.3,
        "temperature_grade": 1.2,
        "ood_threshold": 1e6,  # nothing is out of distribution for random weights
    }
    (policy_dir / "policy.json").write_text(json.dumps(policy))
    out = root / "exported"
    export_main(
        [
            f"--out-dir={out}",
            f"--quality={root / 'quality.pt'}",
            f"--grader={root / 'grader.pt'}",
            f"--segmentation={root / 'segmentation.pt'}",
            "--segmentation-size=128",
            f"--policy-dir={policy_dir}",
        ]
    )
    return out
