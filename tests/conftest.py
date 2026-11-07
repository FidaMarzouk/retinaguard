import zipfile
from pathlib import Path

import pandas as pd
import pytest

from retinaguard.data.preprocess import decode_rgb, encode_jpeg, preprocess
from retinaguard.data.processing import member_name, processed_path

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
