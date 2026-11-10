"""Replay a folder of fundus photos through the running API as one batch, optionally degraded.

    python -m retinaguard.monitoring.replay "IDRiD/1. Original Images/a. Training Set" \\
        --site idrid-blur-2 --blur 2 --out replay/idrid-blur-2.csv

Each photo is optionally blurred (Gaussian, ``--blur`` in pixels of a 512 px image) or darkened
(``--darken``, a brightness factor), re-encoded as JPEG and sent to ``POST /predict`` with the
batch's site label, so the service stores it like real traffic. Writes one row per photo:
file name, prediction id, decision, reason, calibrated probability, OOD score and latency.
"""

import argparse
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
import pandas as pd

from retinaguard.data.preprocess import Image, decode_rgb, encode_jpeg
from retinaguard.models.metrics import auroc
from retinaguard.monitoring.drift import alerts, decision_rates, psi

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
REFERENCE_WIDTH = 512


def degrade(img: Image, *, blur: float = 0.0, darken: float = 1.0) -> Image:
    """Gaussian blur of ``blur`` pixels at 512 px width (scaled to the image), then brightness
    multiplied by ``darken``."""
    out = img
    if blur > 0:
        sigma = blur * img.shape[1] / REFERENCE_WIDTH
        out = np.asarray(cv2.GaussianBlur(out, (0, 0), sigma), dtype=np.uint8)
    if darken != 1.0:
        out = np.clip(out.astype(np.float32) * darken, 0, 255).astype(np.uint8)
    return out


def replay(
    files: list[Path],
    api_url: str,
    site: str,
    *,
    blur: float = 0.0,
    darken: float = 1.0,
    transport: httpx.BaseTransport | None = None,
) -> pd.DataFrame:
    rows = []
    with httpx.Client(base_url=api_url, timeout=120, transport=transport) as client:
        for path in files:
            data = path.read_bytes()
            if blur > 0 or darken != 1.0:
                data = encode_jpeg(degrade(decode_rgb(data), blur=blur, darken=darken))
            response = client.post(
                "/predict", files={"image": (path.name, data)}, data={"site": site}
            )
            response.raise_for_status()
            body = response.json()
            rows.append(
                {
                    "file": path.name,
                    "site": site,
                    "prediction_id": body["prediction_id"],
                    "decision": body["decision"],
                    "reason": body["reason"],
                    "p_referable": body["p_referable"],
                    "ood_score": body["ood_score"],
                    "quality_label": body["quality_label"],
                    "latency_ms": body["latency_ms"],
                }
            )
    return pd.DataFrame(rows)


def replay_summary(tables: pd.DataFrame, reference: dict[str, Any]) -> pd.DataFrame:
    """Per replayed batch (``site``): drift signals against the reference and the real
    performance, from replay rows joined with their true ICDR grade (``dr_grade``)."""
    ood_threshold = float(reference["ood_threshold"])
    rows = []
    for site, batch in tables.groupby("site", sort=False):
        referable = (batch["dr_grade"].astype(int) >= 2).to_numpy()
        ood = batch["ood_score"].astype(float)
        rates = decision_rates(batch["decision"], (ood > ood_threshold)[ood.notna()])
        graded = (batch["decision"] != "RETAKE").to_numpy()
        cleared = (batch["decision"] == "NO_REFER").to_numpy()
        p = batch["p_referable"].astype(float).to_numpy()
        rows.append(
            {
                "batch": site,
                "photos": len(batch),
                **rates,
                "psi_p_referable": psi(reference["p_referable"], p),
                "psi_ood_score": psi(reference["ood_score"], ood),
                "alert": "; ".join(str(a) for a in alerts(rates, reference["rates"])),
                "auroc_graded": auroc(referable[graded], p[graded]),
                "fnr": float(cleared[referable & graded].mean()),
                "referable_retaken": float((~graded)[referable].mean()),
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--site", required=True, help="batch label stored with each prediction")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--blur", type=float, default=0.0)
    parser.add_argument("--darken", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    files = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        parser.error(f"no images in {args.folder}")
    table = replay(files, args.api_url, args.site, blur=args.blur, darken=args.darken)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"{args.site}: {len(table)} photos")
    print(table["decision"].value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()
