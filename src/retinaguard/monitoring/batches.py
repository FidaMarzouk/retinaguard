"""Drift reports of stored predictions, one per site and day, against a reference.

    python -m retinaguard.monitoring.batches reference --manifest data/manifests/manifest.parquet \\
        --grader data/interim/predictions/grader-convnext_tiny-no_lmn \\
        --quality data/interim/predictions/quality-efficientnet_b0 \\
        --image-stats data/interim/image_stats/image_stats.parquet \\
        --out outputs/models/reference.json
    python -m retinaguard.monitoring.batches report --reference outputs/models/reference.json

The reference (EyePACS ``calib``, all processed images, as the deployed chain decides them)
keeps the calibrated referable probabilities, OOD scores and decision rates. A report compares
each (site, day) batch of the ``predictions`` table with it: PSI of the referable probability
and of the OOD score (the service stores no embeddings, so the OOD score stands in for them),
the decision rates and the alert rule of ``drift``. Reports are upserted into ``drift_reports``.
"""

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import delete, insert, select

from retinaguard.monitoring.drift import alerts, decision_rates, psi
from retinaguard.serving.store import PredictionStore, drift_reports, predictions

MIN_BATCH = 20  # smaller batches are reported but never alert


def build_reference(scored: pd.DataFrame, ood_threshold: float) -> dict[str, Any]:
    """Reference from scored images (``p_referable``, ``ood_score``, ``decision``)."""
    graded = scored[scored["decision"] != "RETAKE"]
    rates = decision_rates(scored["decision"], scored["ood_score"] > ood_threshold)
    return {
        "images": len(scored),
        "ood_threshold": ood_threshold,
        "rates": rates,
        "p_referable": np.round(graded["p_referable"].to_numpy(), 6).tolist(),
        "ood_score": np.round(graded["ood_score"].to_numpy(), 4).tolist(),
    }


def batch_reports(records: pd.DataFrame, reference: dict[str, Any]) -> pd.DataFrame:
    """One row per (site, day) of stored prediction records."""
    if records.empty:
        return pd.DataFrame()
    frame = records.assign(
        site=records["site"].fillna("unknown"),
        day=pd.to_datetime(records["created_at"], utc=True).dt.strftime("%Y-%m-%d"),
    )
    rows = []
    for (site, day), batch in frame.groupby(["site", "day"], sort=True):
        ood = batch["ood_score"].astype(float)
        rates = decision_rates(batch["decision"], (ood > reference["ood_threshold"])[ood.notna()])
        raised = alerts(rates, reference["rates"]) if len(batch) >= MIN_BATCH else []
        rows.append(
            {
                "site": site,
                "day": day,
                "predictions": len(batch),
                **rates,
                "error_rate": float((batch["deferral_reason"] == "error").mean()),
                "psi_p_referable": psi(reference["p_referable"], batch["p_referable_calibrated"]),
                "psi_ood_score": psi(reference["ood_score"], ood),
                "alert": "; ".join(str(a) for a in raised) or None,
            }
        )
    return pd.DataFrame(rows)


def write_reports(store: PredictionStore, reports: pd.DataFrame) -> int:
    """Replace the stored reports of the same (site, day); returns the number written."""
    if reports.empty:
        return 0
    now = datetime.now(UTC)
    rows: list[dict[str, Any]] = [
        {str(k): (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()}
        | {"created_at": now}
        for r in reports.to_dict("records")
    ]
    with store.engine.begin() as connection:
        for row in rows:
            connection.execute(
                delete(drift_reports).where(
                    (drift_reports.c.site == row["site"]) & (drift_reports.c.day == row["day"])
                )
            )
        connection.execute(insert(drift_reports), rows)
        for row in rows:
            if row["alert"]:
                store.audit_with(
                    connection,
                    "drift_alert",
                    {"site": row["site"], "day": row["day"], "alert": row["alert"]},
                )
    return len(rows)


def report(store: PredictionStore, reference: dict[str, Any]) -> pd.DataFrame:
    with store.engine.connect() as connection:
        records = pd.DataFrame(connection.execute(select(predictions)).mappings().all())
    reports = batch_reports(records, reference)
    write_reports(store, reports)
    return reports


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    ref = commands.add_parser("reference", help="build reference.json from evaluation outputs")
    ref.add_argument("--manifest", type=Path, required=True)
    ref.add_argument("--grader", type=Path, required=True)
    ref.add_argument("--quality", type=Path, required=True)
    ref.add_argument("--image-stats", type=Path, required=True)
    ref.add_argument("--out", type=Path, required=True)
    rep = commands.add_parser("report", help="drift report of the stored predictions")
    rep.add_argument("--reference", type=Path, required=True)
    rep.add_argument(
        "--database-url", default=os.environ.get("DATABASE_URL", "sqlite:///outputs/retinaguard.db")
    )
    args = parser.parse_args(argv)

    if args.command == "reference":
        from retinaguard.monitoring.simulate import scored_images

        manifest = pd.read_parquet(args.manifest)
        scored, _, thresholds = scored_images(
            manifest, args.grader, args.quality, pd.read_parquet(args.image_stats)
        )
        calib = manifest[(manifest["split"] == "calib") & manifest["processed_path"].notna()]
        assert thresholds.ood_threshold is not None
        reference = build_reference(
            calib[["image_id"]].merge(scored, on="image_id"), thresholds.ood_threshold
        )
        args.out.write_text(json.dumps(reference), encoding="utf-8")
        print(f"reference of {reference['images']} images: {reference['rates']}")
        return
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    reports = report(PredictionStore(args.database_url), reference)
    pd.set_option("display.width", 220)
    print(reports.round(3).to_string(index=False) if not reports.empty else "no predictions")


if __name__ == "__main__":
    main()
