"""Simulated deployment: each test site replayed as daily batches, drift against performance.

    python -m retinaguard.monitoring.simulate --manifest data/manifests/manifest.parquet \\
        --grader data/interim/predictions/grader-convnext_tiny-no_lmn \\
        --quality data/interim/predictions/quality-efficientnet_b0 \\
        --image-stats data/interim/image_stats/image_stats.parquet --out batches.csv

Uses the stored outputs of the deployed models (calibrated referable probability, embedding,
quality prediction) rather than re-running them: the ONNX service reproduces them to 1e-5.
The reference is the EyePACS ``calib`` split. Batches: each site's primary test set in shuffled
batches of ``--batch-size``, plus ``gradual``: EyePACS test batches in which a growing share
(0 to 100%) of photos comes from DDR. For every batch: the drift signals of ``drift``, the alert
rule, and the real performance (referable AUROC, false-negative rate of the full chain,
calibration error), which a deployment would only learn from reviewed cases. Every batch also
gets Evidently's share of drifted columns (prediction, OOD score, image statistics, embedding
principal components); ``--html-dir`` writes an Evidently report per scenario.
"""

import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from retinaguard.data.image_stats import STAT_NAMES
from retinaguard.models.calibration import expected_calibration_error
from retinaguard.models.metrics import auroc
from retinaguard.models.ood import Mahalanobis
from retinaguard.models.quality import QUALITY_COLUMNS, REJECT
from retinaguard.monitoring.drift import alerts, decision_rates, mean_shift, mmd, psi
from retinaguard.monitoring.evidently_reports import (
    EMBEDDING_COMPONENTS,
    embedding_components,
    evidently_drift,
    save_html,
)
from retinaguard.serving.policy import Thresholds, decide
from retinaguard.train.calibrate import fitting_rows
from retinaguard.train.evaluate import evaluation_sets

SITES = ("eyepacs_test", "aptos", "idrid", "ddr")
BATCH_SIZE = 250
DRIFT_STATS = ("brightness", "contrast", "sharpness", "dark_fraction", "bright_fraction")
PC_COLUMNS = [f"embedding_pc{i + 1}" for i in range(EMBEDDING_COMPONENTS)]
EVIDENTLY_NUMERICAL = ["p_referable", "ood_score", *DRIFT_STATS, *PC_COLUMNS]


def scored_images(
    manifest: pd.DataFrame, grader_dir: Path, quality_dir: Path, image_stats: pd.DataFrame
) -> tuple[pd.DataFrame, npt.NDArray[np.float64], Thresholds]:
    """Every predicted image with its full-chain decision, and the embeddings (row ``row``)."""
    policy_dir = grader_dir / "policy"
    thresholds = Thresholds.load(policy_dir / "policy.json")
    index = pd.read_parquet(grader_dir / "predictions.parquet")["image_id"]
    embeddings = np.load(grader_dir / "embeddings.npy").astype(np.float64)
    frame = pd.DataFrame({"image_id": index, "row": np.arange(len(index))})
    frame["ood_score"] = Mahalanobis.load(policy_dir / "ood.npz").score(embeddings)
    calibrated = pd.read_parquet(policy_dir / "predictions.parquet")[["image_id", "p_referable"]]
    quality = pd.read_parquet(quality_dir / "predictions.parquet")
    quality["quality_pred"] = np.where(
        quality[QUALITY_COLUMNS].to_numpy().argmax(1) == REJECT, "reject", "ok"
    )
    frame = (
        frame.merge(calibrated, on="image_id")
        .merge(quality[["image_id", "quality_pred"]], on="image_id")
        .merge(image_stats[["image_id", *STAT_NAMES]], on="image_id", how="left")
    )
    frame["decision"] = [
        decide(p, thresholds, quality_label=q, ood_score=o).decision.value
        for p, q, o in zip(
            frame["p_referable"], frame["quality_pred"], frame["ood_score"], strict=True
        )
    ]
    assert thresholds.ood_threshold is not None
    frame["ood_flag"] = frame["ood_score"] > thresholds.ood_threshold
    return frame, embeddings, thresholds


def batch_metrics(
    batch: pd.DataFrame,
    reference: pd.DataFrame,
    embeddings: npt.NDArray[np.float64],
    reference_rates: dict[str, float],
    seed: int = 0,
) -> dict[str, float | int | str]:
    """Drift signals, alerts and performance of one batch (rows joined with labels)."""
    rates = decision_rates(batch["decision"], batch["ood_flag"])
    evidently = evidently_drift(
        reference, batch, numerical=EVIDENTLY_NUMERICAL, categorical=["decision"]
    )
    mmd2, mmd_p = mmd(embeddings[reference["row"]], embeddings[batch["row"]], seed=seed)
    shift = mean_shift(reference[list(DRIFT_STATS)], batch[list(DRIFT_STATS)])
    referable = (batch["dr_grade"].astype(int) >= 2).to_numpy()
    graded = (batch["decision"] != "RETAKE").to_numpy()
    cleared = (batch["decision"] == "NO_REFER").to_numpy()
    raised = alerts(rates, reference_rates)
    return {
        "images": len(batch),
        "psi_p_referable": psi(reference["p_referable"], batch["p_referable"]),
        "mmd2": mmd2,
        "mmd_p": mmd_p,
        "max_image_shift": float(np.nanmax(np.abs(list(shift.values())))),
        "evidently_drift_share": evidently.drift_share,
        "evidently_drifted_columns": evidently.drifted_columns,
        **{f"shift_{k}": v for k, v in shift.items()},
        **rates,
        "alert": "; ".join(str(a) for a in raised),
        "auroc": auroc(referable, batch["p_referable"]),
        "fnr": float(cleared[referable & graded].mean()) if (referable & graded).any() else np.nan,
        "ece": expected_calibration_error(referable, batch["p_referable"]),
        "referable_share": float(referable.mean()),
    }


def batches(
    sets: dict[str, pd.DataFrame], batch_size: int, seed: int
) -> list[tuple[str, int, pd.DataFrame]]:
    """(scenario, day, rows): each site in shuffled batches, then the gradual EyePACS-to-DDR mix."""
    rng = np.random.default_rng(seed)
    out = []
    for site in SITES:
        rows = sets[site].sample(frac=1, random_state=int(rng.integers(2**31)))
        for day, start in enumerate(range(0, len(rows) - batch_size + 1, batch_size)):
            out.append((site, day, rows.iloc[start : start + batch_size]))
    eyepacs, ddr = sets["eyepacs_test"], sets["ddr"]
    for day, share in enumerate(np.linspace(0, 1, 11)):
        n_ddr = round(share * batch_size)
        mixed = pd.concat(
            [
                eyepacs.sample(batch_size - n_ddr, random_state=int(rng.integers(2**31))),
                ddr.sample(n_ddr, random_state=int(rng.integers(2**31))),
            ]
        )
        out.append(("gradual", day, mixed))
    return out


def simulate(
    manifest: pd.DataFrame,
    grader_dir: Path,
    quality_dir: Path,
    image_stats: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
    seed: int = 0,
    html_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    frame, embeddings, _ = scored_images(manifest, grader_dir, quality_dir, image_stats)
    calib_rows = frame.loc[
        frame["image_id"].isin(fitting_rows(manifest, "calib")["image_id"]), "row"
    ].to_numpy()
    _, components = embedding_components(embeddings[calib_rows], embeddings)
    frame = frame.join(components, on="row")
    reference = fitting_rows(manifest, "calib")[["image_id", "dr_grade"]].merge(
        frame, on="image_id"
    )
    # the reference rates include quality rejects, as incoming traffic would
    calib_all = manifest[(manifest["split"] == "calib") & manifest["processed_path"].notna()]
    reference_rates = decision_rates(
        *(lambda r: (r["decision"], r["ood_flag"]))(
            calib_all[["image_id"]].merge(frame, on="image_id")
        )
    )
    sets = {
        name: rows[["image_id", "dr_grade"]].merge(frame, on="image_id")
        for name, rows in evaluation_sets(manifest).items()
        if name in SITES
    }
    rows = []
    by_scenario: dict[str, list[pd.DataFrame]] = {}
    for i, (scenario, day, batch) in enumerate(batches(sets, batch_size, seed)):
        metrics = batch_metrics(batch, reference, embeddings, reference_rates, seed=i)
        rows.append({"scenario": scenario, "day": day} | metrics)
        if scenario == "gradual":
            rows[-1]["ddr_share"] = day / 10
        by_scenario.setdefault(scenario, []).append(batch)
    if html_dir is not None:
        for scenario, parts in by_scenario.items():
            current = parts[-1] if scenario == "gradual" else pd.concat(parts)  # all-DDR end
            report = evidently_drift(
                reference, current, numerical=EVIDENTLY_NUMERICAL, categorical=["decision"]
            )
            save_html(report, html_dir / f"{scenario}.html")
    reference_rates |= {
        "auroc": auroc(reference["dr_grade"].astype(int) >= 2, reference["p_referable"]),
        "images": float(len(calib_all)),
    }
    return pd.DataFrame(rows), reference_rates


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--grader", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--image-stats", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--html-dir", type=Path, help="write an Evidently report per scenario")
    args = parser.parse_args(argv)

    table, reference = simulate(
        pd.read_parquet(args.manifest),
        args.grader,
        args.quality,
        pd.read_parquet(args.image_stats),
        args.batch_size,
        html_dir=args.html_dir,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print("reference:", {k: round(v, 3) for k, v in reference.items()})
    columns = [
        "psi_p_referable",
        "mmd2",
        "evidently_drift_share",
        "max_image_shift",
        "defer_rate",
        "ood_rate",
        "retake_rate",
        "auroc",
        "fnr",
        "ece",
    ]
    summary = table.groupby("scenario", sort=False)[columns].mean()
    summary["batches"] = table.groupby("scenario", sort=False).size()
    summary["alerts"] = (
        table.assign(a=table["alert"].astype(bool)).groupby("scenario", sort=False)["a"].sum()
    )
    pd.set_option("display.width", 220)
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
