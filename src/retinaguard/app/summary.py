"""Per-site results the app shows, condensed from the evaluation outputs into one JSON file.

    python -m retinaguard.app.summary --manifest data/manifests/manifest.parquet \\
        --grader data/interim/predictions/grader-convnext_tiny-no_lmn \\
        --quality data/interim/predictions/quality-efficientnet_b0

Reads the grader's ``metrics.csv`` (``evaluate``), its calibrated predictions, policy and
embeddings (``calibrate``, ``fit_ood``) and the quality model's predictions and metrics, runs the
full decision chain (``serving.policy.decide``) on every primary test image, and writes
``site_summary.json`` next to this module: numbers only, so the app needs no dataset.
"""

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from retinaguard.app.views import OUTCOMES, SUMMARY_PATH
from retinaguard.models.ood import Mahalanobis
from retinaguard.models.quality import QUALITY_COLUMNS, REJECT
from retinaguard.serving.policy import Decision, Thresholds, decide
from retinaguard.train.evaluate import evaluation_sets

SITES = {
    "eyepacs_test": "EyePACS test (US)",
    "aptos": "APTOS 2019 (India)",
    "idrid": "IDRiD (India)",
    "ddr": "DDR (China)",
}


def outcome_label(decision: Decision, reason: str | None) -> str:
    if decision != Decision.DEFER_TO_HUMAN:
        return decision.value
    return f"DEFER ({(reason or 'error').replace('_', ' ')})"


def decision_mix(rows: pd.DataFrame, thresholds: Thresholds) -> dict[str, float]:
    """Shares of each outcome and the error rates of the chain, on rows with ``p_referable``,
    ``quality_pred`` and ``ood_score``."""
    outcomes = [
        decide(p, thresholds, quality_label=q, ood_score=o)
        for p, q, o in zip(
            rows["p_referable"], rows["quality_pred"], rows["ood_score"], strict=True
        )
    ]
    labels = pd.Series([outcome_label(o.decision, o.reason) for o in outcomes])
    shares = labels.value_counts(normalize=True).reindex(OUTCOMES, fill_value=0.0)
    referable = (rows["dr_grade"].astype(int) >= 2).to_numpy()
    graded = (labels != "RETAKE").to_numpy()
    cleared = (labels == "NO_REFER").to_numpy()
    return {str(k): float(v) for k, v in shares.items()} | {
        "fnr": float(cleared[referable & graded].mean()),
        "human_workload": float((labels.str.startswith("DEFER") | (labels == "RETAKE")).mean()),
    }


def _rows(path: Path) -> dict[str, dict[str, Any]]:
    """A metrics CSV (first column: set name) as ``{set: {column: value}}``."""
    frame = pd.read_csv(path, index_col=0)
    return {str(name): {str(k): v for k, v in row.items()} for name, row in frame.iterrows()}


def build(manifest: pd.DataFrame, grader_dir: Path, quality_dir: Path) -> dict[str, Any]:
    policy_dir = grader_dir / "policy"
    metrics = _rows(grader_dir / "metrics.csv")
    calibration = _rows(policy_dir / "calibration.csv")
    ood = _rows(policy_dir / "ood.csv")
    thresholds = Thresholds.load(policy_dir / "policy.json")
    calibrated = pd.read_parquet(policy_dir / "predictions.parquet")[["image_id", "p_referable"]]
    index = pd.read_parquet(grader_dir / "predictions.parquet")["image_id"]
    embeddings = np.load(grader_dir / "embeddings.npy").astype(np.float64)
    scores = Mahalanobis.load(policy_dir / "ood.npz").score(embeddings)
    ood_scores = pd.Series(scores, index=index, name="ood_score")
    quality = pd.read_parquet(quality_dir / "predictions.parquet")
    quality["quality_pred"] = np.where(
        quality[QUALITY_COLUMNS].to_numpy().argmax(1) == REJECT, "reject", "ok"
    )
    quality_metrics = _rows(quality_dir / "metrics.csv")

    sets = evaluation_sets(manifest)
    sites = []
    for name, label in SITES.items():
        rows = (
            sets[name]
            .merge(calibrated, on="image_id")
            .merge(quality[["image_id", "quality_pred"]], on="image_id")
            .join(ood_scores, on="image_id")
        )
        m = metrics[name]
        sites.append(
            {
                "set": name,
                "site": label,
                "images": int(m["images"]),
                "referable": int(m["referable"]),
                "auroc": float(m["auroc"]),
                "auroc_low": float(m["auroc_low"]),
                "auroc_high": float(m["auroc_high"]),
                "quadratic_kappa": float(m["quadratic_kappa"]),
                "referable_ece_raw": float(calibration[name]["referable_ece_before"]),
                "referable_ece_calibrated": float(calibration[name]["referable_ece_after"]),
                "ood_flagged": float(ood[name]["flagged"]),
            }
            | decision_mix(rows, thresholds)
        )
    test = quality_metrics["eyepacs_test"]
    return {
        "generated": date.today().isoformat(),
        "grader_run": grader_dir.name,
        "quality_run": quality_dir.name,
        "thresholds": thresholds.to_dict(),
        "alpha": json.loads((policy_dir / "policy.json").read_text())["alpha"],
        "sites": sites,
        "quality": {
            "macro_f1": float(test["macro_f1"]),
            "macro_f1_low": float(test["macro_f1_low"]),
            "macro_f1_high": float(test["macro_f1_high"]),
            "recall_reject": float(test["recall_reject"]),
            "ddr_ungradable_recall": float(quality_metrics["ddr"]["recall_reject"]),
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--grader", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args(argv)
    summary = build(pd.read_parquet(args.manifest), args.grader, args.quality)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    table = pd.DataFrame(summary["sites"]).set_index("site")
    print(table[["auroc", "fnr", "human_workload", "REFER", "NO_REFER"]].round(3).to_string())


if __name__ == "__main__":
    main()
