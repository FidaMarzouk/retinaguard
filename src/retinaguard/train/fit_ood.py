"""Fit the Mahalanobis out-of-distribution detector on grader embeddings and report it per site.

    python -m retinaguard.train.fit_ood --manifest manifest.parquet \\
        --predictions-dir predictions/grader-convnext_tiny-no_lmn \\
        --policy-dir predictions/grader-convnext_tiny-no_lmn/policy

The grade-conditional Gaussians are fitted on the ``val`` embeddings (the prediction runs store
embeddings for the evaluation splits only), the threshold is the 99th percentile of ``calib``
scores; both splits leave out EyeQ rejects. Writes ``ood.npz`` and the threshold into
``policy.json`` (from ``calibrate``) in ``--policy-dir``, and ``ood.csv``: per set, the share
flagged (95% cluster bootstrap CI), how well the score separates the site from the EyePACS test
images, and the grader's AUROC and false-negative rate on the kept and flagged images, using the
calibrated scores and thresholds of ``policy.json``.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from retinaguard.models.metrics import auroc, bootstrap_ci
from retinaguard.models.ood import PERCENTILE, Mahalanobis, ood_threshold
from retinaguard.train.calibrate import fitting_rows
from retinaguard.train.evaluate import evaluation_sets

MIN_FLAGGED = 20  # fewer flagged images than this: their AUROC is not reported


def _share(flags: npt.NDArray[np.generic], _: npt.NDArray[np.generic]) -> float:
    return float(np.mean(flags.astype(bool)))


def ood_report(
    sets: dict[str, pd.DataFrame], threshold: float, lambda_low: float, n_resamples: int = 2000
) -> pd.DataFrame:
    """Rows joined with ``ood_score`` and calibrated ``p_referable``, one row of metrics per set."""
    reference = sets.get("eyepacs_test")
    table = {}
    for name, rows in sets.items():
        flagged = (rows["ood_score"] > threshold).to_numpy()
        referable = (rows["dr_grade"].astype(int) >= 2).to_numpy()
        p = rows["p_referable"].to_numpy()
        low, high = bootstrap_ci(
            _share,
            flagged,
            flagged,
            clusters=rows["split_group"].to_numpy(),
            n_resamples=n_resamples,
        )
        result: dict[str, float | int] = {
            "images": len(rows),
            "flagged": float(flagged.mean()),
            "flagged_low": low,
            "flagged_high": high,
            "median_score": float(rows["ood_score"].median()),
            "auroc_kept": auroc(referable[~flagged], p[~flagged]),
            "auroc_flagged": (
                auroc(referable[flagged], p[flagged]) if flagged.sum() >= MIN_FLAGGED else np.nan
            ),
            "fnr_kept": (
                float(np.mean(p[~flagged & referable] < lambda_low))
                if (~flagged & referable).any()
                else np.nan
            ),
        }
        if reference is not None and name != "eyepacs_test" and not name.startswith("eyepacs"):
            site = np.r_[np.zeros(len(reference)), np.ones(len(rows))]
            scores = np.r_[reference["ood_score"].to_numpy(), rows["ood_score"].to_numpy()]
            result["site_auroc"] = auroc(site, scores)
        table[name] = result
    return pd.DataFrame.from_dict(table, orient="index")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--percentile", type=float, default=PERCENTILE)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args(argv)

    manifest = pd.read_parquet(args.manifest)
    index = pd.read_parquet(args.predictions_dir / "predictions.parquet")["image_id"]
    embeddings = np.load(args.predictions_dir / "embeddings.npy")
    position = pd.Series(np.arange(len(index)), index=index)

    def features(rows: pd.DataFrame) -> npt.NDArray[np.float64]:
        return np.asarray(embeddings[position.loc[rows["image_id"]].to_numpy()], np.float64)

    val = fitting_rows(manifest, "val")
    detector = Mahalanobis.fit(features(val), val["dr_grade"].astype(int))
    calib = fitting_rows(manifest, "calib")
    threshold = ood_threshold(detector.score(features(calib)), args.percentile)

    policy_path = args.policy_dir / "policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    calibrated = pd.read_parquet(args.policy_dir / "predictions.parquet")[
        ["image_id", "p_referable"]
    ]
    sets = {"eyepacs_calib": calib} | {
        name: rows
        for name, rows in evaluation_sets(manifest).items()
        if not rows.empty and not name.endswith("_all")
    }
    scored = {
        name: rows.assign(ood_score=detector.score(features(rows))).merge(
            calibrated, on="image_id", validate="one_to_one"
        )
        for name, rows in sets.items()
    }
    table = ood_report(scored, threshold, policy["lambda_low"], args.resamples)

    detector.save(args.policy_dir / "ood.npz")
    policy |= {"ood_threshold": threshold, "ood_percentile": args.percentile}
    policy_path.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    table.to_csv(args.policy_dir / "ood.csv")
    pd.set_option("display.width", 200)
    print(f"Mahalanobis threshold {threshold:.3f} ({args.percentile:g}th percentile of calib)")
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
