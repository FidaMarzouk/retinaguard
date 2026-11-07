"""Grader metrics per evaluation set, with bootstrap 95% confidence intervals.

    python -m retinaguard.train.evaluate --manifest manifest.parquet \\
        --predictions predictions.parquet --out metrics.csv

Sets: the EyePACS internal test split and each external site, every one reported separately
(never pooled). Primary sets count each eye once: images EyeQ rates ``reject`` are left out of
the internal test, and external sites keep one image per duplicate group, dropping groups whose
copies disagree on ``referable`` (``one_image_per_group``). The ``_all`` variants keep every
graded image. Intervals resample ``split_group`` clusters (a patient, or a duplicate group), so
correlated images do not make them too narrow.
"""

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from retinaguard.data.schemas import EXTERNAL_SOURCES
from retinaguard.data.splits import one_image_per_group
from retinaguard.models.grader import N_GRADES
from retinaguard.models.metrics import (
    accuracy,
    auprc,
    auroc,
    bootstrap_ci,
    quadratic_kappa,
    sensitivity_at_specificity,
    specificity_at_sensitivity,
)
from retinaguard.train.predict import PROBABILITY_COLUMNS

Metric = Callable[[npt.NDArray[np.generic], npt.NDArray[np.generic]], float]

# Referable DR (grade >= 2) from the referable head's probability.
REFERABLE_METRICS: dict[str, Metric] = {
    "auroc": auroc,
    "auprc": auprc,
    "sens_at_90_spec": sensitivity_at_specificity,
    "spec_at_95_sens": specificity_at_sensitivity,
}
# 5-class ICDR grade from the grade head's most likely class.
GRADE_METRICS: dict[str, Metric] = {"quadratic_kappa": quadratic_kappa, "accuracy": accuracy}


def _graded(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[rows["dr_grade"].notna() & rows["processed_path"].notna()]


def evaluation_sets(manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Rows counted for each reported set, keyed by set name."""
    internal = _graded(manifest[manifest["split"] == "test_internal"])
    sets = {
        "eyepacs_test": internal[internal["quality_label"].ne("reject").fillna(True)],
        "eyepacs_test_all": internal,
    }
    for source in EXTERNAL_SOURCES:
        site = manifest[(manifest["source"] == source) & (manifest["split"] == "test_external")]
        sets[source] = _graded(one_image_per_group(site))
        sets[f"{source}_all"] = _graded(site)
    return sets


def evaluate_set(
    rows: pd.DataFrame, n_resamples: int = 2000, seed: int = 0
) -> dict[str, float | int]:
    """Point estimates and 95% CIs on rows joined with predictions."""
    grades = rows["dr_grade"].astype(int).to_numpy()
    referable = (grades >= 2).astype(int)
    p_referable = rows["p_referable"].to_numpy()
    predicted = rows[PROBABILITY_COLUMNS].to_numpy().argmax(1)
    clusters = rows["split_group"].to_numpy()
    result: dict[str, float | int] = {
        "images": len(rows),
        "referable": int(referable.sum()),
        "auroc_from_grades": auroc(referable, rows[PROBABILITY_COLUMNS[2:]].sum(axis=1)),
    }
    pairs = [(name, metric, referable, p_referable) for name, metric in REFERABLE_METRICS.items()]
    pairs += [(name, metric, grades, predicted) for name, metric in GRADE_METRICS.items()]
    for name, metric, truth, estimate in pairs:
        result[name] = metric(truth, estimate)
        low, high = bootstrap_ci(
            metric, truth, estimate, clusters=clusters, n_resamples=n_resamples, seed=seed
        )
        result[f"{name}_low"], result[f"{name}_high"] = low, high
    return result


def evaluate(
    manifest: pd.DataFrame, predictions: pd.DataFrame, n_resamples: int = 2000, seed: int = 0
) -> pd.DataFrame:
    """One row of metrics per non-empty evaluation set. Every counted image must have a
    prediction."""
    table = {}
    for name, rows in evaluation_sets(manifest).items():
        if rows.empty:
            continue
        joined = rows.merge(predictions, on="image_id", how="left", validate="one_to_one")
        missing = joined["p_referable"].isna()
        if missing.any():
            raise ValueError(f"{name}: {int(missing.sum())} images have no prediction")
        table[name] = evaluate_set(joined, n_resamples, seed)
    return pd.DataFrame.from_dict(table, orient="index")


def confusion_matrix(rows: pd.DataFrame) -> pd.DataFrame:
    """True grade (rows) against predicted grade (columns), on rows joined with predictions."""
    predicted = rows[PROBABILITY_COLUMNS].to_numpy().argmax(1)
    table = pd.crosstab(
        rows["dr_grade"].astype(int).rename("true"), pd.Series(predicted, name="predicted")
    )
    grades = list(range(N_GRADES))
    return table.reindex(index=grades, columns=grades, fill_value=0)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args(argv)

    table = evaluate(
        pd.read_parquet(args.manifest), pd.read_parquet(args.predictions), args.resamples
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out)
    pd.set_option("display.width", 200)
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
