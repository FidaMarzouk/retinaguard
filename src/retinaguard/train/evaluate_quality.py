"""Quality model metrics per evaluation set, with bootstrap 95% confidence intervals.

    python -m retinaguard.train.evaluate_quality --manifest manifest.parquet \\
        --predictions runs/quality-efficientnet_b0/predictions.parquet --out quality_metrics.csv

- ``eyepacs_val`` and ``eyepacs_test``: EyeQ-labelled images of the EyePACS splits; macro-F1,
  accuracy and per-class recall.
- ``ddr``: DDR images its graders marked ungradable carry the ``reject`` label, so the reject
  recall there is measured on a site never seen in training; ``reject_rate_graded`` is the share
  of gradable DDR images the gate would send back.
- ``aptos`` and ``idrid`` have no quality labels: only the share predicted ``reject``.

Intervals resample ``split_group`` clusters, as in ``evaluate``.
"""

import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from retinaguard.data.schemas import EXTERNAL_SOURCES
from retinaguard.models.metrics import bootstrap_ci
from retinaguard.models.quality import (
    QUALITY_CLASSES,
    QUALITY_COLUMNS,
    REJECT,
    confusion,
    macro_f1,
    quality_metrics,
)
from retinaguard.train.evaluate import Metric

CODES = {name: code for code, name in enumerate(QUALITY_CLASSES)}


def _macro_f1(labels: npt.NDArray[np.generic], predicted: npt.NDArray[np.generic]) -> float:
    return macro_f1(labels, predicted)


def _reject_recall(labels: npt.NDArray[np.generic], predicted: npt.NDArray[np.generic]) -> float:
    rejects = labels == REJECT
    return float(np.mean(predicted[rejects] == REJECT)) if rejects.any() else float("nan")


def _share(flags: npt.NDArray[np.generic], _: npt.NDArray[np.generic]) -> float:
    return float(np.mean(flags.astype(bool)))


def _with_ci(
    result: dict[str, float | int],
    name: str,
    metric: Metric,
    a: npt.NDArray[np.generic],
    b: npt.NDArray[np.generic],
    clusters: npt.NDArray[np.generic],
    n_resamples: int,
) -> None:
    result[name] = metric(a, b)
    result[f"{name}_low"], result[f"{name}_high"] = bootstrap_ci(
        metric, a, b, clusters=clusters, n_resamples=n_resamples
    )


def evaluate_quality(
    manifest: pd.DataFrame, predictions: pd.DataFrame, n_resamples: int = 2000
) -> pd.DataFrame:
    joined = manifest.merge(predictions, on="image_id", how="inner", validate="one_to_one")
    predicted = joined[QUALITY_COLUMNS].to_numpy().argmax(1)
    joined = joined.assign(_predicted=predicted)
    table: dict[str, dict[str, float | int]] = {}

    eyepacs = joined[(joined["source"] == "eyepacs") & joined["quality_label"].notna()]
    for name, split in (("eyepacs_val", "val"), ("eyepacs_test", "test_internal")):
        rows = eyepacs[eyepacs["split"] == split]
        if rows.empty:
            continue
        labels = rows["quality_label"].map(CODES).to_numpy()
        result = quality_metrics(labels, rows[QUALITY_COLUMNS].to_numpy())
        clusters = rows["split_group"].to_numpy()
        guess = rows["_predicted"].to_numpy()
        _with_ci(result, "macro_f1", _macro_f1, labels, guess, clusters, n_resamples)
        _with_ci(result, "recall_reject", _reject_recall, labels, guess, clusters, n_resamples)
        table[name] = result

    for source in EXTERNAL_SOURCES:
        rows = joined[(joined["source"] == source) & (joined["split"] == "test_external")]
        if rows.empty:
            continue
        clusters = rows["split_group"].to_numpy()
        rejected = (rows["_predicted"] == REJECT).to_numpy()
        result = {"images": len(rows)}
        _with_ci(result, "reject_rate", _share, rejected, rejected, clusters, n_resamples)
        ungradable = rows["quality_label"].eq("reject").to_numpy()
        if ungradable.any():
            result["ungradable"] = int(ungradable.sum())
            result["recall_reject"] = float(rejected[ungradable].mean())
            graded = rejected[~ungradable]
            _with_ci(
                result,
                "reject_rate_graded",
                _share,
                graded,
                graded,
                clusters[~ungradable],
                n_resamples,
            )
        table[source] = result
    return pd.DataFrame.from_dict(table, orient="index")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args(argv)

    manifest = pd.read_parquet(args.manifest)
    predictions = pd.read_parquet(args.predictions)
    table = evaluate_quality(manifest, predictions, args.resamples)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out)
    pd.set_option("display.width", 200)
    print(table.round(3).to_string())
    test = manifest[(manifest["split"] == "test_internal") & manifest["quality_label"].notna()]
    rows = test.merge(predictions, on="image_id")
    if not rows.empty:
        print(confusion(rows["quality_label"].map(CODES), rows[QUALITY_COLUMNS]).to_string())


if __name__ == "__main__":
    main()
