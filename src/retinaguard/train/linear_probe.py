"""Linear probe on frozen embeddings, with the grader's two outputs.

    python -m retinaguard.train.linear_probe --manifest manifest.parquet \\
        --embeddings embeddings/dinov2_vitb --out-dir data/interim/predictions/probe-dinov2_vitb

Two logistic regressions on standardised embeddings, fitted on the EyePACS ``train`` split:
multinomial over the ICDR grade and binary referable DR. The inverse regularisation ``C`` is
chosen on ``val`` by referable AUROC; both splits leave out EyeQ rejects, as grader training
does, and external sites are never used. Writes ``predictions.parquet`` and ``embeddings.npy``
in the format of ``predict`` (so ``evaluate`` applies unchanged), and a run folder for
``import_runs``.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from retinaguard.data.datasets import select_rows
from retinaguard.models.metrics import auroc, quadratic_kappa
from retinaguard.tracking import RunLogger, default_output_dir
from retinaguard.train.predict import PROBABILITY_COLUMNS, prediction_rows

C_GRID = (0.001, 0.01, 0.1, 1.0)
MAX_ITER = 2000


def fit_probe(
    features: npt.NDArray[np.floating], grades: npt.NDArray[np.integer], c: float
) -> tuple[Pipeline, Pipeline]:
    """Grade and referable logistic regressions on the same standardised features."""
    grade_model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=MAX_ITER))
    referable_model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=MAX_ITER))
    grade_model.fit(features, grades)
    referable_model.fit(features, grades >= 2)
    return grade_model, referable_model


def probe_predictions(
    grade_model: Pipeline,
    referable_model: Pipeline,
    features: npt.NDArray[np.floating],
    image_ids: pd.Series,
) -> pd.DataFrame:
    """Rows of ``predict`` format: ``image_id``, ``p_grade_0..4``, ``p_referable``."""
    grade_probs = np.zeros((len(features), len(PROBABILITY_COLUMNS)))
    grade_probs[:, grade_model.classes_] = grade_model.predict_proba(features)
    predictions = pd.DataFrame(grade_probs, columns=PROBABILITY_COLUMNS)
    predictions.insert(0, "image_id", image_ids.to_numpy())
    predictions["p_referable"] = referable_model.predict_proba(features)[:, 1]
    return predictions


def _features(
    embeddings: npt.NDArray[np.float16], position: pd.Series, image_ids: pd.Series
) -> npt.NDArray[np.float32]:
    rows = position.loc[image_ids].to_numpy()
    return np.asarray(embeddings[rows], dtype=np.float32)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True, help="extract_embeddings output")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, default=default_output_dir() / "runs")
    args = parser.parse_args(argv)

    manifest = pd.read_parquet(args.manifest)
    embeddings = np.load(args.embeddings / "embeddings.npy")
    index = pd.read_parquet(args.embeddings / "index.parquet")["image_id"]
    position = pd.Series(np.arange(len(index)), index=index)
    meta = json.loads((args.embeddings / "meta.json").read_text(encoding="utf-8"))

    train, val = (select_rows(manifest, [split]) for split in ("train", "val"))
    x_train = _features(embeddings, position, train["image_id"])
    x_val = _features(embeddings, position, val["image_id"])
    y_train = train["dr_grade"].astype(int).to_numpy()
    y_val = val["dr_grade"].astype(int).to_numpy()

    name = args.out_dir.name
    logger = RunLogger(args.runs_dir / name)
    logger.log_params({f"embeddings.{k}": v for k, v in meta.items()} | {"c_grid": list(C_GRID)})
    logger.set_tags({"task": "linear_probe", "model": meta["model"]})
    best: tuple[float, float, tuple[Pipeline, Pipeline]] | None = None
    for step, c in enumerate(C_GRID):
        models = fit_probe(x_train, y_train, c)
        val_auroc = auroc(y_val >= 2, models[1].predict_proba(x_val)[:, 1])
        val_kappa = quadratic_kappa(y_val, models[0].predict(x_val))
        logger.log_metrics(
            {"c": c, "val_referable_auroc": val_auroc, "val_quadratic_kappa": val_kappa}, step
        )
        print(f"C={c:g}: val referable AUROC {val_auroc:.4f}, QWK {val_kappa:.3f}", flush=True)
        if best is None or val_auroc > best[0]:
            best = (val_auroc, c, models)
    assert best is not None
    logger.log_params({"c": best[1]})

    rows = prediction_rows(manifest)
    x = _features(embeddings, position, rows["image_id"])
    predictions = probe_predictions(*best[2], x, rows["image_id"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(args.out_dir / "predictions.parquet", index=False)
    np.save(args.out_dir / "embeddings.npy", x.astype(np.float16))
    print(f"C={best[1]:g} chosen; {len(predictions):,} predictions written to {args.out_dir}")


if __name__ == "__main__":
    main()
