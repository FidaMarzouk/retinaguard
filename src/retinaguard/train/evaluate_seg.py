"""Segmentation metrics per run, site and lesion, with paired comparisons and site gaps.

    python -m retinaguard.train.evaluate_seg outputs/runs/seg-pooled outputs/runs/seg-ddr_only \\
        outputs/runs/seg-idrid_only --reference seg-pooled --out-dir outputs/seg_eval

Reads each run's ``stats.npz`` and ``config.yaml`` (from ``train_seg``). Sets: each site's
``test`` split and, for a site the run did not train on, all of that site's images
(``<site>_all``). Dice and AUPR are computed over all pixels of a set, with percentile bootstrap
95% CIs over images. Writes:

- ``metrics.csv``: one row per run, set and lesion;
- ``sites.csv``: per run and lesion, the worst site's test AUPR and Dice and the gap between
  sites (best minus worst);
- ``comparison.csv``: each run minus the reference on the same test images, with paired CIs.
"""

import argparse
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from omegaconf import OmegaConf

from retinaguard.data.segmentation import LESIONS
from retinaguard.models.segmentation import PixelStats, aupr, dice

MetricFn = Callable[[PixelStats], npt.NDArray[np.float64]]
METRICS: dict[str, MetricFn] = {"aupr": aupr, "dice": dice}


def load_run(run_dir: Path) -> tuple[pd.DataFrame, PixelStats, list[str]]:
    """Rows (``image_id``, ``site``, ``split``), their counts and the run's training sites."""
    with np.load(run_dir / "stats.npz", allow_pickle=False) as data:
        rows = pd.DataFrame({k: data[k] for k in ("image_id", "site", "split")})
        stats = PixelStats(
            data["tp"], data["fp"], data["fn"], data["positive_hist"], data["negative_hist"]
        )
    config = OmegaConf.load(run_dir / "config.yaml")
    return rows, stats, list(config.data.train_sites)


def evaluation_sets(rows: pd.DataFrame, train_sites: list[str]) -> dict[str, npt.NDArray[np.intp]]:
    sets = {}
    for site in sorted(rows["site"].unique()):
        at_site = rows["site"].to_numpy() == site
        sets[site] = np.flatnonzero(at_site & (rows["split"].to_numpy() == "test"))
        if site not in train_sites:
            sets[f"{site}_all"] = np.flatnonzero(at_site)
    return sets


def bootstrap(
    values: Callable[[npt.NDArray[np.intp]], npt.NDArray[np.float64]],
    n_images: int,
    n_resamples: int,
    seed: int = 0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Percentile 95% interval of ``values(index)`` (one value per lesion) over image resamples."""
    rng = np.random.default_rng(seed)
    draws = np.stack([values(rng.integers(0, n_images, n_images)) for _ in range(n_resamples)])
    low, high = np.nanquantile(draws, [0.025, 0.975], axis=0)
    return low, high


def _on(fn: MetricFn, stats: PixelStats, index: npt.NDArray[np.intp]) -> npt.NDArray[np.float64]:
    return fn(stats.take(index))


def _difference(
    fn: MetricFn, a: PixelStats, b: PixelStats, index: npt.NDArray[np.intp]
) -> npt.NDArray[np.float64]:
    return fn(b.take(index)) - fn(a.take(index))


def run_metrics(
    name: str, rows: pd.DataFrame, stats: PixelStats, train_sites: list[str], n_resamples: int
) -> pd.DataFrame:
    records = []
    for set_name, index in evaluation_sets(rows, train_sites).items():
        if not len(index):
            continue
        part = stats.take(index)
        for metric, fn in METRICS.items():
            point = fn(part)
            low, high = bootstrap(partial(_on, fn, part), len(index), n_resamples)
            for k, lesion in enumerate(LESIONS):
                records.append(
                    {
                        "run": name,
                        "set": set_name,
                        "images": len(index),
                        "lesion": lesion,
                        "metric": metric,
                        "value": point[k],
                        "low": low[k],
                        "high": high[k],
                    }
                )
    return pd.DataFrame(records)


def site_gaps(metrics: pd.DataFrame) -> pd.DataFrame:
    """Worst site and best-minus-worst gap of the test sets, per run, lesion and metric."""
    test = metrics[~metrics["set"].str.endswith("_all")]
    grouped = test.groupby(["run", "lesion", "metric"])["value"]
    table = pd.DataFrame({"worst": grouped.min(), "best": grouped.max()})
    table["worst_site"] = test.loc[grouped.idxmin(), "set"].to_numpy()
    table["gap"] = table["best"] - table["worst"]
    return table.reset_index()


def paired_comparison(
    reference: tuple[pd.DataFrame, PixelStats],
    other: tuple[pd.DataFrame, PixelStats],
    name: str,
    n_resamples: int,
) -> pd.DataFrame:
    """``other`` minus ``reference`` on the test images both segmented, per site."""
    (ref_rows, ref_stats), (rows, stats) = reference, other
    ref_position = pd.Series(np.arange(len(ref_rows)), index=ref_rows["image_id"])
    position = pd.Series(np.arange(len(rows)), index=rows["image_id"])
    records = []
    for site in sorted(ref_rows["site"].unique()):
        test = ref_rows[(ref_rows["site"] == site) & (ref_rows["split"] == "test")]
        ids = test["image_id"][test["image_id"].isin(position.index)]
        if ids.empty:
            continue
        a = ref_stats.take(ref_position.loc[ids].to_numpy())
        b = stats.take(position.loc[ids].to_numpy())
        for metric, fn in METRICS.items():
            point = fn(b) - fn(a)
            low, high = bootstrap(partial(_difference, fn, a, b), len(ids), n_resamples)
            for k, lesion in enumerate(LESIONS):
                records.append(
                    {
                        "run": name,
                        "site": site,
                        "images": len(ids),
                        "lesion": lesion,
                        "metric": metric,
                        "difference": point[k],
                        "low": low[k],
                        "high": high[k],
                    }
                )
    return pd.DataFrame(records)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--reference", help="run name the others are compared with")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args(argv)

    loaded = {run.name: load_run(run) for run in args.runs}
    metrics = pd.concat(
        [
            run_metrics(name, rows, stats, sites, args.resamples)
            for name, (rows, stats, sites) in loaded.items()
        ],
        ignore_index=True,
    )
    gaps = site_gaps(metrics)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.out_dir / "metrics.csv", index=False)
    gaps.to_csv(args.out_dir / "sites.csv", index=False)
    pd.set_option("display.width", 200)
    wide = metrics.pivot_table(index=["run", "set"], columns=["metric", "lesion"], values="value")
    print(wide.round(3).to_string())
    if args.reference:
        ref_rows, ref_stats, _ = loaded[args.reference]
        comparison = pd.concat(
            [
                paired_comparison((ref_rows, ref_stats), (rows, stats), name, args.resamples)
                for name, (rows, stats, _) in loaded.items()
                if name != args.reference
            ],
            ignore_index=True,
        )
        comparison.to_csv(args.out_dir / "comparison.csv", index=False)
        print(comparison[comparison["metric"] == "aupr"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
