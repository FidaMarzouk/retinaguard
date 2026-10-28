"""Patient-level, grade-stratified splits.

Images are grouped by patient (or by image when the patient is unknown); groups linked by a
confirmed near-duplicate pair are merged, so the same eye never lands in two splits. Each group is
stratified by its highest DR grade and assigned with a fixed seed, so the split is reproducible.
"""

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

SPLIT_FRACTIONS: Mapping[str, float] = {
    "train": 0.7,
    "val": 0.1,
    "calib": 0.1,
    "test_internal": 0.1,
}
SPLIT_SEED = 20261028
UNGRADED = -1


def split_groups(manifest: pd.DataFrame, duplicate_pairs: Iterable[tuple[str, str]]) -> pd.Series:
    """Group label per image (indexed like ``manifest``): the smallest patient key in its
    component, where patients are linked by near-duplicate image pairs."""
    keys = manifest["patient_id"].fillna(manifest["image_id"])
    key_of = dict(zip(manifest["image_id"], keys, strict=True))
    parent = {key: key for key in keys}

    def root(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for a, b in duplicate_pairs:
        if a not in key_of or b not in key_of:
            continue  # pairs involving other sources are handled where both are present
        ra, rb = root(key_of[a]), root(key_of[b])
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return keys.map(root).rename("split_group")


def assign_splits(
    manifest: pd.DataFrame,
    duplicate_pairs: Iterable[tuple[str, str]] = (),
    fractions: Mapping[str, float] = SPLIT_FRACTIONS,
    seed: int = SPLIT_SEED,
) -> pd.DataFrame:
    """``image_id``, ``split_group`` and ``split`` for every manifest row.

    Within each stratum (a group's highest grade, ungraded groups apart) the groups are shuffled
    and cut at the cumulative fractions, so every split gets its share of every grade.
    """
    if not np.isclose(sum(fractions.values()), 1.0):
        raise ValueError(f"split fractions must sum to 1, got {dict(fractions)}")
    groups = split_groups(manifest, duplicate_pairs)
    grade = manifest["dr_grade"].astype("Int64").fillna(UNGRADED).astype(int)
    stratum = grade.groupby(groups).max()

    rng = np.random.default_rng(seed)
    names = list(fractions)
    cuts = np.cumsum([fractions[name] for name in names])
    split_of: dict[str, str] = {}
    for _, members in stratum.groupby(stratum):
        ordered = rng.permutation(np.sort(members.index.to_numpy()))
        bounds = np.round(cuts * len(ordered)).astype(int)
        for name, part in zip(names, np.split(ordered, bounds[:-1]), strict=True):
            split_of.update(dict.fromkeys(part.tolist(), name))
    return pd.DataFrame(
        {
            "image_id": manifest["image_id"],
            "split_group": groups,
            "split": groups.map(split_of),
        }
    )


def external_splits(
    manifest: pd.DataFrame, duplicate_pairs: Iterable[tuple[str, str]] = ()
) -> pd.DataFrame:
    """Like ``assign_splits`` for an external test source: every image is ``test_external``.

    Duplicate groups are still recorded, so evaluation can count each eye once.
    """
    groups = split_groups(manifest, duplicate_pairs)
    return pd.DataFrame(
        {"image_id": manifest["image_id"], "split_group": groups, "split": "test_external"}
    )


def conflicting_groups(manifest: pd.DataFrame, groups: pd.Series, column: str) -> list[str]:
    """Groups whose images carry more than one distinct non-null value of ``column``, e.g.
    duplicates of one eye graded differently."""
    distinct = manifest[column].groupby(groups.to_numpy()).nunique()
    return sorted(distinct[distinct > 1].index)
