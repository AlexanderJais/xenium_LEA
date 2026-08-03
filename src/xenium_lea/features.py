"""
features.py
-----------
Read the feature list from whichever form a Xenium bundle happens to ship.

10x has changed both the container and the vocabulary across Xenium Ranger
generations, so the audit reads whichever is present and normalises what it
finds rather than assuming one layout:

    cell_feature_matrix/features.tsv.gz   classic; every generation writes it
    cell_feature_matrix.zarr.zip          v3+; the feature list lives in a plain
                                          JSON member, so it is readable without
                                          the zarr library
    cell_feature_matrix.h5                v2+; needs h5py
    gene_panel.json                       the panel design, not the run output —
                                          last resort, since it says what was
                                          ordered rather than what was measured

Vocabulary normalisation matters more than it looks. A v1 run labels RNA targets
``Gene Expression``; a v6 zarr labels them ``gene``. A run that carried
``Blank Codeword`` in v1 carries ``Unassigned Codeword`` and
``Deprecated Codeword`` in v6. Comparing raw labels across a study spanning two
Ranger versions would show a panel difference that is purely nomenclature.

The ``aggregate_gene`` row is dropped. v6 zarr appends a synthetic
``Total transcripts`` feature holding each cell's row sum; counted as a gene it
would dominate every downstream number.
"""

from __future__ import annotations

import gzip
import json
import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

#: Canonical feature types, after normalisation.
GENE = "gene"
AGGREGATE_GENE = "aggregate_gene"
NEGATIVE_CONTROL_PROBE = "negative_control_probe"
NEGATIVE_CONTROL_CODEWORD = "negative_control_codeword"
BLANK_CODEWORD = "blank_codeword"
UNASSIGNED_CODEWORD = "unassigned_codeword"
DEPRECATED_CODEWORD = "deprecated_codeword"
GENOMIC_CONTROL = "genomic_control"

#: Raw label -> canonical, for spellings that survive the generic normaliser
#: differently across versions.
_TYPE_ALIASES = {
    "gene_expression": GENE,
    "rna": GENE,
    "negative_control_probes": NEGATIVE_CONTROL_PROBE,
    "negative_control_codewords": NEGATIVE_CONTROL_CODEWORD,
    "unassigned_codewords": UNASSIGNED_CODEWORD,
    "deprecated_codewords": DEPRECATED_CODEWORD,
    "blank_codewords": BLANK_CODEWORD,
    "genomic_controls": GENOMIC_CONTROL,
}

#: True negative controls: probes and codewords designed not to bind anything.
#: These are comparable across panel versions, which is what makes them the
#: right basis for a cross-run quality metric.
STRICT_CONTROL_TYPES = (
    NEGATIVE_CONTROL_PROBE,
    NEGATIVE_CONTROL_CODEWORD,
    BLANK_CODEWORD,
)

#: Decoding background. Which codewords are unassigned or deprecated depends on
#: the panel design and the Ranger version, so this rate is NOT comparable
#: across runs built on different panels — it is reported separately for that
#: reason.
BACKGROUND_CONTROL_TYPES = (UNASSIGNED_CODEWORD, DEPRECATED_CODEWORD)


def normalise_feature_type(raw: Any) -> str:
    """Map any Xenium feature-type spelling onto a canonical one."""
    s = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return _TYPE_ALIASES.get(s, s)


@dataclass
class FeatureTable:
    """The features of one run, normalised."""

    frame: pd.DataFrame          # columns: feature_id, feature_name, feature_type
    source: str                  # which file it came from
    raw_type_counts: dict[str, int] = field(default_factory=dict)

    @property
    def genes(self) -> list[str]:
        """RNA target names, de-duplicated and sorted."""
        m = self.frame["feature_type"] == GENE
        return sorted(set(self.frame.loc[m, "feature_name"].astype(str)))

    @property
    def duplicate_gene_names(self) -> list[str]:
        m = self.frame["feature_type"] == GENE
        names = self.frame.loc[m, "feature_name"].astype(str)
        return sorted(names[names.duplicated()].unique().tolist())

    @property
    def type_counts(self) -> dict[str, int]:
        return self.frame["feature_type"].value_counts().to_dict()

    def names_of_type(self, *types: str) -> list[str]:
        m = self.frame["feature_type"].isin(types)
        return self.frame.loc[m, "feature_name"].astype(str).tolist()

    @property
    def n_strict_controls(self) -> int:
        return int(self.frame["feature_type"].isin(STRICT_CONTROL_TYPES).sum())

    @property
    def n_background_controls(self) -> int:
        return int(self.frame["feature_type"].isin(BACKGROUND_CONTROL_TYPES).sum())


def _finalise(rows: pd.DataFrame, source: str) -> FeatureTable:
    raw_counts = rows["feature_type"].astype(str).value_counts().to_dict()
    rows = rows.copy()
    rows["feature_type"] = rows["feature_type"].map(normalise_feature_type)

    n_aggregate = int((rows["feature_type"] == AGGREGATE_GENE).sum())
    if n_aggregate:
        # v6 zarr appends a synthetic "Total transcripts" feature holding each
        # cell's row sum. Counted as a gene it would dwarf every real one.
        logger.info(
            "Dropping %d aggregate feature(s) (e.g. 'Total transcripts') from %s",
            n_aggregate, source,
        )
        rows = rows[rows["feature_type"] != AGGREGATE_GENE]

    return FeatureTable(
        frame=rows.reset_index(drop=True),
        source=source,
        raw_type_counts=raw_counts,
    )


# ---------------------------------------------------------------------------
# Readers, in preference order
# ---------------------------------------------------------------------------

def read_features_tsv(path: Path) -> FeatureTable:
    """Classic ``cell_feature_matrix/features.tsv.gz`` (3 unnamed columns)."""
    path = Path(path)
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt") as fh:
        first = fh.readline().rstrip("\n").split("\t")

    # A header row is one whose third field is not a recognisable feature type.
    has_header = len(first) < 3 or normalise_feature_type(first[2]) not in {
        GENE, NEGATIVE_CONTROL_PROBE, NEGATIVE_CONTROL_CODEWORD, BLANK_CODEWORD,
        UNASSIGNED_CODEWORD, DEPRECATED_CODEWORD, GENOMIC_CONTROL, AGGREGATE_GENE,
    }

    df = pd.read_csv(
        path, sep="\t", header=0 if has_header else None, dtype=str,
        compression="gzip" if path.name.endswith(".gz") else None,
    )
    df = df.iloc[:, :3]
    df.columns = ["feature_id", "feature_name", "feature_type"]
    return _finalise(df, path.name)


def read_features_zarr_zip(path: Path) -> FeatureTable:
    """
    ``cell_feature_matrix.zarr.zip``.

    The feature list is a plain JSON member (``cell_features/.zattrs``), so this
    needs no zarr library — only the standard-library zip reader.
    """
    path = Path(path)
    with zipfile.ZipFile(path) as z:
        member = next(
            (n for n in z.namelist() if n.endswith("cell_features/.zattrs")), None
        )
        if member is None:
            raise ValueError(
                f"{path.name} has no 'cell_features/.zattrs' member; "
                "it does not look like a Xenium cell-feature zarr."
            )
        attrs = json.loads(z.read(member))

    names = attrs.get("feature_keys") or attrs.get("feature_names")
    ids = attrs.get("feature_ids") or names
    types = attrs.get("feature_types")
    if not names or not types:
        raise ValueError(
            f"{path.name}: cell_features/.zattrs lacks feature_keys/feature_types "
            f"(present: {sorted(attrs)})."
        )

    df = pd.DataFrame(
        {"feature_id": list(ids), "feature_name": list(names),
         "feature_type": list(types)}
    )
    return _finalise(df, path.name)


def read_features_h5(path: Path) -> FeatureTable:
    """``cell_feature_matrix.h5`` (CellRanger-style HDF5). Needs h5py."""
    path = Path(path)
    try:
        import h5py
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            f"Reading {path.name} needs h5py (pip install h5py). The audit "
            "prefers features.tsv.gz or the zarr zip, neither of which does."
        ) from e

    with h5py.File(path, "r") as fh:
        grp = fh["matrix"]["features"]

        def _col(key):
            return [
                v.decode() if isinstance(v, bytes) else str(v) for v in grp[key][:]
            ]

        df = pd.DataFrame(
            {
                "feature_id": _col("id"),
                "feature_name": _col("name"),
                "feature_type": _col("feature_type"),
            }
        )
    return _finalise(df, path.name)


def read_features_gene_panel_json(path: Path) -> FeatureTable:
    """
    ``gene_panel.json`` — the panel *design*.

    Last resort: it records what was ordered, not what the run measured, and it
    carries no control codewords. Used only when no matrix-side feature list is
    available, and the caller is told which source was used.
    """
    path = Path(path)
    payload = json.loads(path.read_text())

    targets = payload.get("payload", {}).get("targets") or payload.get("targets")
    if not targets:
        raise ValueError(f"{path.name}: no 'targets' array found.")

    rows = []
    for t in targets:
        info = t.get("type", {}) if isinstance(t, dict) else {}
        gene = info.get("data", {}) if isinstance(info, dict) else {}
        name = gene.get("name") or t.get("name")
        gid = gene.get("id") or t.get("id") or name
        if name:
            rows.append({"feature_id": gid, "feature_name": name,
                         "feature_type": GENE})
    if not rows:
        raise ValueError(f"{path.name}: no named targets found.")
    return _finalise(pd.DataFrame(rows), path.name)


#: Bundle-relative locations tried in order, with their reader.
FEATURE_SOURCES: tuple[tuple[str, Any], ...] = (
    ("cell_feature_matrix/features.tsv.gz", read_features_tsv),
    ("cell_feature_matrix/features.tsv", read_features_tsv),
    ("cell_feature_matrix.zarr.zip", read_features_zarr_zip),
    ("cell_feature_matrix.h5", read_features_h5),
    ("gene_panel.json", read_features_gene_panel_json),
)


def find_feature_table(run_dir: Path) -> tuple[FeatureTable | None, list[str]]:
    """
    Read the feature list from the first usable source in ``run_dir``.

    Returns ``(table_or_None, errors)`` where ``errors`` describes every source
    that was present but unreadable — so a bundle that has a zarr the audit
    could not open says so instead of silently reporting no panel.
    """
    run_dir = Path(run_dir)
    errors: list[str] = []

    for rel, reader in FEATURE_SOURCES:
        candidate = run_dir / rel
        if not candidate.exists():
            continue
        try:
            return reader(candidate), errors
        except Exception as e:
            errors.append(f"{rel}: {type(e).__name__}: {e}")

    return None, errors
