"""
matrix.py
---------
Per-gene count totals from whichever count-matrix container a run ships.

Only two numbers are ever needed from the matrix: the total counts per gene, and
(when a QC floor is applied) the total counts per cell. Both are reductions, so
each reader produces them without materialising a cells x genes array.

Sources, in preference order:

    cell_feature_matrix/matrix.mtx.gz   classic MatrixMarket; no extra deps
    cell_feature_matrix.h5              single file, needs h5py
    cell_feature_matrix.zarr.zip        v3+, needs zarr + numcodecs

Each reader takes its feature names from *its own* container rather than from
whatever the panel audit happened to read. The lists differ: a v6 zarr carries a
synthetic ``Total transcripts`` row that ``features.tsv.gz`` does not, so the two
have different lengths and row i is not the same feature in both. Pairing one
container's matrix with another's feature list would silently shift every gene.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import GENE, FeatureTable, read_features_h5, read_features_tsv, read_features_zarr_zip

logger = logging.getLogger(__name__)


@dataclass
class MatrixTotals:
    """Reductions over one run's count matrix."""

    gene_totals: pd.Series      # indexed by gene name, RNA targets only
    n_cells_total: int
    n_cells_used: int
    source: str


def _reduce(
    features: FeatureTable,
    gene_totals_all: np.ndarray,
    n_cells_total: int,
    n_cells_used: int,
    source: str,
) -> MatrixTotals:
    """Slice per-feature totals to RNA targets and key them by gene name."""
    frame = features.frame
    if len(frame) != len(gene_totals_all):
        raise ValueError(
            f"{source}: matrix has {len(gene_totals_all)} feature rows but its "
            f"feature list has {len(frame)} entries."
        )
    mask = (frame["feature_type"] == GENE).to_numpy()
    names = frame.loc[mask, "feature_name"].astype(str).to_numpy()
    values = np.asarray(gene_totals_all)[mask]

    # Duplicate symbols are summed, not dropped: splitting one gene's counts
    # across two rows would understate it in every downstream ratio.
    totals = pd.Series(values, index=names, dtype=float).groupby(level=0).sum()
    return MatrixTotals(
        gene_totals=totals,
        n_cells_total=n_cells_total,
        n_cells_used=n_cells_used,
        source=source,
    )


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def totals_from_mtx(run_dir: Path, min_counts_per_cell: int = 0) -> MatrixTotals:
    """Classic ``cell_feature_matrix/matrix.mtx.gz`` (features x cells)."""
    import scipy.io
    import scipy.sparse as sp

    mtx_dir = Path(run_dir) / "cell_feature_matrix"
    features = read_features_tsv(
        mtx_dir / ("features.tsv.gz" if (mtx_dir / "features.tsv.gz").exists()
                   else "features.tsv")
    )

    mat = sp.csc_matrix(scipy.io.mmread(mtx_dir / "matrix.mtx.gz"))
    n_cells_total = mat.shape[1]

    if min_counts_per_cell > 0:
        rna_rows = (features.frame["feature_type"] == GENE).to_numpy()
        per_cell = np.asarray(mat[rna_rows, :].sum(axis=0)).ravel()
        mat = mat[:, per_cell >= min_counts_per_cell]

    totals = np.asarray(mat.sum(axis=1)).ravel()
    return _reduce(features, totals, n_cells_total, mat.shape[1], "matrix.mtx.gz")


def totals_from_h5(run_dir: Path, min_counts_per_cell: int = 0) -> MatrixTotals:
    """``cell_feature_matrix.h5`` — CSC over cells, CellRanger layout."""
    import h5py
    import scipy.sparse as sp

    path = Path(run_dir) / "cell_feature_matrix.h5"
    features = read_features_h5(path)

    with h5py.File(path, "r") as fh:
        g = fh["matrix"]
        shape = tuple(int(v) for v in g["shape"][:])       # (n_features, n_cells)
        mat = sp.csc_matrix(
            (g["data"][:], g["indices"][:], g["indptr"][:]), shape=shape
        )

    n_cells_total = mat.shape[1]
    if min_counts_per_cell > 0:
        rna_rows = (features.frame["feature_type"] == GENE).to_numpy()
        per_cell = np.asarray(mat[rna_rows, :].sum(axis=0)).ravel()
        mat = mat[:, per_cell >= min_counts_per_cell]

    totals = np.asarray(mat.sum(axis=1)).ravel()
    return _reduce(
        features, totals, n_cells_total, mat.shape[1], "cell_feature_matrix.h5"
    )


def totals_from_zarr(run_dir: Path, min_counts_per_cell: int = 0) -> MatrixTotals:
    """
    ``cell_feature_matrix.zarr.zip``.

    The group holds the matrix twice: ``cell_features/{data,indices,indptr}`` is
    CSR over features, and ``cell_features/csc/*`` is the same matrix indexed by
    cell. Per-gene totals come straight from the feature-major ``indptr``, so no
    matrix is ever assembled.
    """
    try:
        import zarr
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "Reading cell_feature_matrix.zarr.zip needs zarr and numcodecs "
            "(pip install zarr numcodecs). The classic "
            "cell_feature_matrix/matrix.mtx.gz and cell_feature_matrix.h5 need "
            "neither and are tried first."
        ) from e

    path = Path(run_dir) / "cell_feature_matrix.zarr.zip"
    features = read_features_zarr_zip(path)

    # A .zarr.zip must be opened through a ZipStore. Handing zarr.open() the
    # path directly makes it look for a directory store and report the group as
    # missing, which reads as a corrupt file rather than a wrong store type.
    # ZipStore moved namespace between zarr 2 and 3.
    try:
        zip_store = zarr.storage.ZipStore  # zarr >= 3
    except AttributeError:  # pragma: no cover - zarr 2
        zip_store = zarr.ZipStore

    try:
        store = zip_store(str(path), mode="r")
    except TypeError:  # pragma: no cover - signature drift
        store = zip_store(str(path))

    root = zarr.open(store=store, mode="r")
    grp = root["cell_features"]

    indptr = np.asarray(grp["indptr"][:], dtype=np.int64)
    data = np.asarray(grp["data"][:], dtype=np.float64)
    n_features = len(indptr) - 1

    # The zarr feature list includes the synthetic aggregate row, which
    # features.py drops. Reduce against the full list, then filter.
    csc = grp["csc"]
    n_cells_total = len(np.asarray(csc["indptr"][:])) - 1

    if min_counts_per_cell > 0:
        cell_indptr = np.asarray(csc["indptr"][:], dtype=np.int64)
        cell_data = np.asarray(csc["data"][:], dtype=np.float64)
        cell_indices = np.asarray(csc["indices"][:], dtype=np.int64)
        # Row index -> is it an RNA target? Built from the *undropped* order.
        raw_types = _raw_zarr_types(path)
        rna_row = np.array([t == GENE for t in raw_types], dtype=bool)
        per_cell = np.add.reduceat(
            np.where(rna_row[cell_indices], cell_data, 0.0),
            cell_indptr[:-1],
        )
        per_cell[np.diff(cell_indptr) == 0] = 0.0
        keep = per_cell >= min_counts_per_cell

        # Recompute gene totals over kept cells only, from the cell-major view.
        totals_all = np.zeros(n_features, dtype=np.float64)
        for c in np.flatnonzero(keep):
            lo, hi = cell_indptr[c], cell_indptr[c + 1]
            np.add.at(totals_all, cell_indices[lo:hi], cell_data[lo:hi])
        n_cells_used = int(keep.sum())
    else:
        totals_all = np.add.reduceat(data, indptr[:-1])
        totals_all[np.diff(indptr) == 0] = 0.0
        n_cells_used = n_cells_total

    # Drop the rows features.py dropped, in the same order.
    raw_types = _raw_zarr_types(path)
    keep_rows = np.array([t != "aggregate_gene" for t in raw_types], dtype=bool)
    totals_all = totals_all[keep_rows]

    return _reduce(
        features, totals_all, n_cells_total, n_cells_used,
        "cell_feature_matrix.zarr.zip",
    )


def _raw_zarr_types(path: Path) -> list[str]:
    """Normalised feature types in the zarr's original row order."""
    import json
    import zipfile

    from .features import normalise_feature_type

    with zipfile.ZipFile(path) as z:
        member = next(n for n in z.namelist() if n.endswith("cell_features/.zattrs"))
        attrs = json.loads(z.read(member))
    return [normalise_feature_type(t) for t in attrs["feature_types"]]


#: Bundle-relative matrix locations, in preference order.
MATRIX_SOURCES: tuple[tuple[str, object], ...] = (
    ("cell_feature_matrix/matrix.mtx.gz", totals_from_mtx),
    ("cell_feature_matrix.h5", totals_from_h5),
    ("cell_feature_matrix.zarr.zip", totals_from_zarr),
)


def find_matrix_totals(
    run_dir: Path, min_counts_per_cell: int = 0
) -> tuple[MatrixTotals | None, list[str]]:
    """
    Reduce the first usable count matrix in ``run_dir``.

    Returns ``(totals_or_None, errors)``; ``errors`` names every container that
    was present but unreadable, so a missing optional dependency is reported
    rather than looking like a missing file.
    """
    run_dir = Path(run_dir)
    errors: list[str] = []
    for rel, reader in MATRIX_SOURCES:
        if not (run_dir / rel).exists():
            continue
        try:
            return reader(run_dir, min_counts_per_cell), errors  # type: ignore[operator]
        except Exception as e:
            errors.append(f"{rel}: {type(e).__name__}: {e}")
    return None, errors
