"""
Feature-list reading across Xenium Ranger generations.

Every case here is drawn from a real bundle. The vocabulary differences and the
synthetic ``Total transcripts`` row are not hypothetical — they are what a
Xenium Ranger v6 run actually ships, and each would corrupt the panel audit in a
different way if taken at face value.
"""

from __future__ import annotations

import json
import zipfile

import pandas as pd
import pytest

from xenium_lea.features import (
    AGGREGATE_GENE,
    BACKGROUND_CONTROL_TYPES,
    GENE,
    NEGATIVE_CONTROL_PROBE,
    STRICT_CONTROL_TYPES,
    find_feature_table,
    normalise_feature_type,
    read_features_tsv,
    read_features_zarr_zip,
)

from . import fixtures as fx


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Gene Expression", GENE),                       # v1-v4 features.tsv.gz
        ("gene", GENE),                                  # v6 zarr
        ("Negative Control Probe", NEGATIVE_CONTROL_PROBE),
        ("negative_control_probe", NEGATIVE_CONTROL_PROBE),
        ("Unassigned Codeword", "unassigned_codeword"),
        ("Deprecated Codeword", "deprecated_codeword"),
        ("aggregate_gene", AGGREGATE_GENE),
    ],
)
def test_vocabulary_is_normalised(raw, expected):
    assert normalise_feature_type(raw) == expected


def test_two_generations_agree_on_the_same_panel(tmp_path):
    """
    The same panel written in v4 and v6 vocabularies must produce identical gene
    lists — otherwise a study spanning two Ranger versions would show a panel
    difference that is purely nomenclature.
    """
    genes = fx.base_gene_names(12)

    classic = tmp_path / "classic"
    fx.make_run(classic, genes=genes, n_cells=40)
    v6 = fx.write_zarr_features(tmp_path / "cell_feature_matrix.zarr.zip", genes)

    from_tsv = read_features_tsv(classic / "cell_feature_matrix" / "features.tsv.gz")
    from_zarr = read_features_zarr_zip(v6)

    assert from_tsv.genes == from_zarr.genes == sorted(genes)
    assert from_tsv.type_counts[GENE] == from_zarr.type_counts[GENE]


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def test_zarr_zip_is_read_without_the_zarr_library(tmp_path, monkeypatch):
    """
    The feature list lives in a plain JSON zip member, so the panel audit must
    work on a machine with no zarr installed.
    """
    genes = fx.base_gene_names(10)
    path = fx.write_zarr_features(tmp_path / "cell_feature_matrix.zarr.zip", genes)

    import builtins

    real_import = builtins.__import__

    def _no_zarr(name, *a, **kw):
        if name.split(".")[0] in {"zarr", "numcodecs"}:
            raise ImportError(f"{name} is not available")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _no_zarr)

    table = read_features_zarr_zip(path)
    assert table.genes == sorted(genes)


def test_aggregate_gene_row_is_dropped(tmp_path):
    """
    v6 appends a synthetic 'Total transcripts' feature holding each cell's row
    sum. Counted as a gene it would dwarf every real one.
    """
    genes = fx.base_gene_names(10)
    path = fx.write_zarr_features(
        tmp_path / "cell_feature_matrix.zarr.zip", genes, include_aggregate=True
    )

    table = read_features_zarr_zip(path)

    assert "Total transcripts" not in table.genes
    assert AGGREGATE_GENE not in table.type_counts
    assert len(table.genes) == len(genes)
    # ...but the raw counts still record that it was there.
    assert table.raw_type_counts.get("aggregate_gene") == 1


def test_control_types_are_split_strict_vs_background(tmp_path):
    genes = fx.base_gene_names(8)
    path = fx.write_zarr_features(tmp_path / "cell_feature_matrix.zarr.zip", genes)

    table = read_features_zarr_zip(path)

    assert table.n_strict_controls == len(fx.STRICT_CONTROL_FEATURES)
    assert table.n_background_controls == len(fx.BACKGROUND_CONTROL_FEATURES)
    assert set(table.names_of_type(*STRICT_CONTROL_TYPES)).isdisjoint(
        table.names_of_type(*BACKGROUND_CONTROL_TYPES)
    )


# ---------------------------------------------------------------------------
# Source selection
# ---------------------------------------------------------------------------

def test_features_tsv_is_preferred_when_both_exist(tmp_path):
    genes = fx.base_gene_names(10)
    run = tmp_path / "R1"
    fx.make_run(run, genes=genes, n_cells=40)
    fx.write_zarr_features(run / "cell_feature_matrix.zarr.zip", genes)

    table, errors = find_feature_table(run)

    assert table is not None
    assert table.source == "features.tsv.gz"
    assert errors == []


def test_zarr_is_used_when_features_tsv_is_absent(tmp_path):
    """The layout the uploaded run appeared to have at first look."""
    genes = fx.base_gene_names(10)
    run = tmp_path / "R1"
    fx.make_run(run, genes=genes, n_cells=40)
    (run / "cell_feature_matrix" / "features.tsv.gz").unlink()
    fx.write_zarr_features(run / "cell_feature_matrix.zarr.zip", genes)

    table, errors = find_feature_table(run)

    assert table is not None
    assert table.source == "cell_feature_matrix.zarr.zip"
    assert table.genes == sorted(genes)


def test_a_corrupt_source_falls_through_and_is_reported(tmp_path):
    """A present-but-unreadable container must not look like a missing one."""
    genes = fx.base_gene_names(10)
    run = tmp_path / "R1"
    fx.make_run(run, genes=genes, n_cells=40)
    (run / "cell_feature_matrix" / "features.tsv.gz").unlink()
    (run / "cell_feature_matrix.zarr.zip").write_bytes(b"not a zip")
    # A valid fallback further down the chain.
    fx.write_zarr_features(run / "cell_feature_matrix.zarr.zip.bak", genes)

    table, errors = find_feature_table(run)

    assert table is None
    assert any("cell_feature_matrix.zarr.zip" in e for e in errors)


def test_no_feature_source_returns_none(tmp_path):
    run = tmp_path / "R1"
    run.mkdir()
    table, errors = find_feature_table(run)
    assert table is None
    assert errors == []


# ---------------------------------------------------------------------------
# Matrix containers
# ---------------------------------------------------------------------------

def test_mtx_and_zarr_matrices_give_identical_gene_totals(tmp_path):
    """
    Round-trip through both containers.

    The zarr carries a synthetic aggregate row that the MTX does not, so the two
    have different row counts — pairing one's matrix with the other's feature
    list would shift every gene. Totals must nonetheless agree exactly.
    """
    zarr = pytest.importorskip("zarr")
    import numpy as np
    import scipy.io
    import scipy.sparse as sp

    from xenium_lea.matrix import totals_from_mtx, totals_from_zarr

    genes = fx.base_gene_names(12)
    run = tmp_path / "R1"
    fx.make_run(run, genes=genes, n_cells=200, seed=5, mean_counts_per_cell=80.0)

    # Rebuild the same counts as a zarr, in the v6 layout: CSR over features
    # plus a CSC-over-cells view, and the aggregate row appended last.
    dense = np.asarray(
        scipy.io.mmread(run / "cell_feature_matrix" / "matrix.mtx.gz").todense()
    )
    aggregate = dense[: len(genes), :].sum(axis=0, keepdims=True)
    full = np.vstack([dense, aggregate])

    csr = sp.csr_matrix(full)
    csc = sp.csc_matrix(full)
    path = run / "cell_feature_matrix.zarr.zip"
    fx.write_zarr_features(path, genes, n_cells=200, include_aggregate=True)

    store = zarr.storage.ZipStore(str(path), mode="a")
    root = zarr.open(store=store, mode="a")
    grp = root["cell_features"]
    grp.create_array("data", shape=csr.data.shape, dtype="u4")[:] = csr.data
    grp.create_array("indices", shape=csr.indices.shape, dtype="u4")[:] = csr.indices
    grp.create_array("indptr", shape=csr.indptr.shape, dtype="u4")[:] = csr.indptr
    sub = grp.create_group("csc")
    sub.create_array("data", shape=csc.data.shape, dtype="u4")[:] = csc.data
    sub.create_array("indices", shape=csc.indices.shape, dtype="u4")[:] = csc.indices
    sub.create_array("indptr", shape=csc.indptr.shape, dtype="u4")[:] = csc.indptr
    store.close()

    from_mtx = totals_from_mtx(run)
    from_zarr = totals_from_zarr(run)

    assert from_mtx.n_cells_total == from_zarr.n_cells_total == 200
    # The aggregate row must not have become a gene.
    assert "Total transcripts" not in from_zarr.gene_totals.index
    assert len(from_zarr.gene_totals) == len(genes)
    pd.testing.assert_series_equal(
        from_mtx.gene_totals.sort_index(),
        from_zarr.gene_totals.sort_index(),
        check_dtype=False,
    )
