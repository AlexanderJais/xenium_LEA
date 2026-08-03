"""
Segmentation calling and per-run quality.

The segmentation tests matter most: the morphological fingerprint is what lets
the audit call the kit even when the metadata is silent or wrong, so it is
checked against bundles built with each geometry.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from xenium_lea.cell_qc import audit_cell_qc
from xenium_lea.findings import Findings
from xenium_lea.manifest import RunManifest
from xenium_lea.probe import probe_run
from xenium_lea.segmentation_audit import (
    KIT_EXPANSION,
    KIT_STAIN,
    KIT_UNKNOWN,
    audit_segmentation,
)

from . import fixtures as fx


def _probe(tmp_path, run_id, **kwargs):
    run = tmp_path / run_id
    fx.make_run(run, genes=fx.base_gene_names(15), **kwargs)
    entry = RunManifest().add(run_id, f"M_{run_id}", "AGED", run)[0]
    return probe_run(entry, Findings())


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def test_expansion_run_is_called_expansion(tmp_path):
    p = _probe(tmp_path, "R1", n_cells=800, segmentation=fx.EXPANSION, seed=1)
    table, calls = audit_segmentation([p], Findings())

    call = calls["R1"]
    assert call.call == KIT_EXPANSION
    assert call.morphological == KIT_EXPANSION
    # Nucleus-derived boundary: areas are near-deterministically coupled.
    assert call.metrics["nucleus_cell_area_spearman"] > 0.95
    assert call.metrics["area_ratio_cv"] < 0.2


def test_stain_run_is_called_stain(tmp_path):
    p = _probe(tmp_path, "R1", n_cells=800, segmentation=fx.STAIN, seed=2)
    table, calls = audit_segmentation([p], Findings())

    call = calls["R1"]
    assert call.call == KIT_STAIN
    assert call.morphological == KIT_STAIN
    # A traced membrane decouples nucleus and cell area.
    assert call.metrics["nucleus_cell_area_spearman"] < 0.5


def test_morphology_alone_identifies_expansion_without_any_metadata(tmp_path):
    """
    The reason the fingerprint exists: no experiment.xenium, no
    segmentation_method column, and the call must still be right.
    """
    p = _probe(
        tmp_path, "R1", n_cells=800, segmentation=fx.EXPANSION,
        include_experiment=False, seed=3,
    )
    _, calls = audit_segmentation([p], Findings())

    assert calls["R1"].declared == KIT_UNKNOWN
    assert calls["R1"].morphological == KIT_EXPANSION
    assert calls["R1"].call == KIT_EXPANSION


def test_mixed_study_is_flagged(tmp_path):
    a = _probe(tmp_path, "R1", n_cells=800, segmentation=fx.EXPANSION, seed=4)
    b = _probe(tmp_path, "R2", n_cells=800, segmentation=fx.STAIN, seed=5)

    f = Findings()
    table, calls = audit_segmentation([a, b], f)

    assert set(table["segmentation_kit"]) == {KIT_EXPANSION, KIT_STAIN}
    assert f.has("segmentation.mixed_methods")


def test_uniform_study_is_not_flagged_as_mixed(tmp_path):
    a = _probe(tmp_path, "R1", n_cells=800, segmentation=fx.STAIN, seed=6)
    b = _probe(tmp_path, "R2", n_cells=800, segmentation=fx.STAIN, seed=7)

    f = Findings()
    audit_segmentation([a, b], f)

    assert not f.has("segmentation.mixed_methods")
    assert f.has("segmentation.uniform")


def test_morphology_split_is_reported_across_runs(tmp_path):
    probes = [
        _probe(tmp_path, f"E{i}", n_cells=600, segmentation=fx.EXPANSION, seed=10 + i)
        for i in range(2)
    ] + [
        _probe(tmp_path, f"S{i}", n_cells=600, segmentation=fx.STAIN, seed=20 + i)
        for i in range(2)
    ]

    f = Findings()
    audit_segmentation(probes, f)

    assert f.has("segmentation.morphology_split")
    split = f.by_code("segmentation.morphology_split")[0]
    assert set(split.evidence["tightly_coupled"]) == {"E0", "E1"}
    assert set(split.evidence["loosely_coupled"]) == {"S0", "S1"}


def test_declared_metadata_conflicting_with_geometry_is_flagged(tmp_path):
    """A run whose JSON claims the kit but whose geometry says expansion."""
    run = tmp_path / "R1"
    fx.make_run(run, genes=fx.base_gene_names(15), n_cells=800,
                segmentation=fx.EXPANSION, seed=8)
    # Overwrite the metadata with a stain-kit declaration.
    meta = json.loads((run / "experiment.xenium").read_text())
    meta.pop("segmentation_method", None)
    meta.pop("nucleus_expansion_distance", None)
    meta["segmentation"] = {"method": "multimodal cell segmentation",
                            "boundary_stain": "ATP1A1"}
    (run / "experiment.xenium").write_text(json.dumps(meta))

    entry = RunManifest().add("R1", "M1", "AGED", run)[0]
    p = probe_run(entry, Findings())

    f = Findings()
    _, calls = audit_segmentation([p], f)

    assert calls["R1"].declared == KIT_STAIN
    assert calls["R1"].morphological == KIT_EXPANSION
    assert calls["R1"].disagreement
    assert calls["R1"].confidence == "low"
    assert f.has("segmentation.signals_disagree")


def test_too_few_cells_yields_no_morphological_call(tmp_path):
    p = _probe(tmp_path, "R1", n_cells=50, segmentation=fx.EXPANSION, seed=9)
    _, calls = audit_segmentation([p], Findings())
    assert calls["R1"].morphological == KIT_UNKNOWN


# ---------------------------------------------------------------------------
# Cell QC
# ---------------------------------------------------------------------------

def test_control_rate_is_computed_and_tracks_the_planted_value(tmp_path):
    p = _probe(tmp_path, "R1", n_cells=1500, control_rate=0.02,
               mean_counts_per_cell=100.0, seed=11)
    qc = audit_cell_qc([p], Findings())

    row = qc.per_run.iloc[0]
    assert row["control_rate"] == pytest.approx(0.02, abs=0.006)
    assert row["control_counts"] > 0
    assert row["median_transcripts_per_cell"] > 0


def test_high_control_rate_run_is_flagged(tmp_path):
    good = _probe(tmp_path, "R1", n_cells=1200, control_rate=0.01, seed=12)
    bad = _probe(tmp_path, "R2", n_cells=1200, control_rate=0.20, seed=13)

    f = Findings()
    audit_cell_qc([good, bad], f)

    assert f.has("qc.high_control_rate")
    flagged = f.by_code("qc.high_control_rate")[0]
    assert "R2" in flagged.run_ids
    assert "R1" not in flagged.run_ids


def test_zero_count_cells_are_flagged(tmp_path):
    p = _probe(tmp_path, "R1", n_cells=1000, zero_transcript_frac=0.25, seed=14)

    f = Findings()
    qc = audit_cell_qc([p], f)

    assert qc.per_run.iloc[0]["frac_cells_zero_transcripts"] == pytest.approx(
        0.25, abs=0.01
    )
    assert f.has("qc.zero_count_cells")


def test_filter_imbalance_is_flagged(tmp_path):
    """
    One run would lose far more cells than another to the same floor — the
    filter itself becomes a between-run difference.
    """
    clean = _probe(tmp_path, "R1", n_cells=1000, mean_counts_per_cell=120.0, seed=15)
    thin = _probe(tmp_path, "R2", n_cells=1000, mean_counts_per_cell=120.0,
                  zero_transcript_frac=0.4, seed=16)

    f = Findings()
    audit_cell_qc([clean, thin], f)

    assert f.has("qc.filter_imbalance")


def test_outlier_run_is_flagged_by_robust_z(tmp_path):
    probes = [
        _probe(tmp_path, f"R{i}", n_cells=800, mean_counts_per_cell=100.0,
               seed=30 + i)
        for i in range(5)
    ]
    probes.append(
        _probe(tmp_path, "ODD", n_cells=800, mean_counts_per_cell=15.0, seed=99)
    )

    f = Findings()
    qc = audit_cell_qc(probes, f)

    assert f.has("qc.outlier_run")
    flagged = {r for fnd in f.by_code("qc.outlier_run") for r in fnd.run_ids}
    assert "ODD" in flagged
    assert not qc.robust_z.empty


def test_tightly_clustered_runs_are_not_flagged_as_outliers(tmp_path):
    """
    A robust z alone over-fires on tight replicates — eight near-identical runs
    make a 5% difference score several MADs out. The relative-deviation floor is
    what keeps that from becoming a warning.
    """
    probes = [
        _probe(tmp_path, f"R{i}", n_cells=800, mean_counts_per_cell=100.0,
               seed=60 + i)
        for i in range(8)
    ]

    f = Findings()
    qc = audit_cell_qc(probes, f)

    medians = pd.to_numeric(qc.per_run["median_transcripts_per_cell"])
    # The runs really are near-identical...
    assert (medians.max() - medians.min()) / medians.median() < 0.1
    # ...so nothing should be called an outlier.
    assert not f.has("qc.outlier_run")


def test_qc_survives_a_run_with_no_cells_table(tmp_path):
    run = tmp_path / "R1"
    fx.make_run(run, genes=fx.base_gene_names(10), n_cells=50)
    (run / "cells.parquet").unlink()

    entry = RunManifest().add("R1", "M1", "AGED", run)[0]
    f = Findings()
    p = probe_run(entry, f)
    qc = audit_cell_qc([p], f)

    assert qc.per_run.iloc[0]["n_cells"] == 0
    assert f.has("qc.no_cell_tables")
