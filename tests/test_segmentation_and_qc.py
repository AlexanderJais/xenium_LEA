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
    p = _probe(tmp_path, "R1", n_cells=4000, segmentation=fx.EXPANSION, seed=1)
    table, calls = audit_segmentation([p], Findings())

    call = calls["R1"]
    assert call.call == KIT_EXPANSION
    assert call.morphological == KIT_EXPANSION
    # A dilated nucleus cannot exceed the configured distance, so the upper
    # percentiles of implied expansion converge.
    assert call.metrics["expansion_tail_ratio"] <= 1.25
    assert call.metrics["implied_expansion_max"] < 6.0


def test_stain_run_is_called_stain(tmp_path):
    p = _probe(tmp_path, "R1", n_cells=4000, segmentation=fx.STAIN, seed=2)
    table, calls = audit_segmentation([p], Findings())

    call = calls["R1"]
    assert call.call == KIT_STAIN
    assert call.morphological == KIT_STAIN
    # A traced boundary has no ceiling, so the tail keeps climbing.
    assert call.metrics["expansion_tail_ratio"] > 1.25


def test_loosely_coupled_areas_do_not_imply_a_traced_boundary(tmp_path):
    """
    Regression against the rule this replaced.

    Real nucleus expansion stops at neighbouring cells, so nucleus and cell
    area are only loosely coupled — a genuine 100%-expansion run measured
    Spearman 0.79 with a ratio CV of 0.45. The old fingerprint required tight
    coupling and therefore called such a run stain-segmented. The call must now
    rest on the ceiling, not the correlation.
    """
    p = _probe(tmp_path, "R1", n_cells=4000, segmentation=fx.EXPANSION, seed=3)
    _, calls = audit_segmentation([p], Findings())
    call = calls["R1"]

    # Loose coupling, exactly as the real run shows...
    assert call.metrics["nucleus_cell_area_spearman"] < 0.95
    assert call.metrics["area_ratio_cv"] > 0.05
    # ...and still correctly called expansion.
    assert call.morphological == KIT_EXPANSION


def test_morphology_alone_identifies_expansion_without_any_metadata(tmp_path):
    """
    The reason the fingerprint exists: no experiment.xenium, no
    segmentation_method column, and the call must still be right.
    """
    p = _probe(
        tmp_path, "R1", n_cells=4000, segmentation=fx.EXPANSION,
        include_experiment=False, seed=3,
    )
    _, calls = audit_segmentation([p], Findings())

    assert calls["R1"].declared == KIT_UNKNOWN
    assert calls["R1"].morphological == KIT_EXPANSION
    assert calls["R1"].call == KIT_EXPANSION


def test_mixed_study_is_flagged(tmp_path):
    a = _probe(tmp_path, "R1", n_cells=4000, segmentation=fx.EXPANSION, seed=4)
    b = _probe(tmp_path, "R2", n_cells=4000, segmentation=fx.STAIN, seed=5)

    f = Findings()
    table, calls = audit_segmentation([a, b], f)

    assert set(table["segmentation_kit"]) == {KIT_EXPANSION, KIT_STAIN}
    assert f.has("segmentation.mixed_methods")


def test_uniform_study_is_not_flagged_as_mixed(tmp_path):
    a = _probe(tmp_path, "R1", n_cells=4000, segmentation=fx.STAIN, seed=6)
    b = _probe(tmp_path, "R2", n_cells=4000, segmentation=fx.STAIN, seed=7)

    f = Findings()
    audit_segmentation([a, b], f)

    assert not f.has("segmentation.mixed_methods")
    assert f.has("segmentation.uniform")


def test_morphology_split_is_reported_across_runs(tmp_path):
    probes = [
        _probe(tmp_path, f"E{i}", n_cells=4000, segmentation=fx.EXPANSION, seed=10 + i)
        for i in range(2)
    ] + [
        _probe(tmp_path, f"S{i}", n_cells=4000, segmentation=fx.STAIN, seed=20 + i)
        for i in range(2)
    ]

    f = Findings()
    audit_segmentation(probes, f)

    assert f.has("segmentation.morphology_split")
    split = f.by_code("segmentation.morphology_split")[0]
    assert set(split.evidence["bounded_expansion_like"]) == {"E0", "E1"}
    assert set(split.evidence["unbounded_stain_like"]) == {"S0", "S1"}


def test_declared_metadata_conflicting_with_geometry_is_flagged(tmp_path):
    """A run whose JSON claims the kit but whose geometry says expansion."""
    run = tmp_path / "R1"
    fx.make_run(run, genes=fx.base_gene_names(15), n_cells=4000,
                segmentation=fx.EXPANSION, seed=8)
    # Replace the metadata with a stain-kit declaration and no quantitative
    # fractions — the pre-v4 situation, where all the metadata offers is a word.
    meta = json.loads((run / "experiment.xenium").read_text())
    for k in list(meta):
        if "segment" in k or "expansion" in k:
            meta.pop(k)
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


def test_interior_stain_majority_is_called_stain(tmp_path):
    """
    Regression, found on a real Xenium Ranger v6 run.

    A kit run labels the large majority of its cells "Segmented by interior
    stain (18S)", a small minority by boundary stain, and falls back to nucleus
    expansion for the rest. Matching only "boundary" saw just that minority and
    called the whole run nucleus-expanded — inverting the kit assignment for
    every real run.
    """
    p = _probe(tmp_path, "R1", n_cells=4000, segmentation=fx.STAIN, seed=31)

    methods = p.cells["segmentation_method"].value_counts()
    assert methods.idxmax().startswith("Segmented by interior stain")
    # Expansion is present as a fallback, and outnumbers boundary stain...
    assert methods.get("Segmented by nucleus expansion of 5.0µm", 0) > methods.get(
        "Segmented by boundary stain (ATP1A1+CD45+E-Cadherin)", 0
    )

    _, calls = audit_segmentation([p], Findings())
    # ...so a check that ignored interior stain would call this expansion.
    assert calls["R1"].structural == KIT_STAIN
    assert calls["R1"].call == KIT_STAIN


def test_declared_fractions_drive_the_call_when_present(tmp_path):
    """v4+ states the split outright; nothing needs inferring."""
    p = _probe(tmp_path, "R1", n_cells=4000, segmentation=fx.STAIN,
               stain_frac=0.93, seed=32)
    _, calls = audit_segmentation([p], Findings())

    call = calls["R1"]
    assert call.declared == KIT_STAIN
    assert call.metrics["declared_frac_stain"] == pytest.approx(0.93)
    assert call.metrics["declared_frac_nuc_expansion"] == pytest.approx(0.07)
    assert call.metrics["segmentation_stain"] == "Xenium Multi-Tissue Stain"
    assert call.confidence == "high"


def test_stain_fraction_spread_across_kit_runs_is_flagged(tmp_path):
    """
    Two runs both using the kit, but resolving very different shares of the
    section by stain — a graded technical difference in the same direction a
    kit-vs-no-kit difference would push.
    """
    a = _probe(tmp_path, "R1", n_cells=4000, segmentation=fx.STAIN,
               stain_frac=0.95, seed=33)
    b = _probe(tmp_path, "R2", n_cells=4000, segmentation=fx.STAIN,
               stain_frac=0.60, seed=34)

    f = Findings()
    table, _ = audit_segmentation([a, b], f)

    assert set(table["segmentation_kit"]) == {KIT_STAIN}
    assert f.has("segmentation.stain_fraction_spread")


def test_too_few_cells_yields_no_morphological_call(tmp_path):
    p = _probe(tmp_path, "R1", n_cells=500, segmentation=fx.EXPANSION, seed=9)
    _, calls = audit_segmentation([p], Findings())
    assert calls["R1"].morphological == KIT_UNKNOWN


# ---------------------------------------------------------------------------
# Cell QC
# ---------------------------------------------------------------------------

def test_control_rate_is_computed_and_tracks_the_planted_value(tmp_path):
    p = _probe(tmp_path, "R1", n_cells=1500, control_rate=0.02,
               background_rate=0.01, mean_counts_per_cell=100.0, seed=11)
    qc = audit_cell_qc([p], Findings())

    row = qc.per_run.iloc[0]
    # control_rate is the *strict* negative-control rate...
    assert row["control_rate"] == pytest.approx(0.02, rel=0.15)
    # ...unassigned/deprecated codewords are reported apart from it...
    assert row["background_rate"] == pytest.approx(0.01, rel=0.20)
    # ...and the two together make the total.
    assert row["total_control_rate"] == pytest.approx(
        row["control_rate"] + row["background_rate"], rel=1e-6
    )
    assert row["control_counts"] > 0
    assert row["median_transcripts_per_cell"] > 0


def test_background_codewords_do_not_inflate_the_quality_metric(tmp_path):
    """
    Two runs of identical quality, one on a panel version carrying far more
    deprecated codewords. The strict rate must be unmoved — that is the whole
    reason it is reported separately.
    """
    clean = _probe(tmp_path, "R1", n_cells=1200, control_rate=0.002,
                   background_rate=0.002, mean_counts_per_cell=100.0, seed=21)
    noisy = _probe(tmp_path, "R2", n_cells=1200, control_rate=0.002,
                   background_rate=0.15, mean_counts_per_cell=100.0, seed=22)

    f = Findings()
    qc = audit_cell_qc([clean, noisy], f).per_run.set_index("run_id")

    assert qc.loc["R1", "control_rate"] == pytest.approx(
        qc.loc["R2", "control_rate"], rel=0.25
    )
    assert qc.loc["R2", "background_rate"] > 10 * qc.loc["R1", "background_rate"]
    # The strict rate is what the outlier check scores, so no run is flagged.
    assert not f.has("qc.high_control_rate")


def test_high_control_rate_run_is_flagged(tmp_path):
    good = _probe(tmp_path, "R1", n_cells=1200, control_rate=0.002, seed=12)
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


def test_unusually_clean_runs_are_not_flagged(tmp_path):
    """
    A run with unusually *few* control counts is unusually clean. Reporting the
    low tail of a quality metric turns good news into a warning.
    """
    probes = [
        _probe(tmp_path, f"R{i}", n_cells=1500, control_rate=0.006,
               mean_counts_per_cell=100.0, seed=70 + i)
        for i in range(5)
    ]
    probes.append(
        _probe(tmp_path, "CLEAN", n_cells=1500, control_rate=0.0005,
               mean_counts_per_cell=100.0, seed=99)
    )

    f = Findings()
    qc = audit_cell_qc(probes, f).per_run.set_index("run_id")

    assert qc.loc["CLEAN", "control_rate"] < qc["control_rate"].median()
    flagged = {r for fnd in f.by_code("qc.outlier_run") for r in fnd.run_ids}
    assert "CLEAN" not in flagged


def test_tiny_absolute_control_rates_are_not_outliers(tmp_path):
    """
    Regression from the real male stratum: ten healthy runs whose control rates
    all sit around 0.015%, one at 0.020%. That is 34% away and several MADs out,
    and means nothing — without an absolute floor, the healthier a study is the
    more outliers it reports.
    """
    probes = [
        _probe(tmp_path, f"R{i}", n_cells=2000, control_rate=0.00015,
               background_rate=0.0, mean_counts_per_cell=150.0, seed=80 + i)
        for i in range(8)
    ]

    f = Findings()
    qc = audit_cell_qc(probes, f).per_run

    rates = pd.to_numeric(qc["control_rate"])
    # The rates really do scatter by tens of percent relative to each other...
    assert (rates.max() - rates.min()) / rates.median() > 0.25
    # ...but every one is far below anything that could matter.
    assert rates.max() < 0.002
    assert not f.has("qc.outlier_run")


def test_a_genuinely_high_control_rate_is_still_flagged(tmp_path):
    """The floor must not suppress a real problem."""
    probes = [
        _probe(tmp_path, f"R{i}", n_cells=1500, control_rate=0.003,
               mean_counts_per_cell=100.0, seed=90 + i)
        for i in range(5)
    ]
    probes.append(
        _probe(tmp_path, "BAD", n_cells=1500, control_rate=0.15,
               mean_counts_per_cell=100.0, seed=95)
    )

    f = Findings()
    audit_cell_qc(probes, f)

    flagged = {r for fnd in f.by_code("qc.outlier_run") for r in fnd.run_ids}
    assert "BAD" in flagged


def test_no_morphology_split_when_every_run_is_the_same_method(tmp_path):
    """
    Regression from the real female stratum: three stain runs measuring 1.396,
    1.416 and 1.468 were reported as splitting into two groups. They are all
    well past the decision threshold and all the same method — with a handful of
    runs, some gap always looks wide relative to a narrow range. A split is only
    a split if it crosses the boundary.
    """
    probes = [
        _probe(tmp_path, f"S{i}", n_cells=4000, segmentation=fx.STAIN, seed=40 + i)
        for i in range(3)
    ]

    f = Findings()
    table, _ = audit_segmentation(probes, f)

    ratios = pd.to_numeric(table["expansion_tail_ratio"])
    assert (ratios > 1.25).all()          # all the same side of the boundary
    assert not f.has("segmentation.morphology_split")
    assert f.has("segmentation.uniform")


def test_no_morphology_split_when_every_run_is_the_same_method(tmp_path):
    """
    Regression from the real female stratum: three stain runs measuring 1.396,
    1.416 and 1.468 were reported as splitting into two groups. They are all
    well past the decision threshold and all the same method — with a handful of
    runs, some gap always looks wide relative to a narrow range. A split is only
    a split if it crosses the boundary.
    """
    probes = [
        _probe(tmp_path, f"S{i}", n_cells=4000, segmentation=fx.STAIN, seed=40 + i)
        for i in range(3)
    ]

    f = Findings()
    table, _ = audit_segmentation(probes, f)

    ratios = pd.to_numeric(table["expansion_tail_ratio"])
    assert (ratios > 1.25).all()          # all the same side of the boundary
    assert not f.has("segmentation.morphology_split")
    assert f.has("segmentation.uniform")
