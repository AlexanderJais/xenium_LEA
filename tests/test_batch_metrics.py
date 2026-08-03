"""
Tier-1 batch diagnostics.

The regression that matters most here is ``test_zero_filled_genes_are_excluded``:
including a gene one run's panel lacked turns the panel difference into an
apparent batch effect, which is the most confidently wrong result this audit
could produce.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xenium_lea.batch_metrics import (
    aggregate_to_mouse,
    analyse_batch,
    build_pseudobulk,
    normalise_pseudobulk,
    run_pseudobulk,
)
from xenium_lea.findings import Findings
from xenium_lea.manifest import RunManifest
from xenium_lea.panel_audit import audit_panels, load_base_panel
from xenium_lea.probe import probe_run

from . import fixtures as fx


def _probes(tmp_path, specs, base):
    """specs: list of (run_id, mouse_id, condition, addons, gene_scale, seed)."""
    manifest = RunManifest()
    for run_id, mouse, cond, addons, scale, seed in specs:
        d = tmp_path / run_id
        fx.make_run(
            d,
            genes=base + list(addons),
            n_cells=300,
            seed=seed,
            gene_scale=scale,
            mean_counts_per_cell=200.0,
        )
        manifest.add(run_id, mouse, cond, d)
    f = Findings()
    return manifest, [probe_run(e, f) for e in manifest], f


# ---------------------------------------------------------------------------
# Pseudobulk construction
# ---------------------------------------------------------------------------

def test_pseudobulk_totals_match_the_matrix(tmp_path):
    base = fx.base_gene_names(12)
    run = tmp_path / "R1"
    fx.make_run(run, genes=base, n_cells=200, seed=1)
    entry = RunManifest().add("R1", "M1", "AGED", run)[0]
    p = probe_run(entry, Findings())

    totals, info = run_pseudobulk(p)

    assert info["n_cells_total"] == 200
    assert info["n_cells_used"] == 200
    assert set(totals.index) == set(base)
    # Per-gene totals must agree with the per-cell totals in cells.parquet.
    cells = pd.read_parquet(run / "cells.parquet")
    assert totals.sum() == pytest.approx(cells["transcript_counts"].sum())


def test_min_counts_filter_drops_cells(tmp_path):
    base = fx.base_gene_names(12)
    run = tmp_path / "R1"
    fx.make_run(run, genes=base, n_cells=200, seed=2, zero_transcript_frac=0.3)
    entry = RunManifest().add("R1", "M1", "AGED", run)[0]
    p = probe_run(entry, Findings())

    _, unfiltered = run_pseudobulk(p, min_counts_per_cell=0)
    _, filtered = run_pseudobulk(p, min_counts_per_cell=10)

    assert unfiltered["n_cells_used"] == 200
    assert filtered["n_cells_used"] == pytest.approx(140, abs=10)


def test_pseudobulk_cache_round_trips(tmp_path):
    base = fx.base_gene_names(10)
    run = tmp_path / "R1"
    fx.make_run(run, genes=base, n_cells=150, seed=3)
    entry = RunManifest().add("R1", "M1", "AGED", run)[0]
    p = probe_run(entry, Findings())
    cache = tmp_path / "cache"

    first, info1 = run_pseudobulk(p, cache_dir=cache)
    second, info2 = run_pseudobulk(p, cache_dir=cache)

    assert info1["cached"] is False
    assert info2["cached"] is True
    pd.testing.assert_series_equal(
        first.sort_index(), second.sort_index(), check_dtype=False
    )


def test_zero_filled_genes_are_excluded(tmp_path):
    """
    Regression: a gene absent from one run must never reach the batch analysis.

    R1 carries AddX and R2 does not. Including AddX would give R2 a zero column
    indistinguishable from genuine non-expression, and the pseudobulk PCA would
    separate the runs on a bookkeeping artefact.
    """
    base = fx.base_gene_names(12)
    panel_csv = fx.write_base_panel_csv(tmp_path / "panel.csv", base)
    manifest, probes, f = _probes(
        tmp_path,
        [
            ("R1", "M1", "AGED", ["AddX"], None, 1),
            ("R2", "M2", "ADULT", [], None, 2),
        ],
        base,
    )
    panel = audit_panels(probes, load_base_panel(panel_csv), findings=f)

    assert "AddX" not in panel.safe_genes

    counts, skipped = build_pseudobulk(probes, panel.safe_genes, findings=f)

    assert skipped == []
    assert "AddX" not in counts.columns
    assert list(counts.columns) == panel.safe_genes
    # And no column is all-zero, which is what a zero-filled gene would look like.
    assert (counts.sum(axis=0) > 0).all()


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def test_normalise_is_cpm_log1p():
    counts = pd.DataFrame(
        [[10.0, 90.0], [200.0, 1800.0]], index=["a", "b"], columns=["g1", "g2"]
    )
    out = normalise_pseudobulk(counts)
    # Same composition, different depth -> identical after CPM.
    assert out.loc["a", "g1"] == pytest.approx(out.loc["b", "g1"])
    assert out.loc["a", "g1"] == pytest.approx(np.log1p(1e5))


def test_zero_library_does_not_divide_by_zero():
    counts = pd.DataFrame([[0.0, 0.0], [1.0, 1.0]], columns=["g1", "g2"])
    out = normalise_pseudobulk(counts)
    assert np.isfinite(out.to_numpy()).all()


def test_planted_batch_effect_is_recovered(tmp_path):
    """
    Six runs, condition fully crossed with a technical 'kit' factor, and a real
    expression shift planted on the kit. The association analysis must attribute
    the leading axis to the kit rather than to condition.
    """
    base = fx.base_gene_names(30)
    panel_csv = fx.write_base_panel_csv(tmp_path / "panel.csv", base)
    boosted = {g: 6.0 for g in base[:10]}

    specs = [
        ("K1_A", "M1", "AGED", [], boosted, 1),
        ("K1_B", "M2", "ADULT", [], boosted, 2),
        ("K1_C", "M3", "AGED", [], boosted, 3),
        ("K0_A", "M4", "ADULT", [], None, 4),
        ("K0_B", "M5", "AGED", [], None, 5),
        ("K0_C", "M6", "ADULT", [], None, 6),
    ]
    manifest, probes, f = _probes(tmp_path, specs, base)
    panel = audit_panels(probes, load_base_panel(panel_csv), findings=f)
    counts, _ = build_pseudobulk(probes, panel.safe_genes, findings=f)

    meta = pd.DataFrame(
        {
            "condition": [s[2] for s in specs],
            "mouse_id": [s[1] for s in specs],
            "segmentation_kit": [
                "stain_kit" if s[0].startswith("K1") else "nucleus_expansion"
                for s in specs
            ],
        },
        index=[s[0] for s in specs],
    ).reindex(counts.index)

    metrics = analyse_batch(
        counts, meta, factors=("condition", "segmentation_kit"), findings=f
    )

    pc1 = metrics.associations[metrics.associations["pc"] == "PC1"].set_index("factor")
    assert pc1.loc["segmentation_kit", "eta_squared"] > 0.9
    assert pc1.loc["segmentation_kit", "eta_squared"] > pc1.loc["condition", "eta_squared"]
    assert f.has("deep.pc_driver")


def test_no_dominant_factor_when_runs_are_homogeneous(tmp_path):
    base = fx.base_gene_names(30)
    panel_csv = fx.write_base_panel_csv(tmp_path / "panel.csv", base)
    specs = [
        (f"R{i}", f"M{i}", "AGED" if i % 2 else "ADULT", [], None, 10 + i)
        for i in range(6)
    ]
    manifest, probes, f = _probes(tmp_path, specs, base)
    panel = audit_panels(probes, load_base_panel(panel_csv), findings=f)
    counts, _ = build_pseudobulk(probes, panel.safe_genes, findings=f)

    meta = pd.DataFrame(
        {"condition": [s[2] for s in specs], "mouse_id": [s[1] for s in specs]},
        index=[s[0] for s in specs],
    ).reindex(counts.index)

    analyse_batch(counts, meta, factors=("condition", "mouse_id"), findings=f)
    assert f.has("deep.no_dominant_factor")


def test_many_level_factor_is_not_reported_as_a_driver(tmp_path):
    """
    eta-squared is biased upward by level count: a 5-level factor over 6 samples
    explains 80% of anything by arithmetic. It must not be called a driver.
    """
    base = fx.base_gene_names(30)
    panel_csv = fx.write_base_panel_csv(tmp_path / "panel.csv", base)
    specs = [
        (f"R{i}", f"M{i}", "AGED" if i % 2 else "ADULT", [], None, 40 + i)
        for i in range(6)
    ]
    _, probes, f = _probes(tmp_path, specs, base)
    panel = audit_panels(probes, load_base_panel(panel_csv), findings=f)
    counts, _ = build_pseudobulk(probes, panel.safe_genes, findings=f)

    # 5 levels across 6 samples: not degenerate, but nearly so.
    meta = pd.DataFrame(
        {
            "condition": [s[2] for s in specs],
            "run_date": ["d1", "d1", "d2", "d3", "d4", "d5"],
        },
        index=[s[0] for s in specs],
    ).reindex(counts.index)

    metrics = analyse_batch(
        counts, meta, factors=("condition", "run_date"), findings=f
    )

    row = metrics.associations[
        (metrics.associations["pc"] == "PC1")
        & (metrics.associations["factor"] == "run_date")
    ].iloc[0]
    assert row["eta_squared_null"] == pytest.approx(4 / 5)
    assert row["eta_squared"] >= row["eta_squared_null"]  # inflated by construction
    # ...and precisely because of that, it is not announced as a driver.
    drivers = [
        d
        for finding in f.by_code("deep.pc_driver")
        for d in finding.evidence["technical_drivers"]
    ]
    assert "run_date" not in drivers


def test_factor_with_one_level_per_sample_is_excluded(tmp_path):
    base = fx.base_gene_names(20)
    panel_csv = fx.write_base_panel_csv(tmp_path / "panel.csv", base)
    specs = [
        (f"R{i}", f"M{i}", "AGED" if i % 2 else "ADULT", [], None, 50 + i)
        for i in range(5)
    ]
    _, probes, f = _probes(tmp_path, specs, base)
    panel = audit_panels(probes, load_base_panel(panel_csv), findings=f)
    counts, _ = build_pseudobulk(probes, panel.safe_genes, findings=f)

    meta = pd.DataFrame(
        {"condition": [s[2] for s in specs], "mouse_id": [s[1] for s in specs]},
        index=[s[0] for s in specs],
    ).reindex(counts.index)

    metrics = analyse_batch(
        counts, meta, factors=("condition", "mouse_id"), findings=f
    )

    assert "mouse_id" not in set(metrics.associations["factor"])
    assert f.has("deep.factor_degenerate")


def test_too_few_samples_is_reported_not_crashed():
    counts = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=["a", "b"],
                          columns=["g1", "g2"])
    meta = pd.DataFrame({"condition": ["A", "B"]}, index=["a", "b"])
    f = Findings()
    metrics = analyse_batch(counts, meta, factors=("condition",), findings=f)

    assert metrics.pca_scores.empty
    assert f.has("deep.too_few_samples")


def test_aggregate_to_mouse_sums_sections():
    counts = pd.DataFrame(
        [[10.0, 20.0], [30.0, 40.0], [1.0, 1.0]],
        index=["R1", "R2", "R3"],
        columns=["g1", "g2"],
    )
    meta = pd.DataFrame(
        {
            "mouse_id": ["M1", "M1", "M2"],
            "condition": ["AGED", "AGED", "ADULT"],
            "section_id": ["s1", "s2", "s1"],
        },
        index=["R1", "R2", "R3"],
    )

    mcounts, mmeta = aggregate_to_mouse(counts, meta)

    assert list(mcounts.index) == ["M1", "M2"]
    assert mcounts.loc["M1", "g1"] == 40.0
    assert mmeta.loc["M1", "condition"] == "AGED"
    # A factor that varies within an animal has no single mouse-level value.
    assert mmeta.loc["M1", "section_id"] == "mixed"
