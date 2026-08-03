"""
End-to-end: manifest on disk -> audit -> report files -> CLI exit code.

The two scenarios are the ones the study could actually be in — a confounded
design and a clean one — checked all the way through to what the report says and
what the command-line exit code is.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from xenium_lea.audit import run_audit
from xenium_lea.cli import main
from xenium_lea.design import OVERALL_BLOCKED, OVERALL_OK
from xenium_lea.manifest import RunManifest
from xenium_lea.report import write_report

from . import fixtures as fx


def _confounded_study(root):
    """Segmentation kit and add-on panel both track condition. The bad case."""
    return fx.build_study(
        root,
        [
            {"run_id": "AGED_1_s1", "mouse_id": "MA1", "condition": "AGED",
             "addons": ["AddShared", "AddAged"], "segmentation": fx.STAIN, "seed": 1},
            {"run_id": "AGED_1_s2", "mouse_id": "MA1", "condition": "AGED",
             "addons": ["AddShared", "AddAged"], "segmentation": fx.STAIN, "seed": 2},
            {"run_id": "AGED_2_s1", "mouse_id": "MA2", "condition": "AGED",
             "addons": ["AddShared", "AddAged"], "segmentation": fx.STAIN, "seed": 3},
            {"run_id": "ADULT_1_s1", "mouse_id": "MD1", "condition": "ADULT",
             "addons": ["AddShared"], "segmentation": fx.EXPANSION, "seed": 4},
            {"run_id": "ADULT_1_s2", "mouse_id": "MD1", "condition": "ADULT",
             "addons": ["AddShared"], "segmentation": fx.EXPANSION, "seed": 5},
            {"run_id": "ADULT_2_s1", "mouse_id": "MD2", "condition": "ADULT",
             "addons": ["AddShared"], "segmentation": fx.EXPANSION, "seed": 6},
        ],
        base_genes=fx.base_gene_names(25),
        n_cells=400,
    )


def _clean_study(root):
    """Kit split within each mouse — the design that keeps everything estimable."""
    return fx.build_study(
        root,
        [
            {"run_id": "AGED_1_s1", "mouse_id": "MA1", "condition": "AGED",
             "addons": ["AddShared"], "segmentation": fx.STAIN, "seed": 1},
            {"run_id": "AGED_1_s2", "mouse_id": "MA1", "condition": "AGED",
             "addons": ["AddShared"], "segmentation": fx.EXPANSION, "seed": 2},
            {"run_id": "AGED_2_s1", "mouse_id": "MA2", "condition": "AGED",
             "addons": ["AddShared"], "segmentation": fx.STAIN, "seed": 3},
            {"run_id": "ADULT_1_s1", "mouse_id": "MD1", "condition": "ADULT",
             "addons": ["AddShared"], "segmentation": fx.EXPANSION, "seed": 4},
            {"run_id": "ADULT_1_s2", "mouse_id": "MD1", "condition": "ADULT",
             "addons": ["AddShared"], "segmentation": fx.STAIN, "seed": 5},
            {"run_id": "ADULT_2_s1", "mouse_id": "MD2", "condition": "ADULT",
             "addons": ["AddShared"], "segmentation": fx.EXPANSION, "seed": 6},
        ],
        base_genes=fx.base_gene_names(25),
        n_cells=400,
    )


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def test_confounded_study_is_blocked(tmp_path):
    manifest_path, panel_csv, _ = _confounded_study(tmp_path)
    result = run_audit(RunManifest.from_csv(manifest_path), panel_csv)

    design = result.design
    assert design.overall == OVERALL_BLOCKED
    assert design.verdicts["segmentation_kit"].verdict == "ALIASED"
    assert design.verdicts["panel_group"].verdict == "ALIASED"
    assert result.findings.has("design.verdict_blocked")
    assert result.findings.has_errors

    # Only the add-on every run carries survives into the safe set.
    assert "AddShared" in result.panel.safe_genes
    assert "AddAged" not in result.panel.safe_genes


def test_clean_study_is_not_blocked(tmp_path):
    manifest_path, panel_csv, _ = _clean_study(tmp_path)
    result = run_audit(RunManifest.from_csv(manifest_path), panel_csv)

    design = result.design
    assert design.overall == OVERALL_OK
    assert design.verdicts["segmentation_kit"].verdict == "CROSSED"
    assert design.verdicts["panel_group"].verdict == "CONSTANT"
    # Splitting a mouse's sections across kits is the controlled comparison.
    kit_contrasts = [
        c for c in design.within_mouse if c["factor"] == "segmentation_kit"
    ]
    assert {c["mouse_id"] for c in kit_contrasts} == {"MA1", "MD1"}
    # The software version moved with the kit, so it is surfaced as its own
    # within-mouse contrast too — same runs, second reason they differ.
    assert {c["factor"] for c in design.within_mouse} == {
        "segmentation_kit", "analysis_sw_version"
    }
    assert not result.findings.has_errors


def test_deep_pass_adds_metrics_and_stays_on_safe_genes(tmp_path):
    manifest_path, panel_csv, _ = _clean_study(tmp_path)
    result = run_audit(
        RunManifest.from_csv(manifest_path), panel_csv,
        deep=True, cache_dir=tmp_path / "cache",
    )

    assert result.metrics_section is not None
    assert not result.metrics_section.pca_scores.empty
    assert result.metrics_section.n_genes_used <= len(result.panel.safe_genes)
    assert not result.metrics_section.associations.empty
    # Three mice from six sections, so the mouse-level view is separate.
    assert result.metrics_mouse is not None


def test_tier0_skips_the_deep_pass(tmp_path):
    manifest_path, panel_csv, _ = _clean_study(tmp_path)
    result = run_audit(RunManifest.from_csv(manifest_path), panel_csv, deep=False)

    assert result.metrics_section is None
    assert result.findings.has("deep.skipped")


def test_audit_survives_a_broken_run(tmp_path):
    """One unreadable bundle must not abort the sweep."""
    manifest_path, panel_csv, _ = _clean_study(tmp_path)
    broken = tmp_path / "AGED_2_s1" / "cell_feature_matrix" / "features.tsv.gz"
    broken.unlink()

    result = run_audit(RunManifest.from_csv(manifest_path), panel_csv)

    assert result.findings.has("run.features_missing")
    # The other five runs are still audited.
    assert len(result.panel.per_run) == 5
    assert result.design is not None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def test_report_writes_every_output(tmp_path):
    manifest_path, panel_csv, _ = _confounded_study(tmp_path)
    result = run_audit(RunManifest.from_csv(manifest_path), panel_csv)
    out = tmp_path / "audit_out"

    written = write_report(result, out)

    for name in (
        "report.html", "findings.json", "separability_verdict.json",
        "inventory.csv", "panel_per_run.csv", "cell_qc.csv",
        "segmentation_audit.csv", "design_factors.csv", "safe_gene_set.txt",
    ):
        assert name in written, name
        assert written[name].exists(), name

    verdict = json.loads((out / "separability_verdict.json").read_text())
    assert verdict["overall"] == OVERALL_BLOCKED
    assert verdict["factors"]["segmentation_kit"]["verdict"] == "ALIASED"

    findings = json.loads((out / "findings.json").read_text())
    assert findings["counts"]["error"] > 0


def test_report_html_is_self_contained_and_leads_with_the_verdict(tmp_path):
    manifest_path, panel_csv, _ = _confounded_study(tmp_path)
    result = run_audit(
        RunManifest.from_csv(manifest_path), panel_csv,
        deep=True, cache_dir=tmp_path / "cache",
    )
    out = tmp_path / "audit_out"
    write_report(result, out)

    html = (out / "report.html").read_text(encoding="utf-8")

    assert "Separability verdict" in html
    assert OVERALL_BLOCKED in html
    # Figures are embedded, not linked.
    assert "data:image/png;base64," in html
    assert "http://" not in html and "https://" not in html
    assert "<script" not in html.lower()


def test_safe_gene_set_file_matches_the_audit(tmp_path):
    manifest_path, panel_csv, base = _confounded_study(tmp_path)
    result = run_audit(RunManifest.from_csv(manifest_path), panel_csv)
    out = tmp_path / "audit_out"
    write_report(result, out)

    genes = (out / "safe_gene_set.txt").read_text().split()
    assert genes == result.panel.safe_genes
    assert set(base).issubset(set(genes))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_exits_nonzero_on_a_confounded_study(tmp_path):
    manifest_path, panel_csv, _ = _confounded_study(tmp_path)
    out = tmp_path / "out"

    code = main([
        "audit", "--manifest", str(manifest_path),
        "--base-panel", str(panel_csv), "--out", str(out),
    ])

    assert code == 1
    assert (out / "report.html").exists()


def test_cli_exits_zero_on_a_clean_study(tmp_path):
    manifest_path, panel_csv, _ = _clean_study(tmp_path)
    out = tmp_path / "out"

    code = main([
        "audit", "--manifest", str(manifest_path),
        "--base-panel", str(panel_csv), "--out", str(out), "--deep",
    ])

    assert code == 0
    assert (out / "batch_section_associations.csv").exists()


def test_cli_no_fail_still_exits_zero(tmp_path):
    manifest_path, panel_csv, _ = _confounded_study(tmp_path)
    out = tmp_path / "out"

    code = main([
        "audit", "--manifest", str(manifest_path),
        "--base-panel", str(panel_csv), "--out", str(out), "--no-fail",
    ])

    assert code == 0


def test_cli_reports_a_missing_manifest_cleanly(tmp_path):
    code = main([
        "audit", "--manifest", str(tmp_path / "nope.csv"),
        "--base-panel", str(fx.write_base_panel_csv(tmp_path / "p.csv", ["A"])),
        "--out", str(tmp_path / "out"),
    ])
    assert code == 2


def test_partial_sections_state_their_coverage(tmp_path):
    """
    A table silently listing 1 of 14 runs reads as "the other 13 vanished".

    Only runs with a full bundle can contribute gene-level panel contents, so
    that section is legitimately partial — and it has to say so, naming what is
    missing and why.
    """
    from . import fixtures as fx

    root = tmp_path / "study"
    # One full bundle...
    base = fx.base_gene_names(15)
    panel_csv = fx.write_base_panel_csv(root / "base_panel.csv", base)
    full = root / "runs" / "FULL_1"
    fx.make_run(full, genes=base + ["AddA"], n_cells=200, segmentation=fx.STAIN)
    fx.write_metrics_summary(full / "metrics_summary.csv", region_name="FULL_1",
                             run_name="run_nov", panel_design_id="NCY734",
                             stain_frac=0.93)
    rows = [{"run_id": "FULL_1", "mouse_id": "MF", "section_id": "1",
             "condition": "aged", "run_dir": str(full)}]
    # ...and three metrics-only runs.
    for i, (rid, mouse, cond) in enumerate(
        [("M1_1", "M1", "aged"), ("M2_1", "M2", "adult"), ("M3_1", "M3", "adult")]
    ):
        d = root / "runs" / rid
        d.mkdir(parents=True, exist_ok=True)
        fx.write_metrics_summary(d / "metrics_summary.csv", region_name=rid,
                                 run_name="run_jun", panel_design_id="7ZBFXR",
                                 stain_frac=0.0)
        rows.append({"run_id": rid, "mouse_id": mouse, "section_id": "1",
                     "condition": cond, "run_dir": str(d)})
    manifest_path = fx.write_manifest(root / "manifest.csv", rows)

    result = run_audit(RunManifest.from_csv(manifest_path), panel_csv)
    out = tmp_path / "audit_out"
    write_report(result, out)
    html = (out / "report.html").read_text(encoding="utf-8")

    # The panel section covers one run and says so, naming the other three.
    assert "Covers 1 of 4 runs." in html
    for rid in ("M1_1", "M2_1", "M3_1"):
        assert rid in html
    assert "metrics_summary.csv names the panel design but not its genes" in html

    # Sections that do cover everything carry no coverage note...
    assert len(result.cell_qc.per_run) == 4
    assert len(result.segmentation) == 4
    # ...and every run reaches the inventory and the design analysis.
    assert len(result.design.factor_table) == 4


def test_qc_source_is_recorded_per_run(tmp_path):
    """
    Recomputed per-cell values and Ranger's run-level ones are not
    interchangeable, so which one a row came from has to travel with it.
    """
    from . import fixtures as fx

    root = tmp_path / "study"
    base = fx.base_gene_names(12)
    panel_csv = fx.write_base_panel_csv(root / "base_panel.csv", base)

    full = root / "runs" / "FULL_1"
    fx.make_run(full, genes=base, n_cells=300, segmentation=fx.STAIN)
    rows = [{"run_id": "FULL_1", "mouse_id": "MF", "section_id": "1",
             "condition": "aged", "run_dir": str(full)}]

    thin = root / "runs" / "THIN_1"
    thin.mkdir(parents=True, exist_ok=True)
    fx.write_metrics_summary(thin / "metrics_summary.csv", region_name="THIN_1",
                             stain_frac=0.0)
    rows.append({"run_id": "THIN_1", "mouse_id": "MT", "section_id": "1",
                 "condition": "adult", "run_dir": str(thin)})

    manifest_path = fx.write_manifest(root / "manifest.csv", rows)
    result = run_audit(RunManifest.from_csv(manifest_path), panel_csv)

    qc = result.cell_qc.per_run.set_index("run_id")
    assert qc.loc["FULL_1", "qc_source"] == "cells table"
    assert qc.loc["THIN_1", "qc_source"] == "metrics_summary.csv"
    # Both still report a cell count and a median.
    assert qc["n_cells"].gt(0).all()
    assert qc["median_transcripts_per_cell"].notna().all()
