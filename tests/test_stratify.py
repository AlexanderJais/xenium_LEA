"""
Stratified auditing — analysing within the levels of a confounded covariate.

The scenario throughout is the real one: a covariate perfectly aliased with the
panel design and the segmentation method, so the pooled study cannot ask about
it, and splitting on it makes those factors constant. The tests pin both halves
of that trade — what the split fixes, and what it costs.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from xenium_lea.cli import main
from xenium_lea.manifest import RunManifest
from xenium_lea.report import write_stratified_report
from xenium_lea.stratify import run_stratified_audit, split_manifest

from . import fixtures as fx


def _confounded_by_sex(tmp_path):
    """
    Condition is balanced across the two batches; sex is not.

    Every male was processed on one panel design with nucleus expansion, every
    female on the other design with the staining kit — so aged-vs-adult is
    clean while sex is inseparable from the technology.
    """
    manifest_path, panel_csv = fx.build_metrics_study(
        tmp_path,
        [
            ("M_A1", "MA1", "aged", "20250626_run1", "7ZBFXR", 0.0),
            ("M_A2", "MA2", "aged", "20250701_run2", "7ZBFXR", 0.0),
            ("M_D1", "MD1", "adult", "20250626_run1", "7ZBFXR", 0.0),
            ("M_D2", "MD2", "adult", "20250701_run2", "7ZBFXR", 0.0),
            ("F_A1", "FA1", "aged", "20251103_run1", "NCY734", 0.93),
            ("F_A2", "FA2", "aged", "20251106_run2", "NCY734", 0.94),
            ("F_D1", "FD1", "adult", "20251103_run1", "NCY734", 0.95),
            ("F_D2", "FD2", "adult", "20251106_run2", "NCY734", 0.96),
        ],
    )
    m = pd.read_csv(manifest_path)
    m["sex"] = ["male"] * 4 + ["female"] * 4
    m.to_csv(manifest_path, index=False)
    return manifest_path, panel_csv


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def test_split_manifest_partitions_on_a_covariate(tmp_path):
    manifest_path, _ = _confounded_by_sex(tmp_path)
    manifest = RunManifest.from_csv(manifest_path)

    groups = split_manifest(manifest, "sex")

    assert set(groups) == {"male", "female"}
    assert len(groups["male"]) == 4
    assert len(groups["female"]) == 4
    # Every run lands in exactly one stratum.
    assert sorted(groups["male"].run_ids + groups["female"].run_ids) == sorted(
        manifest.run_ids
    )


def test_split_manifest_works_on_a_required_column(tmp_path):
    manifest_path, _ = _confounded_by_sex(tmp_path)
    groups = split_manifest(RunManifest.from_csv(manifest_path), "condition")
    assert set(groups) == {"aged", "adult"}


# ---------------------------------------------------------------------------
# What the split fixes
# ---------------------------------------------------------------------------

def test_splitting_removes_the_covariate_confound(tmp_path):
    manifest_path, panel_csv = _confounded_by_sex(tmp_path)

    strat = run_stratified_audit(manifest_path and RunManifest.from_csv(manifest_path),
                                 panel_csv, split_by="sex")

    # The pooled study cannot ask about sex...
    assert strat.pooled.findings.has("design.covariate_aliased_with_technical")
    assert strat.pooled.findings.has_errors

    # ...and inside each stratum the entangled factors simply do not vary.
    assert len(strat.audited) == 2
    for s in strat.audited:
        verdicts = s.result.design.verdicts
        assert verdicts["segmentation_kit"].verdict == "CONSTANT"
        assert verdicts["panel_design_id"].verdict == "CONSTANT"
        assert verdicts["sex"].verdict == "CONSTANT"
        # The biological contrast survives in both.
        assert s.result.design.condition_levels == ["adult", "aged"]
        assert not s.result.findings.has_errors

    assert strat.findings.has("stratify.confounds_resolved")
    resolved = strat.findings.by_code("stratify.confounds_resolved")[0]
    assert {"panel_design_id", "segmentation_kit"} <= set(
        resolved.evidence["resolved"]
    )


def test_a_covariate_nested_in_condition_is_not_called_an_unresolved_confound(tmp_path):
    """
    Age in weeks determines aged/adult by construction. Reporting it as a
    confound the split failed to fix would say the split did not work when it
    did exactly what was asked.
    """
    manifest_path, panel_csv = _confounded_by_sex(tmp_path)
    m = pd.read_csv(manifest_path)
    m["age_weeks"] = [70, 65, 29, 30, 68, 73, 25, 24]
    m.to_csv(manifest_path, index=False)

    strat = run_stratified_audit(
        RunManifest.from_csv(manifest_path), panel_csv, split_by="sex"
    )

    assert not strat.findings.has("stratify.confounds_remain")


# ---------------------------------------------------------------------------
# What the split costs, and what it forecloses
# ---------------------------------------------------------------------------

def test_replication_cost_is_reported(tmp_path):
    manifest_path, panel_csv = _confounded_by_sex(tmp_path)
    strat = run_stratified_audit(
        RunManifest.from_csv(manifest_path), panel_csv, split_by="sex"
    )

    # Pooled had 4 mice per group; each stratum has 2.
    pooled_reps = strat.pooled.design.replicates
    assert all(v["n_mice"] == 4 for v in pooled_reps.values())
    for s in strat.audited:
        assert all(v["n_mice"] == 2 for v in s.result.design.replicates.values())

    assert strat.findings.has("stratify.low_replication")


def test_loss_of_the_between_stratum_comparison_is_stated(tmp_path):
    manifest_path, panel_csv = _confounded_by_sex(tmp_path)
    strat = run_stratified_audit(
        RunManifest.from_csv(manifest_path), panel_csv, split_by="sex"
    )

    assert strat.findings.has("stratify.no_cross_stratum_comparison")
    msg = strat.findings.by_code("stratify.no_cross_stratum_comparison")[0].message
    # It must say the comparison is lost, not merely deferred.
    assert "cannot be compared" in msg


def test_strata_are_framed_as_independent_replicates(tmp_path):
    """The strongest thing available from this design, and easy to miss."""
    manifest_path, panel_csv = _confounded_by_sex(tmp_path)
    strat = run_stratified_audit(
        RunManifest.from_csv(manifest_path), panel_csv, split_by="sex"
    )

    assert strat.findings.has("stratify.independent_replication")


# ---------------------------------------------------------------------------
# Comparison table + output
# ---------------------------------------------------------------------------

def test_comparison_puts_pooled_and_strata_side_by_side(tmp_path):
    manifest_path, panel_csv = _confounded_by_sex(tmp_path)
    strat = run_stratified_audit(
        RunManifest.from_csv(manifest_path), panel_csv, split_by="sex"
    )

    c = strat.comparison.set_index("stratum")
    assert set(c.index) == {"POOLED", "sex=male", "sex=female"}
    assert c.loc["POOLED", "n_runs"] == 8
    assert c.loc["sex=male", "n_runs"] == 4
    # The pooled run carries the error; the strata do not.
    assert c.loc["POOLED", "n_errors"] > 0
    assert c.loc["sex=male", "n_errors"] == 0


def test_a_stratum_with_too_few_runs_is_skipped_not_crashed(tmp_path):
    manifest_path, panel_csv = _confounded_by_sex(tmp_path)
    m = pd.read_csv(manifest_path)
    m.loc[m.index[-1], "sex"] = "unknown"      # a stratum of one
    m.to_csv(manifest_path, index=False)

    strat = run_stratified_audit(
        RunManifest.from_csv(manifest_path), panel_csv, split_by="sex"
    )

    skipped = [s for s in strat.strata if not s.ok]
    assert len(skipped) == 1
    assert skipped[0].value == "unknown"
    assert strat.findings.has("stratify.stratum_too_small")
    # The others still audit.
    assert len(strat.audited) == 2


def test_single_level_split_is_reported_as_a_no_op(tmp_path):
    manifest_path, panel_csv = _confounded_by_sex(tmp_path)
    m = pd.read_csv(manifest_path)
    m["sex"] = "male"
    m.to_csv(manifest_path, index=False)

    strat = run_stratified_audit(
        RunManifest.from_csv(manifest_path), panel_csv, split_by="sex"
    )
    assert strat.findings.has("stratify.single_level")


def test_stratified_report_writes_an_index_and_one_report_per_stratum(tmp_path):
    manifest_path, panel_csv = _confounded_by_sex(tmp_path)
    strat = run_stratified_audit(
        RunManifest.from_csv(manifest_path), panel_csv, split_by="sex"
    )
    out = tmp_path / "out"

    written = write_stratified_report(strat, out)

    assert (out / "index.html").exists()
    assert (out / "pooled" / "report.html").exists()
    assert (out / "sex=male" / "report.html").exists()
    assert (out / "sex=female" / "report.html").exists()
    assert (out / "stratification_summary.csv").exists()

    index = (out / "index.html").read_text(encoding="utf-8")
    # The index links to each analysis and states the trade.
    assert 'href="sex=male/report.html"' in index
    assert 'href="pooled/report.html"' in index
    assert "independent replicates" in index
    assert "cannot be compared" in index


def test_cli_split_by_runs_end_to_end(tmp_path):
    manifest_path, panel_csv = _confounded_by_sex(tmp_path)
    out = tmp_path / "out"

    code = main([
        "audit", "--manifest", str(manifest_path),
        "--base-panel", str(panel_csv), "--out", str(out),
        "--split-by", "sex",
    ])

    # The exit status follows the strata, because those are the analyses that
    # will be used, and the split has made them error-free. The pooled report is
    # still written and still carries its error — so the run says explicitly
    # that a non-gating error exists, rather than letting exit 0 read as
    # "nothing found".
    assert code == 0
    findings = json.loads((out / "findings.json").read_text())
    codes = {f["code"] for f in findings["findings"]}
    assert "stratify.pooled_errors_not_gating" in codes
    pooled = json.loads((out / "pooled" / "findings.json").read_text())
    assert pooled["counts"]["error"] > 0

    assert (out / "index.html").exists()
    summary = pd.read_csv(out / "stratification_summary.csv")
    assert len(summary) == 3


def test_a_stratum_error_does_gate(tmp_path):
    """Errors inside a stratum are the ones that matter, and they still fail."""
    manifest_path, panel_csv = _confounded_by_sex(tmp_path)
    m = pd.read_csv(manifest_path)
    # Give the male stratum only one mouse per condition — not analysable.
    m.loc[m.sex == "male", "mouse_id"] = ["MA1", "MA1", "MD1", "MD1"]
    m.to_csv(manifest_path, index=False)
    out = tmp_path / "out"

    code = main([
        "audit", "--manifest", str(manifest_path),
        "--base-panel", str(panel_csv), "--out", str(out),
        "--split-by", "sex",
    ])

    assert code == 1
    male = json.loads((out / "sex=male" / "findings.json").read_text())
    assert any(
        f["code"] == "design.single_mouse_condition" for f in male["findings"]
    )


def test_cli_split_by_a_missing_column_puts_everything_in_one_stratum(tmp_path):
    manifest_path, panel_csv = _confounded_by_sex(tmp_path)
    out = tmp_path / "out"

    code = main([
        "audit", "--manifest", str(manifest_path),
        "--base-panel", str(panel_csv), "--out", str(out),
        "--split-by", "genotype", "--no-fail",
    ])

    assert code == 0
    findings = json.loads((out / "findings.json").read_text())
    assert any(f["code"] == "stratify.single_level" for f in findings["findings"])
