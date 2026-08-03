"""
``metrics_summary.csv`` — Ranger's own per-run QC sheet.

A few kB per run, available long before anyone finishes copying multi-GB
bundles, and enough on its own for the inventory, the segmentation call and the
separability verdict. These tests use a fixture built from the real column set.
"""

from __future__ import annotations

import pandas as pd
import pytest

from xenium_lea.audit import run_audit
from xenium_lea.design import (
    GROUPING,
    OVERALL_NO_CONTRAST,
    UNKNOWN_DOMINATED,
    _alias_clusters,
    audit_design,
)
from xenium_lea.findings import Findings
from xenium_lea.manifest import RunManifest
from xenium_lea.metrics import read_metrics_summary, segmentation_from_metrics
from xenium_lea.probe import probe_run

from . import fixtures as fx


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_reads_the_real_column_set(tmp_path):
    path = fx.write_metrics_summary(
        tmp_path / "metrics_summary.csv",
        region_name="F536_1",
        run_name="20250626_Xv1_Lea_Droppman_Run1",
        panel_design_id="7ZBFXR",
        stain_frac=0.0,
    )
    m = read_metrics_summary(path)

    assert m["region_name"] == "F536_1"
    assert m["panel_design_id"] == "7ZBFXR"
    assert m["panel_predesigned_id"] == "mBrain_v1.1"
    assert m["frac_stain"] == 0.0
    assert m["frac_nuc_expansion"] == 1.0
    assert m["parse_error"] is None
    # Covariates that appear nowhere else in the bundle.
    assert m["section_thickness"] is not None
    assert m["transcript_density"] is not None
    assert m["declared_neg_control_probe_rate"] is not None


def test_no_stain_run_is_unambiguous(tmp_path):
    """stain_frac 0.0 / expansion 1.0 needs no inference at all."""
    no_kit = read_metrics_summary(
        fx.write_metrics_summary(tmp_path / "a.csv", stain_frac=0.0)
    )
    kit = read_metrics_summary(
        fx.write_metrics_summary(tmp_path / "b.csv", stain_frac=0.93)
    )

    assert segmentation_from_metrics(no_kit) == "nucleus_expansion"
    assert segmentation_from_metrics(kit) == "stain_kit"


def test_malformed_file_is_survivable(tmp_path):
    p = tmp_path / "metrics_summary.csv"
    p.write_text("not,a,valid\n")
    m = read_metrics_summary(p)
    assert m["parse_error"] is not None
    assert m["panel_design_id"] is None


# ---------------------------------------------------------------------------
# A metrics-only run
# ---------------------------------------------------------------------------

def test_metrics_only_run_still_yields_a_segmentation_call(tmp_path):
    run = tmp_path / "R1"
    run.mkdir()
    fx.write_metrics_summary(run / "metrics_summary.csv", stain_frac=0.0)

    f = Findings()
    entry = RunManifest().add("R1", "M1", "AGED", run)[0]
    p = probe_run(entry, f)

    assert p.metrics_source == "metrics_summary.csv"
    # Metrics fill the gaps experiment.xenium would otherwise supply.
    assert p.experiment["panel_design_id"] == "7ZBFXR"
    assert p.experiment["frac_nuc_expansion"] == 1.0
    # No feature list, but that is a gap rather than a failure.
    assert p.rna_genes == []
    assert not f.has("panel.features_missing")


def test_metrics_only_study_reaches_a_verdict(tmp_path):
    """The state a study is in mid-assembly — and the verdict still lands."""
    manifest_path, panel_csv = fx.build_metrics_study(
        tmp_path,
        [
            # Old batch: no kit, panel 7ZBFXR
            ("F536_1", "F536", "AGED", "run_jun", "7ZBFXR", 0.0),
            ("F536_2", "F536", "AGED", "run_jun", "7ZBFXR", 0.0),
            ("M493_1", "M493", "ADULT", "run_jun", "7ZBFXR", 0.0),
            ("M493_2", "M493", "ADULT", "run_jun", "7ZBFXR", 0.0),
            # New batch: kit, panel NCY734
            ("K238_2", "K238", "AGED", "run_nov", "NCY734", 0.93),
            ("P953_1", "P953", "ADULT", "run_nov", "NCY734", 0.94),
        ],
    )
    result = run_audit(RunManifest.from_csv(manifest_path), panel_csv)

    design = result.design
    # Kit and panel are crossed with condition here, so the biology survives...
    assert design.verdicts["segmentation_kit"].verdict == "CROSSED"
    assert design.overall == "OK"
    # ...but panel, kit and the run they were processed in all move together,
    # so no analysis can attribute an effect to one rather than another.
    cluster = next(c for c in design.alias_clusters if "segmentation_kit" in c)
    assert {"panel_design_id", "segmentation_kit", "run_name"} <= set(cluster)
    assert result.findings.has("segmentation.mixed_methods")
    assert not result.findings.has_errors


def test_metrics_only_availability_is_one_finding_not_thirteen(tmp_path):
    manifest_path, panel_csv = fx.build_metrics_study(
        tmp_path,
        [(f"R{i}", f"M{i}", "AGED" if i % 2 else "ADULT", "run_a", "7ZBFXR", 0.0)
         for i in range(8)],
    )
    result = run_audit(RunManifest.from_csv(manifest_path), panel_csv)

    only = result.findings.by_code("run.metrics_only")
    assert len(only) == 1
    assert len(only[0].run_ids) == 8


# ---------------------------------------------------------------------------
# Design behaviour this data exposed
# ---------------------------------------------------------------------------

def test_no_condition_gives_a_no_contrast_verdict_not_a_nested_one(tmp_path):
    """
    With one condition label, every factor trivially "determines" it. Reporting
    that as nesting would be true and useless; the honest answer is that the
    question is not yet answerable.
    """
    manifest_path, panel_csv = fx.build_metrics_study(
        tmp_path,
        [
            ("A1", "MA", "UNKNOWN", "run_jun", "7ZBFXR", 0.0),
            ("A2", "MA", "UNKNOWN", "run_jun", "7ZBFXR", 0.0),
            ("B1", "MB", "UNKNOWN", "run_nov", "NCY734", 0.93),
            ("B2", "MB", "UNKNOWN", "run_nov", "NCY734", 0.93),
        ],
    )
    result = run_audit(RunManifest.from_csv(manifest_path), panel_csv)

    assert result.design.overall == OVERALL_NO_CONTRAST
    assert result.design.verdicts["segmentation_kit"].verdict == GROUPING
    assert result.findings.has("design.verdict_no_contrast")
    # The technical structure is still reported.
    cluster = next(
        c for c in result.design.alias_clusters if "segmentation_kit" in c
    )
    assert {"panel_design_id", "segmentation_kit"} <= set(cluster)
    assert not result.findings.has_errors


def test_factor_unknown_for_most_runs_is_excluded(tmp_path):
    """
    A field known for only one run splits the study by *what was uploaded*.
    Left in, it would report a confident batch boundary that does not exist.
    """
    table = pd.DataFrame(
        [
            {"run_id": f"R{i}", "mouse_id": f"M{i}", "section_id": "s1",
             "condition": "AGED" if i < 3 else "ADULT",
             "segmentation_kit": "stain_kit" if i % 2 else "nucleus_expansion",
             "instrument_sn": "XETG00163" if i == 0 else "unknown"}
            for i in range(6)
        ]
    )
    f = Findings()
    audit = audit_design(table, findings=f)

    assert audit.verdicts["instrument_sn"].verdict == UNKNOWN_DOMINATED
    assert f.has("design.factors_unknown_dominated")
    # Excluded from aliasing too, so it cannot manufacture a cluster.
    assert not any("instrument_sn" in c for c in audit.alias_clusters)


def test_alias_clusters_collapse_transitively():
    """Eleven factors moving together is one fact, not fifty-five."""
    pairs = [
        {"factor_a": "a", "factor_b": "b", "aliased": True},
        {"factor_a": "b", "factor_b": "c", "aliased": True},
        {"factor_a": "a", "factor_b": "c", "aliased": True},
        {"factor_a": "d", "factor_b": "e", "aliased": True},
        {"factor_a": "a", "factor_b": "d", "aliased": False},
    ]
    clusters = sorted(_alias_clusters(pairs), key=len, reverse=True)
    assert clusters == [["a", "b", "c"], ["d", "e"]]


def test_run_date_is_recovered_from_the_run_name(tmp_path):
    """metrics_summary.csv carries no timestamp — the run name is all there is."""
    manifest_path, panel_csv = fx.build_metrics_study(
        tmp_path,
        [
            ("A1", "MA", "AGED", "20250626_Xv1_Lea_Run1", "7ZBFXR", 0.0),
            ("B1", "MB", "ADULT", "20251106_Lea_Run2", "NCY734", 0.93),
        ],
    )
    result = run_audit(RunManifest.from_csv(manifest_path), panel_csv)

    dates = set(result.design.factor_table["run_date"])
    assert dates == {"2025-06-26", "2025-11-06"}


def test_covariate_aliased_with_a_technical_factor_is_an_error(tmp_path):
    """
    The failure that hides behind a clean primary contrast.

    Condition is perfectly balanced across the two processing batches, so the
    main comparison is sound — but every male was processed in one batch and
    every female in the other, so a sex effect and a batch effect are the same
    contrast and that question is simply unanswerable here.
    """
    manifest_path, panel_csv = fx.build_metrics_study(
        tmp_path,
        [
            ("A1", "MA1", "aged", "run_jun", "7ZBFXR", 0.0),
            ("A2", "MA2", "adult", "run_jun", "7ZBFXR", 0.0),
            ("B1", "MB1", "aged", "run_nov", "NCY734", 0.93),
            ("B2", "MB2", "adult", "run_nov", "NCY734", 0.93),
        ],
    )
    # Sex tracks the batch exactly; condition does not.
    m = pd.read_csv(manifest_path)
    m["sex"] = ["male", "male", "female", "female"]
    m.to_csv(manifest_path, index=False)

    result = run_audit(RunManifest.from_csv(manifest_path), panel_csv)

    # The primary contrast is clean...
    assert result.design.overall == "OK"
    assert result.design.verdicts["segmentation_kit"].verdict == "CROSSED"
    # ...and the covariate is still reported as unanswerable.
    assert result.findings.has("design.covariate_aliased_with_technical")
    assert result.findings.has_errors
    finding = result.findings.by_code("design.covariate_aliased_with_technical")[0]
    assert finding.evidence["covariate"] == "sex"
    assert "segmentation_kit" in finding.evidence["aliased_with"]


def test_covariate_nested_in_condition_does_not_downgrade_the_verdict(tmp_path):
    """
    Age in weeks determines aged/adult — it is the definition of the groups,
    not a confound. The technical verdict must be unmoved by it.
    """
    manifest_path, panel_csv = fx.build_metrics_study(
        tmp_path,
        [
            ("A1", "MA1", "aged", "run_jun", "7ZBFXR", 0.0),
            ("A2", "MA2", "adult", "run_jun", "7ZBFXR", 0.0),
            ("B1", "MB1", "aged", "run_nov", "NCY734", 0.93),
            ("B2", "MB2", "adult", "run_nov", "NCY734", 0.93),
        ],
    )
    m = pd.read_csv(manifest_path)
    m["age_weeks"] = [70, 29, 68, 25]
    m.to_csv(manifest_path, index=False)

    result = run_audit(RunManifest.from_csv(manifest_path), panel_csv)

    assert result.design.verdicts["age_weeks"].verdict in ("NESTED", "PER_RUN")
    assert result.design.overall == "OK"
    assert not result.findings.has("design.verdict_caution")


def test_one_run_missing_metadata_does_not_make_every_field_partial(tmp_path):
    """
    Regression from the real female stratum.

    Three of four runs had a full bundle and one had only metrics_summary.csv.
    Every field experiment.xenium supplies then had two "levels" — the real
    value and "unknown" — and each was reported as PARTIAL, dragging an
    otherwise clean stratum to CAUTION. "unknown" is missing information, not a
    level: contrasting it against a real value compares "we measured X" with
    "we did not look".
    """
    table = pd.DataFrame(
        [
            {"run_id": "R1", "mouse_id": "M1", "section_id": "s1",
             "condition": "aged", "instrument_sn": "XETG00163",
             "chemistry_version": "v1"},
            {"run_id": "R2", "mouse_id": "M2", "section_id": "s1",
             "condition": "aged", "instrument_sn": "XETG00163",
             "chemistry_version": "v1"},
            {"run_id": "R3", "mouse_id": "M3", "section_id": "s1",
             "condition": "adult", "instrument_sn": "XETG00163",
             "chemistry_version": "v1"},
            {"run_id": "R4", "mouse_id": "M4", "section_id": "s1",
             "condition": "adult", "instrument_sn": "unknown",
             "chemistry_version": "unknown"},
        ]
    )
    f = Findings()
    audit = audit_design(table, findings=f)

    # Constant among the runs where it is known — not PARTIAL.
    assert audit.verdicts["instrument_sn"].verdict == "CONSTANT"
    assert audit.verdicts["chemistry_version"].verdict == "CONSTANT"
    assert audit.overall == "OK"
    # ...and the reduced coverage is stated rather than silently assumed away.
    assert f.has("design.factors_partial_coverage")


def test_a_factor_unknown_for_most_runs_is_still_excluded_entirely(tmp_path):
    """The minority case is classified on the rest; the majority case is not."""
    table = pd.DataFrame(
        [
            {"run_id": f"R{i}", "mouse_id": f"M{i}", "section_id": "s1",
             "condition": "aged" if i < 3 else "adult",
             "instrument_sn": "XETG00163" if i == 0 else "unknown"}
            for i in range(6)
        ]
    )
    f = Findings()
    audit = audit_design(table, findings=f)

    assert audit.verdicts["instrument_sn"].verdict == UNKNOWN_DOMINATED
    assert f.has("design.factors_unknown_dominated")


def test_one_run_missing_metadata_does_not_make_every_field_partial():
    """
    Regression from the female stratum mid-assembly.

    Three of four runs had a full bundle and one had only metrics_summary.csv.
    Every field experiment.xenium supplies then had two "levels" — the real
    value and "unknown" — and each was reported as PARTIAL, dragging an
    otherwise clean stratum to CAUTION. "unknown" is missing information, not a
    level: contrasting it against a real value compares "we measured X" with
    "we did not look".
    """
    table = pd.DataFrame(
        [
            {"run_id": "R1", "mouse_id": "M1", "section_id": "s1",
             "condition": "aged", "instrument_sn": "XETG00163",
             "chemistry_version": "v1"},
            {"run_id": "R2", "mouse_id": "M2", "section_id": "s1",
             "condition": "aged", "instrument_sn": "XETG00163",
             "chemistry_version": "v1"},
            {"run_id": "R3", "mouse_id": "M3", "section_id": "s1",
             "condition": "adult", "instrument_sn": "XETG00163",
             "chemistry_version": "v1"},
            {"run_id": "R4", "mouse_id": "M4", "section_id": "s1",
             "condition": "adult", "instrument_sn": "unknown",
             "chemistry_version": "unknown"},
        ]
    )
    f = Findings()
    audit = audit_design(table, findings=f)

    # Constant among the runs where it is known — not PARTIAL.
    assert audit.verdicts["instrument_sn"].verdict == "CONSTANT"
    assert audit.verdicts["chemistry_version"].verdict == "CONSTANT"
    assert audit.overall == "OK"
    # ...and the reduced coverage is stated rather than silently assumed away.
    assert f.has("design.factors_partial_coverage")


def test_a_factor_unknown_for_most_runs_is_still_excluded_entirely():
    """The minority case is classified on the rest; the majority case is not."""
    table = pd.DataFrame(
        [
            {"run_id": f"R{i}", "mouse_id": f"M{i}", "section_id": "s1",
             "condition": "aged" if i < 3 else "adult",
             "instrument_sn": "XETG00163" if i == 0 else "unknown"}
            for i in range(6)
        ]
    )
    f = Findings()
    audit = audit_design(table, findings=f)

    assert audit.verdicts["instrument_sn"].verdict == UNKNOWN_DOMINATED
    assert f.has("design.factors_unknown_dominated")
