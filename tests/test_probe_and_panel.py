"""
Tier-0 probe and the gene-panel audit.

Each test plants one defect in a synthetic bundle and asserts the audit finds
it — the point being that these are the defects that otherwise pass silently
into an analysis.
"""

from __future__ import annotations

import gzip
import json

import pandas as pd
import pytest

from xenium_lea.findings import Findings
from xenium_lea.manifest import RunManifest
from xenium_lea.panel_audit import audit_panels, load_base_panel
from xenium_lea.probe import parse_experiment_metadata, probe_run

from . import fixtures as fx


@pytest.fixture
def study(tmp_path):
    """Two panel groups: A-runs carry AddX, B-runs carry AddY, both carry AddShared."""
    base = fx.base_gene_names(30)
    manifest_path, panel_csv, _ = fx.build_study(
        tmp_path,
        [
            {"run_id": "A1", "mouse_id": "M1", "condition": "AGED",
             "addons": ["AddShared", "AddX"]},
            {"run_id": "A2", "mouse_id": "M2", "condition": "AGED",
             "addons": ["AddShared", "AddX"]},
            {"run_id": "B1", "mouse_id": "M3", "condition": "ADULT",
             "addons": ["AddShared", "AddY"]},
            {"run_id": "B2", "mouse_id": "M4", "condition": "ADULT",
             "addons": ["AddShared", "AddY"]},
        ],
        base_genes=base,
        n_cells=120,
    )
    return manifest_path, panel_csv, base


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------

def test_probe_keeps_control_features(tmp_path):
    """Controls are the panel-independent quality signal — they must survive."""
    run = tmp_path / "R1"
    fx.make_run(run, genes=fx.base_gene_names(10), n_cells=50)
    manifest = RunManifest().add("R1", "M1", "AGED", run)

    p = probe_run(manifest[0], Findings())

    assert p.n_rna == 10
    # Counts are keyed by the *normalised* type, so a v4 and a v6 bundle report
    # the same thing (see test_features.py).
    assert p.feature_type_counts["negative_control_probe"] == 2
    assert p.feature_type_counts["negative_control_codeword"] == 1
    assert p.n_control_features == len(fx.CONTROL_FEATURES)
    assert p.n_strict_control_features == len(fx.STRICT_CONTROL_FEATURES)
    assert p.n_background_control_features == len(fx.BACKGROUND_CONTROL_FEATURES)
    # ...and controls are not mistaken for genes.
    assert not any(
        g.startswith(("NegControl", "Unassigned", "Deprecated")) for g in p.rna_genes
    )


def test_probe_reads_cell_qc_columns(tmp_path):
    run = tmp_path / "R1"
    fx.make_run(run, genes=fx.base_gene_names(10), n_cells=80)
    manifest = RunManifest().add("R1", "M1", "AGED", run)

    p = probe_run(manifest[0], Findings())

    assert p.n_cells == 80
    assert p.cells_source == "cells.parquet"
    for col in ("transcript_counts", "control_probe_counts", "cell_area",
                "nucleus_area"):
        assert p.has_cell_column(col), col


def test_probe_reads_older_csv_generation(tmp_path):
    run = tmp_path / "R1"
    fx.make_run(run, genes=fx.base_gene_names(10), n_cells=60,
                segmentation=fx.EXPANSION, cells_format="csv")
    manifest = RunManifest().add("R1", "M1", "AGED", run)

    p = probe_run(manifest[0], Findings())
    assert p.cells_source == "cells.csv.gz"
    assert p.n_cells == 60


def test_probe_flags_duplicate_gene_symbols(tmp_path):
    """A duplicate becomes a phantom add-on gene downstream — flag it here."""
    run = tmp_path / "R1"
    genes = fx.base_gene_names(10)
    fx.make_run(run, genes=genes + ["Base001"], n_cells=50)
    manifest = RunManifest().add("R1", "M1", "AGED", run)

    f = Findings()
    p = probe_run(manifest[0], f)

    assert p.duplicate_gene_symbols == ["Base001"]
    assert f.has("panel.duplicate_gene_symbol")


def test_missing_experiment_json_is_survivable(tmp_path):
    run = tmp_path / "R1"
    fx.make_run(run, genes=fx.base_gene_names(10), n_cells=40,
                include_experiment=False)
    manifest = RunManifest().add("R1", "M1", "AGED", run)

    f = Findings()
    manifest.validate(f)
    p = probe_run(manifest[0], f)

    assert p.ok  # the panel audit still works
    assert f.has("run.experiment_missing")


# ---------------------------------------------------------------------------
# experiment.xenium parsing
# ---------------------------------------------------------------------------

def test_experiment_parse_finds_nested_segmentation_keys(tmp_path):
    p = tmp_path / "experiment.xenium"
    p.write_text(json.dumps({
        "analysis_sw_version": "xenium-3.0.0",
        "panel_name": "Mouse Brain",
        "segmentation": {
            "method": "multimodal cell segmentation",
            "boundary_stain": "ATP1A1",
        },
    }))

    meta = parse_experiment_metadata(p)

    assert meta["analysis_sw_version"] == "xenium-3.0.0"
    assert meta["panel_name"] == "Mouse Brain"
    # Nesting must not hide the segmentation settings.
    assert "segmentation.method" in meta["segmentation_keys"]
    assert "segmentation.boundary_stain" in meta["segmentation_keys"]
    assert meta["parse_error"] is None


def test_experiment_parse_survives_malformed_json(tmp_path):
    p = tmp_path / "experiment.xenium"
    p.write_text("{not json")
    meta = parse_experiment_metadata(p)
    assert meta["parse_error"] is not None
    assert meta["panel_name"] is None


def test_experiment_parse_records_unrecognised_keys(tmp_path):
    """An unknown key must surface in raw_keys rather than vanish."""
    p = tmp_path / "experiment.xenium"
    p.write_text(json.dumps({"some_future_field": 42, "panel_name": "X"}))
    meta = parse_experiment_metadata(p)
    assert "some_future_field" in meta["raw_keys"]


# ---------------------------------------------------------------------------
# panel audit
# ---------------------------------------------------------------------------

def _probe_study(manifest_path):
    manifest = RunManifest.from_csv(manifest_path)
    f = Findings()
    probes = [probe_run(e, f) for e in manifest]
    return manifest, probes, f


def test_panel_groups_and_safe_gene_set(study):
    manifest_path, panel_csv, base = study
    _, probes, f = _probe_study(manifest_path)
    audit = audit_panels(probes, load_base_panel(panel_csv), findings=f)

    # A1/A2 share one add-on set, B1/B2 another.
    assert audit.n_panel_groups == 2
    assert audit.panel_groups["A1"] == audit.panel_groups["A2"]
    assert audit.panel_groups["B1"] == audit.panel_groups["B2"]
    assert audit.panel_groups["A1"] != audit.panel_groups["B1"]

    # Safe = base + the add-on every run carries. AddX/AddY are not safe.
    assert set(audit.safe_genes) == set(base) | {"AddShared"}
    assert "AddX" not in audit.safe_genes
    assert "AddY" not in audit.safe_genes

    assert f.has("panel.heterogeneous_addons")
    assert f.has("panel.safe_gene_set")


def test_addon_categories(study):
    manifest_path, panel_csv, _ = study
    _, probes, f = _probe_study(manifest_path)
    audit = audit_panels(probes, load_base_panel(panel_csv), findings=f)

    cats = audit.gene_categories().set_index("gene")
    assert cats.loc["AddShared", "category"] == "shared_all"
    assert cats.loc["AddShared", "n_runs"] == 4
    assert cats.loc["AddX", "category"] == "shared_partial"
    assert cats.loc["AddX", "n_runs"] == 2


def test_missing_base_gene_is_an_error(tmp_path):
    """
    The defect that passes silently elsewhere: a base gene absent from one run
    gets zero-filled and, in xenium-spatial, is never flagged.
    """
    base = fx.base_gene_names(20)
    panel_csv = fx.write_base_panel_csv(tmp_path / "panel.csv", base)

    fx.make_run(tmp_path / "R1", genes=base, n_cells=60, seed=1)
    fx.make_run(tmp_path / "R2", genes=base[:-1], n_cells=60, seed=2)  # one short

    manifest = (
        RunManifest()
        .add("R1", "M1", "AGED", tmp_path / "R1")
        .add("R2", "M2", "ADULT", tmp_path / "R2")
    )
    f = Findings()
    probes = [probe_run(e, f) for e in manifest]
    audit = audit_panels(probes, load_base_panel(panel_csv), findings=f)

    assert audit.missing_base["R2"] == [base[-1]]
    assert audit.missing_base["R1"] == []
    assert f.has("panel.base_incomplete")
    assert f.has_errors


def test_homogeneous_panels_produce_no_warning(tmp_path):
    base = fx.base_gene_names(15)
    manifest_path, panel_csv, _ = fx.build_study(
        tmp_path,
        [
            {"run_id": "R1", "mouse_id": "M1", "condition": "AGED",
             "addons": ["AddA"]},
            {"run_id": "R2", "mouse_id": "M2", "condition": "ADULT",
             "addons": ["AddA"]},
        ],
        base_genes=base,
        n_cells=60,
    )
    _, probes, f = _probe_study(manifest_path)
    audit = audit_panels(probes, load_base_panel(panel_csv), findings=f)

    assert audit.n_panel_groups == 1
    assert set(audit.safe_genes) == set(base) | {"AddA"}
    assert f.has("panel.homogeneous_addons")
    assert not f.has("panel.heterogeneous_addons")


def test_retention_table_spans_every_threshold(study):
    manifest_path, panel_csv, _ = study
    _, probes, f = _probe_study(manifest_path)
    audit = audit_panels(probes, load_base_panel(panel_csv), findings=f)

    ret = audit.retention.set_index("min_runs")
    assert list(ret.index) == [1, 2, 3, 4]
    assert ret.loc[1, "addon_genes_kept"] == 3      # AddShared, AddX, AddY
    assert ret.loc[4, "addon_genes_kept"] == 1      # only AddShared is everywhere
    # A stricter rule can never keep more genes.
    assert ret["addon_genes_kept"].is_monotonic_decreasing
