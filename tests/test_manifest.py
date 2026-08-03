"""Manifest parsing and the study-level checks it can make before touching data."""

from __future__ import annotations

import pandas as pd
import pytest

from xenium_lea.findings import Findings
from xenium_lea.manifest import RunManifest

from . import fixtures as fx


def _rows(tmp_path, specs):
    """Make real run dirs so the existence checks pass."""
    rows = []
    for i, (run_id, mouse, cond) in enumerate(specs):
        d = tmp_path / run_id
        fx.make_run(d, genes=fx.base_gene_names(8), n_cells=30, seed=i)
        rows.append(
            {"run_id": run_id, "mouse_id": mouse, "section_id": f"s{i}",
             "condition": cond, "run_dir": str(d)}
        )
    return rows


def test_from_csv_reads_required_columns(tmp_path):
    rows = _rows(tmp_path, [("R1", "M1", "AGED"), ("R2", "M2", "ADULT")])
    path = fx.write_manifest(tmp_path / "m.csv", rows)

    m = RunManifest.from_csv(path)

    assert len(m) == 2
    assert m.run_ids == ["R1", "R2"]
    assert m.mouse_ids == ["M1", "M2"]
    assert m[0].run_dir.exists()


def test_comment_and_blank_lines_are_skipped(tmp_path):
    """The shipped template has an explanatory comment block above the rows."""
    rows = _rows(tmp_path, [("R1", "M1", "AGED")])
    path = tmp_path / "m.csv"
    path.write_text(
        "run_id,mouse_id,section_id,condition,run_dir\n"
        "# this is a comment\n"
        "\n"
        f"R1,M1,s0,AGED,{rows[0]['run_dir']}\n"
    )

    m = RunManifest.from_csv(path)
    assert len(m) == 1
    assert m[0].run_id == "R1"


def test_missing_required_column_raises(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text("run_id,condition,run_dir\nR1,AGED,/tmp/x\n")
    with pytest.raises(ValueError, match="mouse_id"):
        RunManifest.from_csv(path)


def test_section_id_defaults_to_run_id(tmp_path):
    rows = _rows(tmp_path, [("R1", "M1", "AGED")])
    rows[0]["section_id"] = ""
    path = fx.write_manifest(tmp_path / "m.csv", rows)

    m = RunManifest.from_csv(path)
    assert m[0].section_id == "R1"


def test_unknown_columns_are_carried_as_overrides(tmp_path):
    rows = _rows(tmp_path, [("R1", "M1", "AGED")])
    rows[0]["segmentation_kit"] = "stain_kit"
    path = fx.write_manifest(tmp_path / "m.csv", rows,
                             extra_columns=["segmentation_kit"])

    m = RunManifest.from_csv(path)
    assert m[0].overrides == {"segmentation_kit": "stain_kit"}


def test_duplicate_run_id_is_an_error(tmp_path):
    rows = _rows(tmp_path, [("R1", "M1", "AGED"), ("R1", "M2", "ADULT")])
    path = fx.write_manifest(tmp_path / "m.csv", rows)

    f = RunManifest.from_csv(path).validate(Findings())
    assert f.has("manifest.duplicate_run_id")
    assert f.has_errors


def test_mouse_in_two_conditions_is_an_error(tmp_path):
    """One animal cannot be both aged and adult."""
    rows = _rows(tmp_path, [("R1", "M1", "AGED"), ("R2", "M1", "ADULT")])
    path = fx.write_manifest(tmp_path / "m.csv", rows)

    f = RunManifest.from_csv(path).validate(Findings())
    assert f.has("manifest.mouse_multiple_conditions")
    assert f.has_errors


def test_missing_run_directory_is_an_error(tmp_path):
    rows = _rows(tmp_path, [("R1", "M1", "AGED")])
    rows.append(
        {"run_id": "R2", "mouse_id": "M2", "section_id": "s1",
         "condition": "ADULT", "run_dir": str(tmp_path / "does_not_exist")}
    )
    path = fx.write_manifest(tmp_path / "m.csv", rows)

    f = RunManifest.from_csv(path).validate(Findings())
    assert f.has("run.dir_missing")


def test_sections_per_mouse_and_replicate_counts(tmp_path):
    rows = _rows(
        tmp_path,
        [("R1", "M1", "AGED"), ("R2", "M1", "AGED"),
         ("R3", "M2", "ADULT"), ("R4", "M3", "ADULT")],
    )
    path = fx.write_manifest(tmp_path / "m.csv", rows)
    m = RunManifest.from_csv(path)

    spm = m.sections_per_mouse().set_index("mouse_id")
    assert spm.loc["M1", "n_sections"] == 2
    assert spm.loc["M2", "n_sections"] == 1

    mpc = m.mice_per_condition().set_index("condition")
    # Two sections from one aged mouse is n=1, not n=2.
    assert mpc.loc["AGED", "n_mice"] == 1
    assert mpc.loc["AGED", "n_sections"] == 2
    assert mpc.loc["ADULT", "n_mice"] == 2


def test_multi_section_mice_are_reported(tmp_path):
    rows = _rows(
        tmp_path,
        [("R1", "M1", "AGED"), ("R2", "M1", "AGED"),
         ("R3", "M2", "ADULT"), ("R4", "M3", "ADULT")],
    )
    path = fx.write_manifest(tmp_path / "m.csv", rows)

    f = RunManifest.from_csv(path).validate(Findings())
    assert f.has("design.sections_per_mouse")


def test_single_mouse_condition_is_an_error(tmp_path):
    """Four sections from two animals still cannot support a group comparison."""
    rows = _rows(
        tmp_path,
        [("R1", "M1", "AGED"), ("R2", "M1", "AGED"),
         ("R3", "M2", "ADULT"), ("R4", "M2", "ADULT")],
    )
    path = fx.write_manifest(tmp_path / "m.csv", rows)

    f = RunManifest.from_csv(path).validate(Findings())
    assert f.has("design.single_mouse_condition")
    assert f.has_errors
