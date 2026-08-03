"""
The separability engine — the highest-value tests in the suite.

Everything else the audit reports is a description. This is the part that says
whether the study's biological comparison is recoverable at all, so each verdict
is pinned against a hand-built design whose right answer is known by
construction.
"""

from __future__ import annotations

import pandas as pd
import pytest

from xenium_lea.design import (
    ALIASED,
    CONSTANT,
    CROSSED,
    NESTED,
    OVERALL_BLOCKED,
    OVERALL_CAUTION,
    OVERALL_OK,
    PARTIAL,
    PER_RUN,
    audit_design,
    classify_factor,
    cramers_v,
    determines,
    is_aliased,
)
from xenium_lea.findings import Findings


def _table(rows):
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def test_determines_is_directional():
    # Fine-grained -> coarse is a function; the reverse is not.
    fine = ["a1", "a2", "b1", "b2"]
    coarse = ["A", "A", "B", "B"]
    assert determines(fine, coarse)
    assert not determines(coarse, fine)
    assert not is_aliased(fine, coarse)


def test_is_aliased_detects_exact_relabelling():
    assert is_aliased(["x", "x", "y", "y"], ["1", "1", "2", "2"])
    assert not is_aliased(["x", "x", "y", "y"], ["1", "2", "1", "2"])


def test_cramers_v_is_one_for_exact_relabelling_and_zero_when_balanced():
    assert cramers_v(["x", "x", "y", "y"], ["1", "1", "2", "2"]) == pytest.approx(1.0)
    assert cramers_v(["x", "y", "x", "y"], ["1", "1", "2", "2"]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# classify_factor
# ---------------------------------------------------------------------------

def test_constant_factor():
    assert classify_factor(["k"] * 4, ["A", "A", "B", "B"]) == CONSTANT


def test_crossed_factor_varies_within_every_condition():
    #   condition:  A  A  B  B
    #   kit:        1  2  1  2      -> fully crossed
    assert classify_factor(["1", "2", "1", "2"], ["A", "A", "B", "B"]) == CROSSED


def test_aliased_factor_changes_exactly_where_condition_does():
    #   condition:  A  A  B  B
    #   kit:        1  1  2  2      -> perfectly confounded
    assert classify_factor(["1", "1", "2", "2"], ["A", "A", "B", "B"]) == ALIASED


def test_partial_factor_varies_in_one_condition_only():
    #   condition:  A  A  B  B
    #   kit:        1  2  2  2      -> varies within A, constant within B
    assert classify_factor(["1", "2", "2", "2"], ["A", "A", "B", "B"]) == PARTIAL


def test_factor_nested_within_condition():
    # The factor varies within each condition, but no level of it is shared
    # across conditions, so there is no fixed level at which A and B can be
    # compared. Not CROSSED — more levels does not buy separability.
    assert classify_factor(
        ["a1", "a1", "a2", "b1", "b1", "b2"],
        ["A", "A", "A", "B", "B", "B"],
    ) == NESTED


def test_factor_with_one_level_per_run_is_not_a_batch_factor():
    """
    A factor that uniquely identifies each run labels the sample rather than
    grouping samples. Treating it as confounded would mark every study where
    each section was its own instrument run as unrecoverable.
    """
    assert classify_factor(
        ["r1", "r2", "r3", "r4"], ["A", "A", "B", "B"]
    ) == PER_RUN


def test_factor_coarser_than_condition_is_aliased():
    # Constant within every condition, changing only where condition changes —
    # a sub-contrast of condition, however few levels it has.
    assert classify_factor(
        ["1", "1", "1", "1", "2", "2"], ["A", "A", "B", "B", "C", "C"]
    ) == ALIASED


# ---------------------------------------------------------------------------
# audit_design — the real scenario the study is in
# ---------------------------------------------------------------------------

def _study(kits, panels, conditions, mice=None):
    n = len(conditions)
    mice = mice or [f"M{i}" for i in range(n)]
    return _table(
        [
            {
                "run_id": f"R{i}",
                "mouse_id": mice[i],
                "section_id": f"s{i}",
                "condition": conditions[i],
                "segmentation_kit": kits[i],
                "panel_group": panels[i],
            }
            for i in range(n)
        ]
    )


def test_blocked_when_segmentation_kit_tracks_condition():
    """The scenario that motivates this whole tool."""
    table = _study(
        kits=["stain_kit", "stain_kit", "nucleus_expansion", "nucleus_expansion"],
        panels=["P1"] * 4,
        conditions=["AGED", "AGED", "ADULT", "ADULT"],
    )
    f = Findings()
    audit = audit_design(table, findings=f)

    assert audit.overall == OVERALL_BLOCKED
    assert audit.verdicts["segmentation_kit"].verdict == ALIASED
    assert f.has("design.factor_aliased")
    assert f.has("design.verdict_blocked")
    assert f.has_errors


def test_ok_when_kit_is_crossed_with_condition():
    table = _study(
        kits=["stain_kit", "nucleus_expansion", "stain_kit", "nucleus_expansion"],
        panels=["P1"] * 4,
        conditions=["AGED", "AGED", "ADULT", "ADULT"],
    )
    f = Findings()
    audit = audit_design(table, findings=f)

    assert audit.overall == OVERALL_OK
    assert audit.verdicts["segmentation_kit"].verdict == CROSSED
    assert not f.has_errors
    assert f.has("design.verdict_ok")


def test_caution_when_panel_is_unbalanced_but_not_confounded():
    table = _study(
        kits=["stain_kit"] * 4,
        panels=["P1", "P2", "P2", "P2"],
        conditions=["AGED", "AGED", "ADULT", "ADULT"],
    )
    f = Findings()
    audit = audit_design(table, findings=f)

    assert audit.overall == OVERALL_CAUTION
    assert audit.verdicts["panel_group"].verdict == PARTIAL
    assert not f.has_errors


def test_nested_factor_gives_caution_not_blocked():
    """
    Each condition was run on its own two dates. The date effect cannot be
    adjusted for as a fixed effect, but with two dates per condition its
    variance is estimable — that is caution, not a dead end.
    """
    # Six runs over four dates: two dates inside each condition, none shared
    # across them. Fewer levels than runs, so this is real nesting rather than
    # a per-run label.
    dates = ["2024-01-05", "2024-01-05", "2024-01-06",
             "2024-02-11", "2024-02-11", "2024-02-12"]
    conds = ["AGED"] * 3 + ["ADULT"] * 3
    table = _table(
        [
            {"run_id": f"R{i + 1}", "mouse_id": f"M{i + 1}", "section_id": "s1",
             "condition": conds[i], "run_date": dates[i],
             "segmentation_kit": "stain_kit", "panel_group": "P1"}
            for i in range(6)
        ]
    )
    f = Findings()
    audit = audit_design(table, findings=f)

    assert audit.verdicts["run_date"].verdict == NESTED
    assert audit.overall == OVERALL_CAUTION
    assert f.has("design.factor_nested")
    assert f.has("design.verdict_caution")
    # Nesting is a warning, not an error: the study is still analysable.
    assert not f.has_errors
    # The message must point at the route that does work.
    assert "random effect" in audit.verdicts["run_date"].message


def test_factors_aliased_with_each_other_are_flagged():
    """Kit and panel both crossed with condition, but identical to each other."""
    table = _study(
        kits=["stain_kit", "nucleus_expansion", "stain_kit", "nucleus_expansion"],
        panels=["P1", "P2", "P1", "P2"],
        conditions=["AGED", "AGED", "ADULT", "ADULT"],
    )
    f = Findings()
    audit = audit_design(table, findings=f)

    pair = next(
        p for p in audit.factor_pairs
        if {p["factor_a"], p["factor_b"]} == {"segmentation_kit", "panel_group"}
    )
    assert pair["aliased"] is True
    assert f.has("design.factors_aliased")
    # Both are separately crossed with condition, so the biological contrast is
    # untouched: two technical factors moving together only means an effect
    # cannot be attributed to one rather than the other.
    assert audit.overall == OVERALL_OK


def test_within_mouse_contrast_is_found():
    """Two sections of one animal differing only in segmentation kit."""
    table = _table(
        [
            {"run_id": "R1", "mouse_id": "M1", "section_id": "s1",
             "condition": "AGED", "segmentation_kit": "stain_kit",
             "panel_group": "P1"},
            {"run_id": "R2", "mouse_id": "M1", "section_id": "s2",
             "condition": "AGED", "segmentation_kit": "nucleus_expansion",
             "panel_group": "P1"},
            {"run_id": "R3", "mouse_id": "M2", "section_id": "s1",
             "condition": "ADULT", "segmentation_kit": "stain_kit",
             "panel_group": "P1"},
            {"run_id": "R4", "mouse_id": "M2", "section_id": "s2",
             "condition": "ADULT", "segmentation_kit": "nucleus_expansion",
             "panel_group": "P1"},
        ]
    )
    f = Findings()
    audit = audit_design(table, findings=f)

    assert len(audit.within_mouse) == 2
    assert {c["mouse_id"] for c in audit.within_mouse} == {"M1", "M2"}
    assert all(c["factor"] == "segmentation_kit" for c in audit.within_mouse)
    assert f.has("design.within_mouse_contrast")
    # Split sections make the kit crossed with condition, so nothing is blocked.
    assert audit.overall == OVERALL_OK


def test_no_within_mouse_contrast_when_each_mouse_is_internally_uniform():
    table = _study(
        kits=["stain_kit", "nucleus_expansion", "stain_kit", "nucleus_expansion"],
        panels=["P1"] * 4,
        conditions=["AGED", "AGED", "ADULT", "ADULT"],
        mice=["M1", "M2", "M3", "M4"],
    )
    f = Findings()
    audit = audit_design(table, findings=f)
    assert audit.within_mouse == []
    assert f.has("design.no_within_mouse_contrast")


def test_replicate_counts_are_at_the_mouse_level():
    """Four sections from two mice is n=1 per group, not n=2."""
    table = _table(
        [
            {"run_id": "R1", "mouse_id": "M1", "section_id": "s1",
             "condition": "AGED", "segmentation_kit": "stain_kit",
             "panel_group": "P1"},
            {"run_id": "R2", "mouse_id": "M1", "section_id": "s2",
             "condition": "AGED", "segmentation_kit": "stain_kit",
             "panel_group": "P1"},
            {"run_id": "R3", "mouse_id": "M2", "section_id": "s1",
             "condition": "ADULT", "segmentation_kit": "stain_kit",
             "panel_group": "P1"},
            {"run_id": "R4", "mouse_id": "M2", "section_id": "s2",
             "condition": "ADULT", "segmentation_kit": "stain_kit",
             "panel_group": "P1"},
        ]
    )
    audit = audit_design(table, findings=Findings())
    assert audit.replicates["AGED"] == {"n_mice": 1, "n_sections": 2}
    assert audit.replicates["ADULT"] == {"n_mice": 1, "n_sections": 2}


def test_verdict_serialises_to_json_friendly_dict():
    table = _study(
        kits=["stain_kit", "stain_kit", "nucleus_expansion", "nucleus_expansion"],
        panels=["P1"] * 4,
        conditions=["AGED", "AGED", "ADULT", "ADULT"],
    )
    d = audit_design(table, findings=Findings()).to_dict()

    import json

    json.dumps(d)  # must not raise
    assert d["overall"] == OVERALL_BLOCKED
    assert d["factors"]["segmentation_kit"]["verdict"] == ALIASED
    assert d["n_runs"] == 4
