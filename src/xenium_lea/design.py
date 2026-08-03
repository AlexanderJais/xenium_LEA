"""
design.py
---------
Is the batch effect separable from the biology?

This is the question the rest of the audit exists to serve. Measuring a batch
effect is easy; knowing whether you are *allowed* to remove it is not. If the
segmentation kit was used on every AGED run and no ADULT run, then "segmentation
effect" and "ageing effect" are the same column of the design matrix. No
integration method can separate them — Harmony will happily produce a clean,
well-mixed embedding, and the ageing signal will have gone into the correction
along with the kit effect.

So for every technical factor T, the question is not "how big is T" but "does T
vary *within* each condition?":

    CONSTANT   T has one level in the whole study. No effect is possible.
    CROSSED    T varies within every condition. The T effect is estimable and
               can be adjusted for without touching the condition contrast.
    PARTIAL    T varies within some conditions but not all. Estimable, with
               reduced power and an imbalance worth reporting.
    ALIASED    T is constant within each condition but differs between them.
               T and condition are the same contrast. **Not estimable.**

Two further checks matter as much:

* **Factor-vs-factor aliasing.** If ``panel_group`` and ``segmentation_kit``
  move together, then even when both are CROSSED with condition you cannot
  attribute an effect to one rather than the other.
* **Within-mouse contrasts.** With several sections per mouse, a technical
  factor that differs between two sections of the *same animal* is a controlled
  comparison: same biology, differing only technically. These pairs are the most
  informative structure in the whole dataset for measuring a technical effect
  cleanly, and they are listed explicitly.

Nothing here is corrected or modelled away. The output is a verdict.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .findings import Findings

logger = logging.getLogger(__name__)

CONSTANT = "CONSTANT"
CROSSED = "CROSSED"
PARTIAL = "PARTIAL"
NESTED = "NESTED"
ALIASED = "ALIASED"

OVERALL_OK = "OK"
OVERALL_CAUTION = "CAUTION"
OVERALL_BLOCKED = "BLOCKED"

#: Columns treated as technical factors. ``mouse_id`` is excluded: it is nested
#: within condition by design, and that nesting is correct rather than a defect.
TECHNICAL_FACTORS = (
    "panel_group",
    "segmentation_kit",
    "analysis_sw_version",
    "instrument_sw_version",
    "instrument_sn",
    "run_date",
    "panel_name",
    "cells_source",
    "preservation_method",
)


# ---------------------------------------------------------------------------
# Association primitives
# ---------------------------------------------------------------------------

def determines(a: Iterable, b: Iterable) -> bool:
    """
    True when ``a`` determines ``b``: every level of ``a`` maps to exactly one
    level of ``b``. This is the exact, non-statistical notion of confounding —
    it does not depend on sample size.
    """
    mapping: dict[Any, Any] = {}
    for av, bv in zip(a, b):
        if av in mapping:
            if mapping[av] != bv:
                return False
        else:
            mapping[av] = bv
    return True


def is_aliased(a: Iterable, b: Iterable) -> bool:
    """True when ``a`` and ``b`` determine each other — an exact relabelling."""
    a, b = list(a), list(b)
    return determines(a, b) and determines(b, a)


def cramers_v(a: Iterable, b: Iterable) -> float:
    """
    Cramér's V between two categorical vectors.

    Reported alongside the exact aliasing check as a strength-of-association
    number for the partially-confounded cases; V == 1.0 means one factor is
    predictable from the other.
    """
    a, b = pd.Series(list(a), dtype="object"), pd.Series(list(b), dtype="object")
    table = pd.crosstab(a, b)
    if table.size == 0 or min(table.shape) < 2:
        return 0.0
    n = float(table.to_numpy().sum())
    if n == 0:
        return 0.0
    # chi-square without SciPy's correction, so V==1 for an exact relabelling.
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum(np.where(expected > 0, (table - expected) ** 2 / expected, 0.0))
    denom = n * (min(table.shape) - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else 0.0


def classify_factor(factor: Iterable, condition: Iterable) -> str:
    """
    Classify a technical factor against the biological condition.

    See the module docstring for what each verdict means. The order of the
    checks is the argument:

    1. One level overall -> CONSTANT.
    2. Constant within *every* condition -> ALIASED. The factor changes exactly
       where the condition changes, so the two are the same contrast.
    3. The factor determines the condition -> NESTED. It varies within a
       condition, but no level of it is shared *across* conditions, so there is
       no fixed factor level at which the conditions can be compared. Including
       it as a fixed effect absorbs the condition term entirely.
    4. Otherwise some level spans conditions, so the comparison is anchored:
       CROSSED if the factor varies within every condition, PARTIAL if not.
    """
    fac = pd.Series(list(factor), dtype="object")
    con = pd.Series(list(condition), dtype="object")

    if fac.nunique(dropna=False) <= 1:
        return CONSTANT

    varies = fac.groupby(con.values, dropna=False).nunique(dropna=False) > 1
    if not varies.any():
        return ALIASED
    if determines(fac.tolist(), con.tolist()):
        return NESTED
    return CROSSED if varies.all() else PARTIAL


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class FactorVerdict:
    """The separability verdict for one technical factor."""

    factor: str
    verdict: str
    n_levels: int
    levels: list[str]
    cramers_v_vs_condition: float
    levels_by_condition: dict[str, list[str]]
    crosstab: dict[str, dict[str, int]]
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor": self.factor,
            "verdict": self.verdict,
            "n_levels": self.n_levels,
            "levels": self.levels,
            "cramers_v_vs_condition": round(self.cramers_v_vs_condition, 4),
            "levels_by_condition": self.levels_by_condition,
            "crosstab": self.crosstab,
            "message": self.message,
        }


@dataclass
class DesignAudit:
    """The full design analysis."""

    factor_table: pd.DataFrame
    verdicts: dict[str, FactorVerdict] = field(default_factory=dict)
    factor_pairs: list[dict[str, Any]] = field(default_factory=list)
    within_mouse: list[dict[str, Any]] = field(default_factory=list)
    replicates: dict[str, dict[str, int]] = field(default_factory=dict)
    overall: str = OVERALL_OK
    condition_levels: list[str] = field(default_factory=list)

    def verdict_frame(self) -> pd.DataFrame:
        if not self.verdicts:
            return pd.DataFrame(
                columns=["factor", "verdict", "n_levels", "cramers_v_vs_condition",
                         "levels", "message"]
            )
        return pd.DataFrame(
            [
                {
                    "factor": v.factor,
                    "verdict": v.verdict,
                    "n_levels": v.n_levels,
                    "cramers_v_vs_condition": round(v.cramers_v_vs_condition, 4),
                    "levels": "; ".join(v.levels),
                    "message": v.message,
                }
                for v in self.verdicts.values()
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall,
            "condition_levels": self.condition_levels,
            "replicates": self.replicates,
            "factors": {k: v.to_dict() for k, v in self.verdicts.items()},
            "factor_pairs": self.factor_pairs,
            "within_mouse_contrasts": self.within_mouse,
            "n_runs": int(len(self.factor_table)),
            "n_mice": (
                int(self.factor_table["mouse_id"].nunique())
                if "mouse_id" in self.factor_table.columns
                else 0
            ),
        }


# ---------------------------------------------------------------------------
# Factor table assembly
# ---------------------------------------------------------------------------

def build_factor_table(
    probes,
    panel_groups: dict[str, str] | None = None,
    segmentation: pd.DataFrame | None = None,
    manifest=None,
) -> pd.DataFrame:
    """
    Assemble one row per run holding every factor the design analysis considers.

    Manifest override columns win over derived values but are recorded under a
    ``declared_`` prefix as well, so a disagreement stays visible.
    """
    rows = []
    seg_by_run: dict[str, str] = {}
    if segmentation is not None and not segmentation.empty:
        seg_by_run = dict(
            zip(segmentation["run_id"], segmentation["segmentation_kit"])
        )

    overrides_by_run: dict[str, dict[str, Any]] = {}
    if manifest is not None:
        overrides_by_run = {e.run_id: dict(e.overrides) for e in manifest}

    for p in probes:
        exp = p.experiment or {}
        row: dict[str, Any] = {
            "run_id": p.run_id,
            "mouse_id": p.mouse_id,
            "section_id": p.section_id,
            "condition": p.condition,
            "panel_group": (panel_groups or {}).get(p.run_id, "unknown"),
            "segmentation_kit": seg_by_run.get(p.run_id, "unknown"),
            "analysis_sw_version": _clean(exp.get("analysis_sw_version")),
            "instrument_sw_version": _clean(exp.get("instrument_sw_version")),
            "instrument_sn": _clean(exp.get("instrument_sn")),
            "run_date": _date_only(exp.get("run_start_time")),
            "panel_name": _clean(exp.get("panel_name")),
            "preservation_method": _clean(exp.get("preservation_method")),
            "cells_source": _clean(p.cells_source),
            "n_cells": p.n_cells,
            "n_rna_targets": p.n_rna,
        }
        for k, v in overrides_by_run.get(p.run_id, {}).items():
            if k in row and str(row[k]) not in ("unknown", "None", ""):
                row[f"declared_{k}"] = v
            row[k] = v
        rows.append(row)

    return pd.DataFrame(rows)


def _clean(v: Any) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "unknown"
    s = str(v).strip()
    return s if s else "unknown"


def _date_only(v: Any) -> str:
    """Bucket a run timestamp to its date — runs on one day are one batch."""
    if v is None:
        return "unknown"
    s = str(v).strip()
    if not s:
        return "unknown"
    for sep in ("T", " "):
        if sep in s:
            return s.split(sep, 1)[0]
    return s[:10] if len(s) >= 10 else s


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def audit_design(
    factor_table: pd.DataFrame,
    findings: Findings | None = None,
    factors: Iterable[str] = TECHNICAL_FACTORS,
) -> DesignAudit:
    """Produce the separability verdict."""
    f = findings if findings is not None else Findings()

    if factor_table.empty:
        f.error("design.no_runs", "No runs available for the design analysis.")
        return DesignAudit(factor_table=factor_table, overall=OVERALL_BLOCKED)

    condition = factor_table["condition"].astype(str)
    cond_levels = sorted(condition.unique())

    audit = DesignAudit(
        factor_table=factor_table,
        condition_levels=cond_levels,
        replicates=_replicate_counts(factor_table),
    )

    # -- factor vs condition ---------------------------------------------
    considered = [
        c for c in factors
        if c in factor_table.columns
        and factor_table[c].astype(str).nunique() > 0
    ]

    for col in considered:
        values = factor_table[col].astype(str)
        levels = sorted(values.unique())
        verdict = classify_factor(values, condition)
        v_stat = cramers_v(values, condition)

        by_cond = {
            str(c): sorted(values[condition == c].unique()) for c in cond_levels
        }
        ct = pd.crosstab(values, condition)
        crosstab = {
            str(idx): {str(c): int(ct.loc[idx, c]) for c in ct.columns}
            for idx in ct.index
        }

        message = _verdict_message(col, verdict, levels, by_cond)
        audit.verdicts[col] = FactorVerdict(
            factor=col,
            verdict=verdict,
            n_levels=len(levels),
            levels=levels,
            cramers_v_vs_condition=v_stat,
            levels_by_condition=by_cond,
            crosstab=crosstab,
            message=message,
        )

        if verdict == ALIASED:
            f.error(
                "design.factor_aliased",
                message,
                evidence={
                    "factor": col,
                    "levels_by_condition": by_cond,
                    "crosstab": crosstab,
                    "cramers_v": round(v_stat, 4),
                },
            )
        elif verdict == NESTED:
            f.warning(
                "design.factor_nested",
                message,
                evidence={
                    "factor": col,
                    "levels_by_condition": by_cond,
                    "crosstab": crosstab,
                    "cramers_v": round(v_stat, 4),
                },
            )
        elif verdict == PARTIAL:
            f.warning(
                "design.factor_partial",
                message,
                evidence={
                    "factor": col,
                    "levels_by_condition": by_cond,
                    "crosstab": crosstab,
                    "cramers_v": round(v_stat, 4),
                },
            )
        elif verdict == CROSSED:
            f.info("design.factor_crossed", message,
                   evidence={"factor": col, "levels_by_condition": by_cond})

    # -- factor vs factor --------------------------------------------------
    varying = [
        c for c in considered
        if audit.verdicts[c].verdict != CONSTANT
    ]
    for i, a in enumerate(varying):
        for b in varying[i + 1:]:
            va = factor_table[a].astype(str)
            vb = factor_table[b].astype(str)
            aliased = is_aliased(va, vb)
            pair = {
                "factor_a": a,
                "factor_b": b,
                "aliased": bool(aliased),
                "a_determines_b": bool(determines(va, vb)),
                "b_determines_a": bool(determines(vb, va)),
                "cramers_v": round(cramers_v(va, vb), 4),
            }
            audit.factor_pairs.append(pair)
            if aliased:
                f.warning(
                    "design.factors_aliased",
                    f"'{a}' and '{b}' are perfectly aliased with each other — "
                    "they change together across every run. Even where each is "
                    "separable from condition, an effect attributed to one could "
                    "equally be the other; they cannot be told apart in this "
                    "dataset.",
                    evidence=pair,
                )

    # -- within-mouse contrasts --------------------------------------------
    audit.within_mouse = _within_mouse_contrasts(factor_table, varying)
    if audit.within_mouse:
        f.info(
            "design.within_mouse_contrast",
            f"{len(audit.within_mouse)} within-mouse technical contrast(s) "
            "found — sections of the same animal that differ in "
            + ", ".join(sorted({c["factor"] for c in audit.within_mouse}))
            + ". These are controlled comparisons (same biology, differing only "
            "technically) and are the cleanest way to measure the size of a "
            "technical effect in this study.",
            evidence={"contrasts": audit.within_mouse},
        )
    elif varying:
        f.info(
            "design.no_within_mouse_contrast",
            "No technical factor varies between two sections of the same mouse, "
            "so there is no controlled comparison available for measuring a "
            "technical effect independently of biology.",
        )

    # -- overall -----------------------------------------------------------
    # The verdict is about one thing: whether the *biological* contrast is
    # separable from the technical factors. Two technical factors being aliased
    # with each other is reported above but does not enter here — it means an
    # effect cannot be attributed to one of them rather than the other, which
    # never threatens the condition comparison.
    verdict_values = {v.verdict for v in audit.verdicts.values()}
    if ALIASED in verdict_values:
        audit.overall = OVERALL_BLOCKED
    elif PARTIAL in verdict_values or NESTED in verdict_values:
        audit.overall = OVERALL_CAUTION
    else:
        audit.overall = OVERALL_OK

    _report_overall(audit, f)
    return audit


def _replicate_counts(table: pd.DataFrame) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for cond, sub in table.groupby(table["condition"].astype(str)):
        out[str(cond)] = {
            "n_mice": int(sub["mouse_id"].nunique()),
            "n_sections": int(len(sub)),
        }
    return out


def _within_mouse_contrasts(
    table: pd.DataFrame, factors: Iterable[str]
) -> list[dict[str, Any]]:
    """Find technical factors that vary between sections of the same mouse."""
    out: list[dict[str, Any]] = []
    for mouse, sub in table.groupby(table["mouse_id"].astype(str)):
        if len(sub) < 2:
            continue
        for col in factors:
            vals = sub[col].astype(str)
            if vals.nunique() > 1:
                out.append(
                    {
                        "mouse_id": str(mouse),
                        "condition": str(sub["condition"].iloc[0]),
                        "factor": col,
                        "levels": sorted(vals.unique()),
                        "runs": {
                            str(r.run_id): str(getattr(r, col))
                            for r in sub.itertuples()
                        },
                    }
                )
    return out


def _verdict_message(
    factor: str, verdict: str, levels: list[str], by_cond: dict[str, list[str]]
) -> str:
    shown = ", ".join(levels[:6]) + (" ..." if len(levels) > 6 else "")
    spread = "; ".join(f"{c}: {', '.join(v)}" for c, v in by_cond.items())

    if verdict == CONSTANT:
        return (
            f"'{factor}' is constant across the study ({shown}) — it cannot "
            "contribute a batch effect."
        )
    if verdict == CROSSED:
        return (
            f"'{factor}' varies within every condition ({spread}). Its effect is "
            "estimable and can be adjusted for without touching the condition "
            "contrast."
        )
    if verdict == PARTIAL:
        return (
            f"'{factor}' varies within some conditions but not all ({spread}). "
            "Its effect is estimable but with reduced power and an unbalanced "
            "design; the imbalance biases any naive comparison."
        )
    if verdict == NESTED:
        return (
            f"'{factor}' varies within a condition, but no level of it is shared "
            f"across conditions ({spread}) — it is nested within condition. "
            "There is no fixed level of "
            f"'{factor}' at which the conditions can be compared, so including "
            "it as a fixed effect would absorb the condition term entirely. It "
            "is not correctable, but because it has replication within a "
            "condition its variance *is* estimable: treat it as a random effect "
            "and test the condition difference against that "
            f"{factor}-to-{factor} variation."
        )
    return (
        f"'{factor}' is constant within each condition but differs between them "
        f"({spread}). It is perfectly confounded with condition: the two are the "
        "same contrast, so the effect of "
        f"'{factor}' cannot be estimated or removed without removing the "
        "biological difference along with it."
    )


def _report_overall(audit: DesignAudit, f: Findings) -> None:
    aliased = [k for k, v in audit.verdicts.items() if v.verdict == ALIASED]
    partial = [k for k, v in audit.verdicts.items() if v.verdict == PARTIAL]
    nested = [k for k, v in audit.verdicts.items() if v.verdict == NESTED]

    if audit.overall == OVERALL_BLOCKED:
        f.error(
            "design.verdict_blocked",
            "SEPARABILITY VERDICT: BLOCKED. "
            f"{len(aliased)} technical factor(s) are perfectly confounded with "
            f"condition ({', '.join(aliased)}). Any between-condition difference "
            "you measure is a mixture of biology and these technical factors, in "
            "unknown proportion, and no batch-correction method can separate "
            "them — correcting removes the biology too. Options, in order of "
            "preference: (1) find or generate runs that break the confound "
            "(e.g. re-run one animal per group with the other segmentation "
            "method); (2) restrict claims to effects robust within a single "
            "level of the confounded factor; (3) report the comparison as "
            "confounded. Note that within-mouse contrasts, if present, still let "
            "you measure the size of the technical effect even though they "
            "cannot remove it from the condition comparison.",
            evidence={
                "aliased_factors": aliased,
                "details": {k: audit.verdicts[k].to_dict() for k in aliased},
            },
        )
    elif audit.overall == OVERALL_CAUTION:
        parts = []
        if partial:
            parts.append(
                f"{', '.join(partial)} "
                f"{'is' if len(partial) == 1 else 'are'} unbalanced across "
                "conditions — adjust for "
                f"{'it' if len(partial) == 1 else 'them'} explicitly rather than "
                "relying on an unsupervised integration step"
            )
        if nested:
            parts.append(
                f"{', '.join(nested)} "
                f"{'is' if len(nested) == 1 else 'are'} nested within condition "
                "— not correctable as a fixed effect, but with replication "
                "inside each condition, so test the condition difference against "
                "that variation using a random effect"
            )
        f.warning(
            "design.verdict_caution",
            "SEPARABILITY VERDICT: CAUTION. No technical factor is perfectly "
            "confounded with condition, so the biological contrast is "
            "recoverable — but " + "; and ".join(parts) + ". Check in either "
            "case that the condition difference survives the adjustment.",
            evidence={
                "partial_factors": partial,
                "nested_factors": nested,
                "factor_pairs": audit.factor_pairs,
            },
        )
    else:
        f.info(
            "design.verdict_ok",
            "SEPARABILITY VERDICT: OK. Every technical factor either is constant "
            "across the study or varies within every condition, so no technical "
            "effect is confounded with the biological contrast. Batch effects "
            "here are a matter of statistical efficiency, not validity.",
        )
