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
import re
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

#: A factor with a distinct level for every run. It identifies the sample rather
#: than grouping samples, so it carries no batch information: its "effect" is
#: just the residual, and it cannot be adjusted for without fitting one
#: parameter per observation. Reported, but kept out of the verdict — otherwise
#: any study where each section was its own instrument run would read as
#: confounded on that ground alone.
PER_RUN = "PER_RUN"

#: A factor that groups the runs, in a study with no biological contrast to
#: test it against. Says nothing about separability — there is nothing to
#: separate from — but it is still the technical block structure, and it is what
#: will decide the verdict once conditions are known.
GROUPING = "GROUPING"

#: A factor whose value is unknown for most runs. The split it produces tracks
#: which files were available, not anything about the experiment, so it is
#: reported and then kept out of the verdict and out of the alias clusters.
#: Without this, a partially-assembled study reports a confident batch boundary
#: that is really the boundary between uploaded and not-yet-uploaded bundles.
UNKNOWN_DOMINATED = "UNKNOWN_DOMINATED"

#: Placeholder written by ``build_factor_table`` when a field cannot be read.
UNKNOWN_LEVEL = "unknown"

#: Fraction of runs that may be ``unknown`` before a factor is disqualified.
MAX_UNKNOWN_FRACTION = 0.25

OVERALL_OK = "OK"
OVERALL_CAUTION = "CAUTION"
OVERALL_BLOCKED = "BLOCKED"

#: Fewer than two conditions: the separability question is not yet answerable.
#: Reported rather than guessed, because a single-level condition column makes
#: every factor look trivially nested within it.
OVERALL_NO_CONTRAST = "NO_CONTRAST"

#: Columns treated as technical factors. ``mouse_id`` is excluded: it is nested
#: within condition by design, and that nesting is correct rather than a defect.
TECHNICAL_FACTORS = (
    "panel_group",
    "panel_design_id",
    "segmentation_kit",
    "run_name",
    "analysis_sw_version",
    "instrument_sw_version",
    "instrument_sn",
    "chemistry_version",
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

    n_levels = fac.nunique(dropna=False)
    if n_levels <= 1:
        return CONSTANT
    if n_levels == len(fac) and len(fac) > 1:
        # One level per run: this labels the sample, it does not group samples.
        return PER_RUN
    if con.nunique(dropna=False) <= 1:
        # No biological contrast, so nothing to be confounded *with*. Without
        # this guard every factor determines the (constant) condition and would
        # be reported as nested within it — true but vacuous.
        return GROUPING

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
    alias_clusters: list[list[str]] = field(default_factory=list)
    covariates: list[str] = field(default_factory=list)
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
            "alias_clusters": self.alias_clusters,
            "covariates": self.covariates,
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
            # panel_design_id names the *custom add-on* design. Two runs can
            # share panel_name and predesigned base and still carry different
            # add-on genes, so this discriminates where panel_name cannot — and
            # it works from metrics_summary.csv alone, without any gene list.
            "panel_design_id": _clean(exp.get("panel_design_id")),
            "segmentation_kit": seg_by_run.get(p.run_id, "unknown"),
            # The instrument run is the batch in the ordinary sense: sections
            # processed together share reagents, operator and machine state.
            "run_name": _clean(exp.get("run_name")),
            "analysis_sw_version": _clean(exp.get("analysis_sw_version")),
            "instrument_sw_version": _clean(exp.get("instrument_sw_version")),
            "instrument_sn": _clean(exp.get("instrument_sn")),
            "chemistry_version": _clean(exp.get("chemistry_version")),
            "run_date": _date_only(
                exp.get("run_start_time") or _date_from_run_name(exp.get("run_name"))
            ),
            "panel_name": _clean(exp.get("panel_name")),
            "panel_predesigned_id": _clean(exp.get("panel_predesigned_id")),
            "preservation_method": _clean(exp.get("preservation_method")),
            "cells_source": _clean(p.cells_source),
            "region_name": _clean(exp.get("region_name")),
            "n_cells": p.n_cells or _int_or_none(exp.get("num_cells")),
            "n_rna_targets": p.n_rna,
            # Continuous technical covariates. Not classified as factors — they
            # have a level per run — but they belong in the inventory because
            # each shifts downstream numbers and none is visible in the matrix.
            "stain_frac": _num_or_none(exp.get("frac_stain")),
            "section_thickness": _num_or_none(exp.get("section_thickness")),
            "transcript_density": _num_or_none(exp.get("transcript_density")),
            "frac_transcripts_assigned": _num_or_none(
                exp.get("fraction_transcripts_assigned")
            ),
            "total_cell_area": _num_or_none(exp.get("total_cell_area")),
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


_RUN_NAME_DATE_RE = re.compile(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})")


def _date_from_run_name(run_name: Any) -> str | None:
    """
    Recover the run date from a run name like ``20250626_Xv1_Lea_Run1``.

    ``metrics_summary.csv`` carries no timestamp, so for a metrics-only run this
    is the only way to place it in time — and run date is often the factor that
    separates one processing batch from another.
    """
    if not run_name:
        return None
    m = _RUN_NAME_DATE_RE.search(str(run_name))
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def _num_or_none(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else round(f, 6)


def _int_or_none(v: Any) -> int | None:
    n = _num_or_none(v)
    return None if n is None else int(n)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def audit_design(
    factor_table: pd.DataFrame,
    findings: Findings | None = None,
    factors: Iterable[str] = TECHNICAL_FACTORS,
    covariates: Iterable[str] = (),
) -> DesignAudit:
    """
    Produce the separability verdict.

    ``covariates`` are extra columns supplied in the manifest — sex, age, batch
    of surgery, anything the experimenter tracked. They are analysed alongside
    the technical factors but carry a distinct risk: a *biological* covariate
    that is indistinguishable from a technical factor means that biological
    question cannot be asked of this dataset at all, whatever happens to the
    primary contrast.
    """
    f = findings if findings is not None else Findings()
    covariates = [c for c in covariates if c in factor_table.columns]

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
    unknown_dominated: dict[str, int] = {}
    considered = [
        c for c in list(factors) + list(covariates)
        if c in factor_table.columns
        and factor_table[c].astype(str).nunique() > 0
    ]
    audit.covariates = list(covariates)

    for col in considered:
        values = factor_table[col].astype(str)
        levels = sorted(values.unique())
        n_unknown = int((values == UNKNOWN_LEVEL).sum())

        if (
            n_unknown
            and len(levels) > 1
            and n_unknown > MAX_UNKNOWN_FRACTION * len(values)
        ):
            # Unknown for most runs: the split reflects which files were
            # uploaded, not the experiment.
            verdict = UNKNOWN_DOMINATED
        else:
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

        if col in covariates and verdict in (NESTED, CROSSED, PARTIAL):
            # Covariate structure is reported through the covariate findings.
            f.info(
                "design.covariate_structure",
                f"Covariate '{col}' vs condition: {verdict}. " + message,
                evidence={"covariate": col, "verdict": verdict,
                          "levels_by_condition": by_cond},
            )
        elif verdict == ALIASED:
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
        elif verdict == PER_RUN:
            f.info(
                "design.factor_per_run",
                message,
                evidence={"factor": col, "n_levels": len(levels)},
            )
        elif verdict == UNKNOWN_DOMINATED:
            unknown_dominated[col] = n_unknown
        elif verdict == GROUPING:
            f.info(
                "design.factor_grouping",
                message,
                evidence={"factor": col, "levels": levels,
                          "runs_per_level": {
                              str(k): int(v) for k, v in
                              values.value_counts().items()}},
            )

    if unknown_dominated:
        # One fact, however many fields it touches: these all come from the same
        # missing files.
        f.warning(
            "design.factors_unknown_dominated",
            f"{len(unknown_dominated)} factor(s) are unknown for most runs and "
            f"are excluded from the verdict: {', '.join(sorted(unknown_dominated))}. "
            "The groups they form track which files have been uploaded rather "
            "than anything about the experiment, so treating them as batch "
            "factors would invent a boundary. Supplying the missing "
            "experiment.xenium files makes them informative.",
            evidence={"factors": unknown_dominated, "n_runs": len(factor_table)},
        )

    # -- factor vs factor --------------------------------------------------
    varying = [
        c for c in considered
        if audit.verdicts[c].verdict
        not in (CONSTANT, PER_RUN, UNKNOWN_DOMINATED)
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

    # Aliasing is transitive, so report equivalence *classes* rather than every
    # pair: eleven factors moving together is one fact about the study, not
    # fifty-five separate warnings.
    _report_alias_clusters(audit, f)
    _report_covariate_confounding(audit, f, covariates)

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
    # The overall verdict answers one question: are the *technical* effects
    # separable from the biology? A supplied covariate nested within condition
    # is biology — age in weeks nested within aged/adult is the definition of
    # the groups, not a confound — so covariates are reported on their own terms
    # (including the aliased-with-technical error above) and kept out of here.
    verdict_values = {
        v.verdict for k, v in audit.verdicts.items() if k not in set(covariates)
    }
    if len(cond_levels) < 2:
        audit.overall = OVERALL_NO_CONTRAST
    elif ALIASED in verdict_values:
        audit.overall = OVERALL_BLOCKED
    elif PARTIAL in verdict_values or NESTED in verdict_values:
        audit.overall = OVERALL_CAUTION
    else:
        audit.overall = OVERALL_OK

    _report_overall(audit, f)
    return audit


def _alias_clusters(pairs: list[dict[str, Any]]) -> list[list[str]]:
    """
    Group factors into sets that are mutually indistinguishable.

    Aliasing is an equivalence relation, so the pairs collapse to connected
    components — union-find over the aliased edges.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for p in pairs:
        if p["aliased"]:
            union(p["factor_a"], p["factor_b"])

    groups: dict[str, list[str]] = {}
    for node in parent:
        groups.setdefault(find(node), []).append(node)
    return [sorted(g) for g in groups.values() if len(g) > 1]


def _report_alias_clusters(audit: DesignAudit, f: Findings) -> None:
    """One finding per set of mutually indistinguishable factors."""
    clusters = _alias_clusters(audit.factor_pairs)
    audit.alias_clusters = clusters
    if not clusters:
        return

    for group in sorted(clusters, key=len, reverse=True):
        levels = {
            g: audit.verdicts[g].levels for g in group if g in audit.verdicts
        }
        f.warning(
            "design.factors_aliased",
            f"{len(group)} factors change together across every run and are "
            f"therefore indistinguishable: {', '.join(group)}. They partition "
            "the runs the same way, so an effect attributed to one could equally "
            "be any of the others — no analysis of this dataset can say which. "
            "In practice this is one batch boundary wearing several names; treat "
            "it as a single factor and be explicit about that in any writeup.",
            evidence={"factors": group, "levels": levels},
        )


def _report_covariate_confounding(
    audit: DesignAudit, f: Findings, covariates: Iterable[str]
) -> None:
    """
    Flag a supplied covariate that is indistinguishable from a technical factor.

    This is a different failure from a confounded primary contrast, and easy to
    miss because the primary contrast can be perfectly clean while it holds. If
    every male was processed in one batch and every female in another, then a
    sex difference and a batch difference are the same contrast: that question
    is unanswerable from this dataset no matter how well the main comparison
    behaves.
    """
    covariates = [c for c in covariates if c in audit.verdicts]
    if not covariates:
        return

    technical = set(TECHNICAL_FACTORS)
    for cov in covariates:
        partners = sorted(
            {
                p["factor_b"] if p["factor_a"] == cov else p["factor_a"]
                for p in audit.factor_pairs
                if p["aliased"] and cov in (p["factor_a"], p["factor_b"])
            }
            & technical
        )
        if not partners:
            continue
        f.error(
            "design.covariate_aliased_with_technical",
            f"The covariate '{cov}' is perfectly aliased with technical "
            f"factor(s): {', '.join(partners)}. Every level of '{cov}' was "
            "processed under a different technical condition, so a "
            f"'{cov}' effect and a technical effect are the same contrast. Any "
            f"analysis of '{cov}' in this dataset — including stratifying or "
            "adjusting by it — would report the batch difference under that "
            "name. This is separate from the primary contrast, which may be "
            "perfectly sound.",
            evidence={
                "covariate": cov,
                "aliased_with": partners,
                "levels_by_condition": audit.verdicts[cov].levels_by_condition,
                "crosstab": audit.verdicts[cov].crosstab,
            },
        )


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
    if verdict == UNKNOWN_DOMINATED:
        return (
            f"'{factor}' is unknown for most runs, so the groups it forms track "
            "which files have been uploaded rather than anything about the "
            "experiment. Excluded from the verdict and from the aliasing "
            "analysis; supply the missing experiment.xenium files to make it "
            "informative."
        )
    if verdict == GROUPING:
        return (
            f"'{factor}' splits the runs into {len(levels)} group(s) ({shown}). "
            "Whether that threatens the analysis depends entirely on how the "
            "biological conditions fall across those groups — supply the "
            "condition column to get a separability verdict."
        )
    if verdict == PER_RUN:
        return (
            f"'{factor}' has a distinct value for every run ({len(levels)} "
            "levels). It labels the sample rather than grouping samples, so it "
            "carries no batch information — its 'effect' is the residual, and "
            "adjusting for it would fit one parameter per observation. Excluded "
            "from the verdict."
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
    elif audit.overall == OVERALL_NO_CONTRAST:
        grouping = [k for k, v in audit.verdicts.items() if v.verdict == GROUPING]
        f.warning(
            "design.verdict_no_contrast",
            "SEPARABILITY VERDICT: NOT YET ANSWERABLE. Every run carries the "
            "same condition label, so there is no biological contrast to protect "
            "and nothing can be confounded with it. The technical block "
            "structure is reported above"
            + (f" ({', '.join(grouping)} each group the runs)" if grouping else "")
            + " — fill in the condition column and re-run to find out whether "
            "those blocks line up with the biology. If they do, the comparison "
            "is not recoverable; if they cross it, it is.",
            evidence={"grouping_factors": grouping,
                      "alias_clusters": audit.alias_clusters},
        )
    else:
        f.info(
            "design.verdict_ok",
            "SEPARABILITY VERDICT: OK. Every technical factor either is constant "
            "across the study or varies within every condition, so no technical "
            "effect is confounded with the biological contrast. Batch effects "
            "here are a matter of statistical efficiency, not validity.",
        )
