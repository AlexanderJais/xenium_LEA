"""
stratify.py
-----------
Run the audit separately within each level of a manifest column.

When a covariate turns out to be perfectly aliased with a technical factor,
analysing within its levels is usually the right response — and it is worth
being precise about what that buys and what it costs, because it is not simply
"the safe option".

**What it buys.** Inside a stratum the aliased factors are *constant*. If every
male section was processed on one panel design with one segmentation method,
then within the male stratum there is no panel difference and no segmentation
difference to confound anything. Two consequences follow that the pooled
analysis cannot offer:

* The **whole panel** is usable, not just the genes shared across designs.
  Genes dropped from the pooled safe set because one design lacked them are
  perfectly comparable inside the design that has them.
* Technical factors that were unadjustable become non-existent.

**What it costs.** Replicates. Splitting an n=4 vs n=4 comparison by a factor
that halves it leaves n=2 vs n=2 in each stratum, and no stratum can borrow
strength from the other.

**What it forecloses.** The strata can no longer be compared to each other. If
the split is on sex, and sex is aliased with batch, then a male-vs-female
difference remains exactly as unanswerable as before — stratifying does not
recover it, it accepts losing it.

**What it turns the study into.** Two independent replicates of the same
biological question, run under entirely different technical conditions. An
effect appearing in both is far better evidence than one pooled result, because
no shared technical artefact could produce it. That framing is usually the
strongest thing available from a confounded design, and it is reported here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .audit import run_audit
from .cell_qc import DEFAULT_COUNT_THRESHOLDS
from .design import ALIASED, CONSTANT, NESTED, PARTIAL
from .findings import Findings
from .manifest import RunEntry, RunManifest
from .report import AuditResult

logger = logging.getLogger(__name__)

#: A level with fewer runs than this cannot support an audit worth the name.
MIN_RUNS_PER_STRATUM = 2


@dataclass
class Stratum:
    """One level of the splitting column, audited on its own."""

    key: str
    value: str
    manifest: RunManifest
    result: AuditResult | None = None
    skipped_reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.result is not None


@dataclass
class StratifiedResult:
    """Per-stratum audits plus the pooled one, and the comparison between them."""

    split_by: str
    strata: list[Stratum] = field(default_factory=list)
    pooled: AuditResult | None = None
    findings: Findings = field(default_factory=Findings)
    comparison: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def audited(self) -> list[Stratum]:
        return [s for s in self.strata if s.ok]


def split_manifest(manifest: RunManifest, column: str) -> dict[str, RunManifest]:
    """
    Partition a manifest by one of its columns.

    The column may be a required field (``condition``) or any covariate the
    manifest carried. Levels keep manifest order so downstream tables are
    stable.
    """
    out: dict[str, list[RunEntry]] = {}
    for entry in manifest:
        value = _value_of(entry, column)
        out.setdefault(value, []).append(entry)
    return {k: RunManifest(v) for k, v in out.items()}


def _value_of(entry: RunEntry, column: str) -> str:
    if column in entry.overrides:
        return str(entry.overrides[column]).strip()
    value = getattr(entry, column, None)
    return str(value).strip() if value is not None else "unknown"


def run_stratified_audit(
    manifest: RunManifest,
    base_panel_csv: Path | str,
    split_by: str,
    deep: bool = False,
    min_counts_per_cell: int = 0,
    qc_thresholds: Sequence[int] = DEFAULT_COUNT_THRESHOLDS,
    cache_dir: Path | None = None,
    include_pooled: bool = True,
) -> StratifiedResult:
    """
    Audit each level of ``split_by`` independently.

    The pooled audit is run too, and kept: it is what says *why* the split was
    needed, and comparing the two is how the cost of splitting becomes visible.
    """
    result = StratifiedResult(split_by=split_by)
    f = result.findings

    groups = split_manifest(manifest, split_by)
    if len(groups) < 2:
        only = next(iter(groups), "?")
        f.warning(
            "stratify.single_level",
            f"'{split_by}' has one level ({only!r}) across the study, so "
            "splitting on it changes nothing. The pooled audit is reported "
            "unchanged.",
            evidence={"split_by": split_by, "levels": sorted(groups)},
        )

    if include_pooled:
        logger.info("Auditing pooled study (%d runs)", len(manifest))
        result.pooled = run_audit(
            manifest, base_panel_csv, deep=deep,
            min_counts_per_cell=min_counts_per_cell,
            qc_thresholds=qc_thresholds,
            cache_dir=cache_dir,
        )

    for value in sorted(groups):
        sub = groups[value]
        key = f"{split_by}={value}"
        stratum = Stratum(key=key, value=value, manifest=sub)

        if len(sub) < MIN_RUNS_PER_STRATUM:
            stratum.skipped_reason = (
                f"only {len(sub)} run(s); too few to audit"
            )
            f.warning(
                "stratify.stratum_too_small",
                f"Stratum {key} has {len(sub)} run(s) and was not audited. "
                "Its runs are still covered by the pooled audit.",
                evidence={"stratum": key, "n_runs": len(sub),
                          "runs": sub.run_ids},
                run_ids=sub.run_ids,
            )
            result.strata.append(stratum)
            continue

        logger.info("Auditing stratum %s (%d runs)", key, len(sub))
        stratum.result = run_audit(
            sub, base_panel_csv, deep=deep,
            min_counts_per_cell=min_counts_per_cell,
            qc_thresholds=qc_thresholds,
            cache_dir=cache_dir,
        )
        result.strata.append(stratum)

    result.comparison = _build_comparison(result)
    _report_stratification(result, manifest)
    return result


# ---------------------------------------------------------------------------
# Comparison + findings
# ---------------------------------------------------------------------------

def _build_comparison(result: StratifiedResult) -> pd.DataFrame:
    """One row per stratum, plus the pooled study, side by side."""
    rows = []

    def _row(label: str, res: AuditResult | None, n_runs: int) -> dict[str, Any]:
        if res is None or res.design is None:
            return {"stratum": label, "n_runs": n_runs, "verdict": "not audited"}
        design = res.design
        counts = res.findings.counts()
        reps = design.replicates
        constant = sorted(
            k for k, v in design.verdicts.items() if v.verdict == CONSTANT
        )
        problem = sorted(
            k for k, v in design.verdicts.items()
            if v.verdict in (ALIASED, NESTED, PARTIAL)
        )
        return {
            "stratum": label,
            "n_runs": n_runs,
            "n_mice": int(design.factor_table["mouse_id"].nunique()),
            "conditions": ", ".join(design.condition_levels),
            "mice_per_condition": "; ".join(
                f"{c}={v['n_mice']}" for c, v in sorted(reps.items())
            ),
            "verdict": design.overall,
            "n_errors": counts["error"],
            "n_warnings": counts["warning"],
            "n_safe_genes": len(res.panel.safe_genes) if res.panel else 0,
            "constant_technical_factors": ", ".join(constant),
            "unresolved_factors": ", ".join(problem),
        }

    if result.pooled is not None:
        rows.append(
            _row("POOLED", result.pooled, len(result.pooled.design.factor_table)
                 if result.pooled.design is not None else 0)
        )
    for s in result.strata:
        rows.append(_row(s.key, s.result, len(s.manifest)))
    return pd.DataFrame(rows)


def _report_stratification(result: StratifiedResult, full: RunManifest) -> None:
    """Say what the split fixed, what it cost, and what it gave up."""
    f = result.findings
    audited = result.audited
    if not audited:
        return

    pooled_design = result.pooled.design if result.pooled else None

    # -- what the split resolved ----------------------------------------
    if pooled_design is not None:
        was_problematic = {
            k for k, v in pooled_design.verdicts.items()
            if v.verdict in (ALIASED, NESTED, PARTIAL)
        }
        # ...and every factor the pooled run found aliased with a covariate.
        for finding in result.pooled.findings.by_code(
            "design.covariate_aliased_with_technical"
        ):
            was_problematic.update(finding.evidence.get("aliased_with", []))

        resolved = sorted(
            k for k in was_problematic
            if all(
                s.result.design is not None
                and s.result.design.verdicts.get(k)
                and s.result.design.verdicts[k].verdict == CONSTANT
                for s in audited
            )
        )
        if resolved:
            f.info(
                "stratify.confounds_resolved",
                f"Splitting on '{result.split_by}' makes "
                f"{', '.join(resolved)} constant inside every stratum. Those "
                "factors were confounded in the pooled study and simply do not "
                "vary here, so nothing has to be adjusted for them.",
                evidence={"resolved": resolved},
            )

        # Technical factors only. A covariate nested within condition — age in
        # weeks inside aged/adult — is the definition of the groups, and
        # reporting it as an unresolved confound would say the split failed when
        # it did exactly what was asked.
        still = sorted(
            {
                k
                for s in audited
                if s.result.design is not None
                for k, v in s.result.design.verdicts.items()
                if v.verdict in (ALIASED, NESTED)
                and k not in set(s.result.design.covariates)
            }
        )
        if still:
            f.warning(
                "stratify.confounds_remain",
                f"Even within strata, {', '.join(still)} remain confounded with "
                "condition. Splitting on "
                f"'{result.split_by}' did not fix these.",
                evidence={"factors": still},
            )

    # -- what it gained on the panel -------------------------------------
    pooled_safe = (
        len(result.pooled.panel.safe_genes)
        if result.pooled and result.pooled.panel
        else 0
    )
    stratum_safe = {
        s.key: len(s.result.panel.safe_genes)
        for s in audited
        if s.result.panel is not None
    }
    if pooled_safe and stratum_safe and min(stratum_safe.values()) > pooled_safe:
        f.info(
            "stratify.panel_gain",
            "Each stratum uses more genes than the pooled study: "
            + "; ".join(f"{k} = {v}" for k, v in sorted(stratum_safe.items()))
            + f", against {pooled_safe} pooled. Genes dropped from the pooled "
            "set because one panel design lacked them are perfectly comparable "
            "inside the design that carries them.",
            evidence={"pooled": pooled_safe, "per_stratum": stratum_safe},
        )

    # -- what it cost ------------------------------------------------------
    thin = []
    for s in audited:
        design = s.result.design
        if design is None:
            continue
        for cond, reps in design.replicates.items():
            if reps["n_mice"] < 3:
                thin.append(f"{s.key}: {cond} n={reps['n_mice']} mice")
    if thin:
        f.warning(
            "stratify.low_replication",
            "Splitting leaves few animals per group — "
            + "; ".join(thin)
            + ". At this replication, treat within-stratum results as effect "
            "sizes and directions rather than significance tests. The pooled "
            "study had more animals per group but could not separate the "
            "technical factors; that is the trade being made.",
            evidence={"strata": thin},
        )

    # -- what the pooled errors mean now -----------------------------------
    # The exit code follows the strata, because those are the analyses that will
    # be used; the pooled run is reference material. But a report containing
    # errors that do not gate has to say so, or the zero exit reads as "no
    # problems found".
    if result.pooled is not None:
        pooled_errors = result.pooled.findings.by_severity("error")
        if pooled_errors:
            f.warning(
                "stratify.pooled_errors_not_gating",
                f"The pooled analysis carries {len(pooled_errors)} error(s) — "
                + "; ".join(sorted({e.code for e in pooled_errors}))
                + f". Splitting on '{result.split_by}' is how those are being "
                "addressed, so they do not gate this run: the exit status "
                "follows the per-stratum analyses, which are the ones to use. "
                "The pooled report is kept alongside as the record of why the "
                "split was needed.",
                evidence={
                    "pooled_error_codes": sorted({e.code for e in pooled_errors}),
                    "n_pooled_errors": len(pooled_errors),
                },
            )

    # -- what it forecloses, and what it becomes ---------------------------
    keys = [s.key for s in audited]
    f.warning(
        "stratify.no_cross_stratum_comparison",
        f"The strata ({', '.join(keys)}) cannot be compared with each other. "
        f"'{result.split_by}' was split on precisely because it is entangled "
        "with the technical factors, so a between-stratum difference is still "
        "a mixture of that covariate and the batch. Splitting accepts losing "
        "that comparison rather than recovering it.",
        evidence={"strata": keys},
    )

    if len(audited) >= 2:
        conditions = {
            tuple(s.result.design.condition_levels)
            for s in audited
            if s.result.design is not None
        }
        if len(conditions) == 1:
            f.info(
                "stratify.independent_replication",
                f"Every stratum tests the same contrast "
                f"({' vs '.join(next(iter(conditions)))}) under different "
                "technical conditions — different panel design, different "
                "segmentation, different run. That makes them independent "
                "replicates: an effect found in both is far stronger evidence "
                "than one pooled result, because no shared technical artefact "
                "could produce it. Compare the per-stratum results gene by gene "
                "and report what reproduces.",
                evidence={"strata": keys},
            )
