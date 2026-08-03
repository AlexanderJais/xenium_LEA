"""
segmentation_audit.py
---------------------
Which runs used the cell segmentation staining kit, and what it did to the cells.

Three *independent* signals are collected and reported side by side rather than
collapsed into one number, because each fails differently:

1. **Declared** — segmentation keys in ``experiment.xenium`` plus the analysis
   software version. Authoritative when present, but absent entirely on older
   runs and renamed between Xenium Ranger generations.
2. **Structural** — which columns and files the run actually produced
   (``nucleus_area``, ``nucleus_count``, ``segmentation_method``, boundary
   parquets; ``cells.parquet`` vs the older ``cells.csv.gz``). Robust, but it
   tracks the software generation, which is correlated with — not identical to —
   the kit.
3. **Morphological** — the shape of the segmentation output itself. This one
   cannot be faked by metadata: nucleus expansion grows each nucleus outward by
   a fixed distance, so no cell's boundary can sit further from its nucleus than
   that. A traced membrane obeys no such bound. The fingerprint reads that
   **ceiling** — see :func:`_morphological_signal`.

Where the three disagree, **the disagreement is the finding**. The kit call is
reported with a confidence and with every number behind it, so a human can
overrule it.

The absolute thresholds below are deliberately conservative. The more reliable
signal in practice is *relative*: within one study, expansion-segmented and
stain-segmented runs separate cleanly on these metrics even when no single
absolute cutoff would be right. So a cross-run split is computed too, and a
bimodal split is reported whatever the absolute calls say.

A caution about the morphological signal, learned the hard way. Its first
version tested whether nucleus and cell area were tightly *coupled*, reasoning
that a dilated nucleus determines its cell. That is true of an idealised
dilation and false of the real thing: expansion stops at neighbouring cells, so
cell size tracks local density as much as nucleus size. A genuine 100%
nucleus-expansion run scored as stain under that rule. Any future change here
should be checked against real output of both kinds before it is believed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .findings import Findings
from .probe import RunProbe

logger = logging.getLogger(__name__)

KIT_STAIN = "stain_kit"              # multimodal cell segmentation staining
KIT_EXPANSION = "nucleus_expansion"  # DAPI nucleus + fixed-distance dilation
KIT_UNKNOWN = "unknown"

#: Ratio of the 99.9th to the 99th percentile of implied radial expansion,
#: above which the distribution has no ceiling and the boundary was traced
#: rather than dilated. See :func:`_morphological_signal` for why this is the
#: statistic. Calibrated on real runs of each kind (1.06 for a 100%
#: nucleus-expansion run, 1.47 for a 93%-stain one) and set conservatively
#: between them, nearer the expansion side so a marginal case is left unknown
#: rather than called wrongly.
EXPANSION_TAIL_RATIO_MAX = 1.25

#: Implied expansion distances beyond this (um) are not plausible for a
#: dilation-based segmentation at any setting 10x offers, so a ceiling estimated
#: above it is not a ceiling.
MAX_PLAUSIBLE_EXPANSION_UM = 20.0

#: Minimum cells needed before the morphological fingerprint means anything.
#: The statistic reads the extreme tail, so it needs more cells than a
#: median-based one would.
MIN_CELLS_FOR_MORPHOLOGY = 2000

#: Substrings that mark a cell as nucleus-expanded, matched case-insensitively
#: against the per-cell ``segmentation_method`` string or the declared metadata.
_EXPANSION_PATTERNS = ("expansion", "interior - nucleus")

#: Substrings that mark a cell as stain-resolved. ``interior stain`` matters:
#: Xenium Ranger v4+ writes "Segmented by interior stain (18S)" for the large
#: majority of cells in a kit run, and matching only "boundary" would see just
#: the small boundary-stain minority and call the whole run nucleus-expanded.
_STAIN_PATTERNS = (
    "boundary",
    "interior stain",
    "interior - stain",
    "multimodal",
    "segmentation_stain",
    "cell stain",
)


@dataclass
class SegmentationCall:
    """The segmentation verdict for one run, with its evidence."""

    run_id: str
    call: str = KIT_UNKNOWN
    confidence: str = "none"          # high | medium | low | none
    declared: str = KIT_UNKNOWN
    structural: str = KIT_UNKNOWN
    morphological: str = KIT_UNKNOWN
    disagreement: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        row = {
            "run_id": self.run_id,
            "segmentation_kit": self.call,
            "confidence": self.confidence,
            "signal_declared": self.declared,
            "signal_structural": self.structural,
            "signal_morphological": self.morphological,
            "signals_disagree": self.disagreement,
        }
        row.update(self.metrics)
        return row


def _declared_signal(probe: RunProbe) -> tuple[str, dict[str, Any]]:
    """
    Read the kit off experiment.xenium.

    Xenium Ranger v4+ states this outright: ``segmented_cell_stain_frac`` and
    ``segmented_cell_nuc_expansion_frac`` are the instrument's own tally of how
    each cell was resolved. Where those exist nothing needs inferring, so they
    are checked first and the fractions are carried into the report.
    """
    exp = probe.experiment or {}
    seg_keys = exp.get("segmentation_keys") or {}
    evidence: dict[str, Any] = {
        "segmentation_keys": seg_keys,
        "analysis_sw_version": exp.get("analysis_sw_version"),
        "segmentation_stain": exp.get("segmentation_stain"),
    }

    # 1. Quantitative fractions (v4+). Note a kit run still expands the minority
    #    of cells where no stain resolved, so this is a majority call, not an
    #    all-or-nothing one.
    frac_stain = _as_float(exp.get("frac_stain"))
    frac_expansion = _as_float(exp.get("frac_nuc_expansion"))
    if frac_stain is not None or frac_expansion is not None:
        evidence["declared_frac_stain"] = frac_stain
        evidence["declared_frac_nuc_expansion"] = frac_expansion
        s = frac_stain or 0.0
        e = frac_expansion or 0.0
        if s > e:
            return KIT_STAIN, evidence
        if e > s:
            return KIT_EXPANSION, evidence

    # 2. A named stain reagent is unambiguous even without fractions.
    if str(exp.get("segmentation_stain") or "").strip():
        return KIT_STAIN, evidence

    if not seg_keys:
        return KIT_UNKNOWN, evidence

    # 3. Fall back to word-matching whatever segmentation keys exist.
    blob = " ".join(f"{k}={v}" for k, v in seg_keys.items()).lower()
    has_distance = any(
        "expansion_distance" in k.lower() and _is_positive_number(v)
        for k, v in seg_keys.items()
    )
    if any(p in blob for p in _STAIN_PATTERNS):
        return KIT_STAIN, evidence
    if has_distance or any(p in blob for p in _EXPANSION_PATTERNS):
        return KIT_EXPANSION, evidence
    return KIT_UNKNOWN, evidence


def _as_float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def _is_positive_number(v: Any) -> bool:
    try:
        return float(v) > 0
    except (TypeError, ValueError):
        return False


def _structural_signal(probe: RunProbe) -> tuple[str, dict[str, Any]]:
    """Infer the kit from which outputs the run produced."""
    cells = probe.cells
    cols = set(cells.columns) if cells is not None else set()
    files = probe.files_present or {}

    evidence = {
        "cells_source": probe.cells_source,
        "has_nucleus_area": "nucleus_area" in cols,
        "has_nucleus_count": "nucleus_count" in cols,
        "has_segmentation_method_column": "segmentation_method" in cols,
        "has_cell_boundaries": bool(files.get("cell_boundaries.parquet")),
        "has_nucleus_boundaries": bool(files.get("nucleus_boundaries.parquet")),
    }

    # A per-cell segmentation_method column is definitive: it names the method
    # cell by cell. A run where most cells were resolved by stain used the kit.
    if cells is not None and "segmentation_method" in cols:
        vc = cells["segmentation_method"].astype(str).str.lower().value_counts()
        evidence["segmentation_method_values"] = {
            str(k): int(v) for k, v in vc.head(10).items()
        }
        total = int(vc.sum())
        if total:
            stain_n = int(
                sum(v for k, v in vc.items() if any(p in k for p in _STAIN_PATTERNS))
            )
            expansion_n = int(
                sum(v for k, v in vc.items() if any(p in k for p in _EXPANSION_PATTERNS))
            )
            evidence["frac_stain_resolved"] = round(stain_n / total, 4)
            evidence["frac_expansion_resolved"] = round(expansion_n / total, 4)
            if stain_n > expansion_n:
                return KIT_STAIN, evidence
            if expansion_n > 0:
                return KIT_EXPANSION, evidence

    # Older bundles (imported segmentation, no stain) wrote cells.csv.gz and
    # carried no nucleus_count. That combination is characteristic of the
    # pre-kit generation.
    if probe.cells_source and probe.cells_source.startswith("cells.csv"):
        return KIT_EXPANSION, evidence
    if cols and "nucleus_count" not in cols and "nucleus_area" in cols:
        return KIT_EXPANSION, evidence

    return KIT_UNKNOWN, evidence


def _morphological_signal(probe: RunProbe) -> tuple[str, dict[str, Any]]:
    """
    Fingerprint the segmentation from the geometry it produced.

    Nucleus expansion grows each nucleus outward by a fixed distance, stopping
    early where it meets a neighbour. So the *implied radial expansion* of every
    cell,

        d = sqrt(cell_area / pi) - sqrt(nucleus_area / pi)

    has a **hard ceiling** at the configured distance: no cell can exceed it,
    however isolated. A traced membrane obeys no such bound, and its d has a
    long tail.

    The statistic is therefore the tail ratio ``p99.9(d) / p99(d)``. A ceiling
    makes the upper percentiles converge (ratio near 1); an unbounded
    distribution keeps climbing. It is scale-free, so it needs no knowledge of
    the expansion distance that was configured, and it is read from percentiles
    rather than the maximum so one ragged polygon cannot flip the call.

    An earlier version tested whether nucleus and cell area were *tightly
    coupled*, on the reasoning that a dilated nucleus determines its cell. Real
    data disproved that: because expansion stops at neighbours, cell size
    depends on local density as much as on nucleus size, and a genuine 100%
    nucleus-expansion run showed correlation 0.79 with a ratio CV of 0.45 —
    scoring as stain under the old rule. The ceiling survives that clipping;
    the coupling does not.
    """
    metrics: dict[str, Any] = {
        "n_cells_morphology": 0,
        "implied_expansion_p99": np.nan,
        "implied_expansion_p999": np.nan,
        "implied_expansion_max": np.nan,
        "expansion_tail_ratio": np.nan,
        "nucleus_cell_area_spearman": np.nan,
        "area_ratio_cv": np.nan,
        "area_ratio_median": np.nan,
        "cell_area_median": np.nan,
        "nucleus_area_median": np.nan,
        "frac_cells_no_nucleus": np.nan,
    }

    cells = probe.cells
    if cells is None or "cell_area" not in cells.columns:
        return KIT_UNKNOWN, metrics

    cell_area = pd.to_numeric(cells["cell_area"], errors="coerce")
    metrics["cell_area_median"] = _round(cell_area.median())

    if "nucleus_area" not in cells.columns:
        return KIT_UNKNOWN, metrics

    nucleus_area = pd.to_numeric(cells["nucleus_area"], errors="coerce")
    metrics["nucleus_area_median"] = _round(nucleus_area.median())

    valid = cell_area.notna() & nucleus_area.notna() & (cell_area > 0)
    n_valid = int(valid.sum())
    metrics["n_cells_morphology"] = n_valid

    # Cells with no nucleus cannot have been produced by nucleus expansion, so
    # a non-zero fraction is itself informative.
    with np.errstate(invalid="ignore"):
        metrics["frac_cells_no_nucleus"] = _round(
            float((nucleus_area.fillna(0) <= 0).mean())
        )

    ca = cell_area[valid].to_numpy(dtype=float)
    na = nucleus_area[valid].to_numpy(dtype=float)

    # Descriptive, and kept because they are what a human looks at first —
    # they simply are not what the call is made on.
    if len(ca) > 1 and np.ptp(ca) > 0 and np.ptp(na) > 0:
        metrics["nucleus_cell_area_spearman"] = _round(
            float(stats.spearmanr(na, ca).statistic)
        )
        ratio = na / ca
        metrics["area_ratio_median"] = _round(float(np.median(ratio)))
        metrics["area_ratio_cv"] = _round(_cv(pd.Series(ratio)))

    if n_valid < MIN_CELLS_FOR_MORPHOLOGY:
        return KIT_UNKNOWN, metrics

    # Implied radial expansion, from circular-equivalent radii.
    d = np.sqrt(ca / np.pi) - np.sqrt(na / np.pi)
    d = d[np.isfinite(d)]
    if len(d) < MIN_CELLS_FOR_MORPHOLOGY:
        return KIT_UNKNOWN, metrics

    p99, p999 = np.percentile(d, [99.0, 99.9])
    metrics["implied_expansion_p99"] = _round(float(p99), 3)
    metrics["implied_expansion_p999"] = _round(float(p999), 3)
    metrics["implied_expansion_max"] = _round(float(d.max()), 3)

    if p99 <= 0 or p99 > MAX_PLAUSIBLE_EXPANSION_UM:
        return KIT_UNKNOWN, metrics

    tail_ratio = float(p999 / p99)
    metrics["expansion_tail_ratio"] = _round(tail_ratio, 4)

    return (
        KIT_EXPANSION if tail_ratio <= EXPANSION_TAIL_RATIO_MAX else KIT_STAIN
    ), metrics


def _cv(s: pd.Series) -> float:
    """Coefficient of variation, NaN-safe."""
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 2:
        return float("nan")
    m = float(s.mean())
    if m == 0:
        return float("nan")
    return float(s.std(ddof=1) / abs(m))


def _round(v, nd: int = 4):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else round(f, nd)


def audit_segmentation(
    probes: list[RunProbe],
    findings: Findings | None = None,
) -> tuple[pd.DataFrame, dict[str, SegmentationCall]]:
    """
    Call the segmentation method for every run.

    Returns the per-run table and the calls keyed by ``run_id``. The
    ``segmentation_kit`` column feeds ``design.py``, where what actually matters
    is decided: whether the kit difference lines up with condition.
    """
    f = findings if findings is not None else Findings()
    calls: dict[str, SegmentationCall] = {}

    for p in probes:
        declared, decl_ev = _declared_signal(p)
        structural, struct_ev = _structural_signal(p)
        morphological, metrics = _morphological_signal(p)

        votes = [v for v in (declared, structural, morphological) if v != KIT_UNKNOWN]
        distinct = set(votes)

        if not votes:
            call, confidence = KIT_UNKNOWN, "none"
        elif len(distinct) == 1:
            call = votes[0]
            confidence = {1: "low", 2: "medium", 3: "high"}[len(votes)]
        else:
            # Signals conflict. Prefer declared, then morphological — the
            # morphology is the segmentation output itself, whereas the
            # structural signal really tracks the software generation.
            call = declared if declared != KIT_UNKNOWN else morphological
            confidence = "low"

        # Surface the quantitative fractions in the table, not just the evidence
        # blob: a kit run is rarely 100% stain-resolved, and *how much* of it was
        # resolved by stain is a graded technical covariate in its own right.
        metrics = dict(metrics)
        metrics["declared_frac_stain"] = _round(decl_ev.get("declared_frac_stain"))
        metrics["declared_frac_nuc_expansion"] = _round(
            decl_ev.get("declared_frac_nuc_expansion")
        )
        metrics["observed_frac_stain"] = _round(struct_ev.get("frac_stain_resolved"))
        metrics["observed_frac_expansion"] = _round(
            struct_ev.get("frac_expansion_resolved")
        )
        metrics["segmentation_stain"] = decl_ev.get("segmentation_stain")

        calls[p.run_id] = SegmentationCall(
            run_id=p.run_id,
            call=call,
            confidence=confidence,
            declared=declared,
            structural=structural,
            morphological=morphological,
            disagreement=len(distinct) > 1,
            metrics=metrics,
            evidence={"declared": decl_ev, "structural": struct_ev},
        )

    table = pd.DataFrame([c.to_row() for c in calls.values()])
    if table.empty:
        return table, calls

    # -- findings --------------------------------------------------------
    conflicted = [c.run_id for c in calls.values() if c.disagreement]
    if conflicted:
        f.warning(
            "segmentation.signals_disagree",
            f"{len(conflicted)} run(s) give conflicting segmentation evidence "
            f"({', '.join(sorted(conflicted))}). The declared metadata, the "
            "output structure and the geometry do not agree; treat the kit call "
            "for these runs as provisional and check the per-run metrics.",
            evidence={
                r: {
                    "declared": calls[r].declared,
                    "structural": calls[r].structural,
                    "morphological": calls[r].morphological,
                    "metrics": calls[r].metrics,
                }
                for r in sorted(conflicted)
            },
            run_ids=sorted(conflicted),
        )

    unknown = [c.run_id for c in calls.values() if c.call == KIT_UNKNOWN]
    if unknown:
        f.warning(
            "segmentation.unknown",
            f"Could not determine the segmentation method for "
            f"{len(unknown)} run(s): {', '.join(sorted(unknown))}. Supply it as "
            "a `segmentation_kit` column in the manifest if you know it — "
            "otherwise it cannot be tested for confounding with condition.",
            run_ids=sorted(unknown),
        )

    kits = {c.call for c in calls.values() if c.call != KIT_UNKNOWN}
    if len(kits) > 1:
        by_kit: dict[str, list[str]] = {}
        for c in calls.values():
            by_kit.setdefault(c.call, []).append(c.run_id)
        f.warning(
            "segmentation.mixed_methods",
            "The study mixes segmentation methods: "
            + "; ".join(
                f"{k} = {', '.join(sorted(v))}" for k, v in sorted(by_kit.items())
            )
            + ". Segmentation sets what counts as a cell, so it shifts "
            "transcripts per cell, cell area and cell-type proportions — this is "
            "typically the largest technical effect in a mixed study. Whether it "
            "is recoverable depends on the separability verdict.",
            evidence={k: sorted(v) for k, v in by_kit.items()},
        )
    elif len(kits) == 1:
        f.info(
            "segmentation.uniform",
            f"All runs with a determinable method used {kits.pop()} — "
            "segmentation cannot contribute a batch effect here.",
        )

    _report_stain_fraction_spread(table, f)

    # -- relative split ---------------------------------------------------
    # Absolute thresholds are conservative on purpose; a clean bimodal split in
    # the morphology across runs is meaningful even where no absolute cutoff
    # fires, so report it independently of the calls above.
    _report_morphology_split(table, f)

    return table, calls


def _report_stain_fraction_spread(table: pd.DataFrame, f: Findings) -> None:
    """
    Flag a graded difference in how much of each run the stain actually resolved.

    Segmentation is not binary even within a kit run: some cells fall back to
    nucleus expansion where no stain was resolved. Two runs both called
    ``stain_kit`` can still differ substantially in that mix, which shifts cell
    size and transcripts per cell in the same direction a kit-vs-no-kit
    difference would — just less.
    """
    col = "declared_frac_stain"
    if col not in table.columns:
        return
    v = pd.to_numeric(table[col], errors="coerce")
    same_kit = table.loc[v.notna() & (table["segmentation_kit"] == KIT_STAIN)]
    if len(same_kit) < 2:
        return
    vals = pd.to_numeric(same_kit[col], errors="coerce")
    spread = float(vals.max() - vals.min())
    if spread < 0.10:
        return
    f.warning(
        "segmentation.stain_fraction_spread",
        f"Runs called {KIT_STAIN} differ in how much of the section the stain "
        f"actually resolved: {vals.min():.1%} to {vals.max():.1%} "
        f"(spread {spread:.1%}). The remainder fell back to nucleus expansion, "
        "so these runs are not equivalent even though the kit was used "
        "throughout — treat the stain fraction as a graded technical covariate "
        "rather than assuming a clean two-level factor.",
        evidence={
            "per_run": {
                str(r.run_id): _round(getattr(r, col)) for r in same_kit.itertuples()
            },
            "spread": round(spread, 4),
        },
    )


def _report_morphology_split(table: pd.DataFrame, f: Findings) -> None:
    """Flag a bimodal morphological fingerprint across runs."""
    col = "expansion_tail_ratio"
    if col not in table.columns:
        return
    vals = pd.to_numeric(table[col], errors="coerce").dropna()
    if len(vals) < 3:
        return

    # A gap in the sorted values wider than half the total range separates two
    # populations far more reliably than any fixed cutoff.
    s = vals.sort_values()
    gaps = s.diff().dropna()
    if gaps.empty:
        return
    span = float(s.max() - s.min())
    max_gap = float(gaps.max())
    if span <= 0 or max_gap < 0.5 * span or max_gap < 0.05:
        return

    split_at = float(s[gaps.idxmax()])
    low = table.loc[pd.to_numeric(table[col], errors="coerce") < split_at, "run_id"]
    high = table.loc[pd.to_numeric(table[col], errors="coerce") >= split_at, "run_id"]
    if low.empty or high.empty:
        return

    # Only report a split that crosses the decision boundary. Runs all on the
    # same side of it are all the same segmentation method, and the "gap"
    # between them is ordinary run-to-run variation — with a handful of runs,
    # some gap always looks wide relative to a narrow range.
    if not (
        float(s.min()) < EXPANSION_TAIL_RATIO_MAX <= float(s.max())
    ):
        return

    f.warning(
        "segmentation.morphology_split",
        f"Runs split into two groups on the implied-expansion tail ratio "
        f"(gap of {max_gap:.2f} at {split_at:.2f}): bounded = "
        f"{', '.join(sorted(low))}; unbounded = {', '.join(sorted(high))}. "
        "A bounded tail means every cell's growth beyond its nucleus stopped at "
        "the same ceiling, which is what nucleus expansion does and a traced "
        "membrane does not. This split comes from the geometry alone and holds "
        "regardless of what the metadata declares.",
        evidence={
            "split_at_tail_ratio": round(split_at, 4),
            "max_gap": round(max_gap, 4),
            "bounded_expansion_like": sorted(low.tolist()),
            "unbounded_stain_like": sorted(high.tolist()),
        },
    )
