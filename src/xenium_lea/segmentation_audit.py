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
   cannot be faked by metadata: nucleus-expansion segmentation derives the cell
   boundary from the nucleus by dilating it a fixed distance, so ``cell_area``
   becomes a near-deterministic function of ``nucleus_area`` (high rank
   correlation, tight nucleus/cell area ratio). Stain-based segmentation traces
   a real membrane, so the two areas decouple.

Where the three disagree, **the disagreement is the finding**. The kit call is
reported with a confidence and with every number behind it, so a human can
overrule it.

The absolute thresholds below are deliberately conservative. The more reliable
signal in practice is *relative*: within one study, expansion-segmented and
stain-segmented runs separate cleanly on these metrics even when no single
absolute cutoff would be right. So a cross-run split is computed too, and a
bimodal split is reported whatever the absolute calls say.
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

#: Rank correlation between nucleus_area and cell_area above which the cell
#: boundary looks derived from the nucleus rather than independently traced.
EXPANSION_SPEARMAN_MIN = 0.90

#: Coefficient of variation of nucleus_area/cell_area below which the ratio
#: looks fixed by construction rather than biological.
EXPANSION_RATIO_CV_MAX = 0.20

#: Minimum cells needed before the morphological fingerprint means anything.
MIN_CELLS_FOR_MORPHOLOGY = 200

_EXPANSION_PATTERNS = ("expansion", "nucleus expansion", "interior - nucleus")
_STAIN_PATTERNS = ("boundary", "interior - stain", "multimodal", "segmentation_stain")


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
    """Read the kit off experiment.xenium, if it says anything at all."""
    seg_keys = probe.experiment.get("segmentation_keys") or {}
    evidence: dict[str, Any] = {
        "segmentation_keys": seg_keys,
        "analysis_sw_version": probe.experiment.get("analysis_sw_version"),
    }
    if not seg_keys:
        return KIT_UNKNOWN, evidence

    blob = " ".join(f"{k}={v}" for k, v in seg_keys.items()).lower()

    # An explicit expansion *distance* is the strongest declared evidence of
    # expansion segmentation — check it before the generic word match, since
    # newer runs mention both (they expand only where no stain was resolved).
    has_distance = any(
        "expansion_distance" in k.lower() and _is_positive_number(v)
        for k, v in seg_keys.items()
    )
    has_stain = any(p in blob for p in _STAIN_PATTERNS)

    if has_stain:
        return KIT_STAIN, evidence
    if has_distance or any(p in blob for p in _EXPANSION_PATTERNS):
        return KIT_EXPANSION, evidence
    return KIT_UNKNOWN, evidence


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

    Expansion segmentation dilates the nucleus by a fixed distance, so cell area
    is a monotone function of nucleus area and the nucleus/cell ratio is nearly
    constant. A traced membrane decouples the two.
    """
    metrics: dict[str, Any] = {
        "n_cells_morphology": 0,
        "nucleus_cell_area_spearman": np.nan,
        "area_ratio_cv": np.nan,
        "area_ratio_median": np.nan,
        "cell_area_cv": np.nan,
        "cell_area_median": np.nan,
        "nucleus_area_median": np.nan,
        "frac_cells_no_nucleus": np.nan,
    }

    cells = probe.cells
    if cells is None or "cell_area" not in cells.columns:
        return KIT_UNKNOWN, metrics

    cell_area = pd.to_numeric(cells["cell_area"], errors="coerce")
    metrics["cell_area_median"] = _round(cell_area.median())
    metrics["cell_area_cv"] = _round(_cv(cell_area))

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
            float(((nucleus_area.fillna(0) <= 0)).mean())
        )

    if n_valid < MIN_CELLS_FOR_MORPHOLOGY:
        return KIT_UNKNOWN, metrics

    ca = cell_area[valid].to_numpy(dtype=float)
    na = nucleus_area[valid].to_numpy(dtype=float)

    # Spearman needs variation in both; a constant column yields NaN.
    if np.ptp(ca) > 0 and np.ptp(na) > 0:
        rho = stats.spearmanr(na, ca).statistic
        metrics["nucleus_cell_area_spearman"] = _round(float(rho))
    else:
        rho = np.nan

    ratio = na / ca
    metrics["area_ratio_median"] = _round(float(np.median(ratio)))
    metrics["area_ratio_cv"] = _round(_cv(pd.Series(ratio)))

    rho_v = metrics["nucleus_cell_area_spearman"]
    cv_v = metrics["area_ratio_cv"]
    if rho_v is None or cv_v is None or np.isnan(rho_v) or np.isnan(cv_v):
        return KIT_UNKNOWN, metrics

    looks_expanded = rho_v >= EXPANSION_SPEARMAN_MIN and cv_v <= EXPANSION_RATIO_CV_MAX
    if looks_expanded:
        return KIT_EXPANSION, metrics
    # Clearly decoupled areas: a traced boundary.
    if rho_v < EXPANSION_SPEARMAN_MIN and cv_v > EXPANSION_RATIO_CV_MAX:
        return KIT_STAIN, metrics
    return KIT_UNKNOWN, metrics


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

    # -- relative split ---------------------------------------------------
    # Absolute thresholds are conservative on purpose; a clean bimodal split in
    # the morphology across runs is meaningful even where no absolute cutoff
    # fires, so report it independently of the calls above.
    _report_morphology_split(table, f)

    return table, calls


def _report_morphology_split(table: pd.DataFrame, f: Findings) -> None:
    """Flag a bimodal morphological fingerprint across runs."""
    col = "nucleus_cell_area_spearman"
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

    f.warning(
        "segmentation.morphology_split",
        f"Runs split into two groups on the nucleus/cell area coupling "
        f"(gap of {max_gap:.2f} at rho={split_at:.2f}): "
        f"loosely coupled = {', '.join(sorted(low))}; tightly coupled = "
        f"{', '.join(sorted(high))}. Tight coupling indicates nucleus-expansion "
        "segmentation. This split is derived from the geometry alone and holds "
        "regardless of what the metadata declares.",
        evidence={
            "split_at_spearman": round(split_at, 4),
            "max_gap": round(max_gap, 4),
            "loosely_coupled": sorted(low.tolist()),
            "tightly_coupled": sorted(high.tolist()),
        },
    )
