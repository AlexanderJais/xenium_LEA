"""
cell_qc.py
----------
Per-run quality metrics, computed from ``cells.parquet`` alone — no matrix read.

The headline metric is the **negative-control rate**: counts on probes and
codewords designed to bind nothing, divided by everything decoded. It is the one
quality number that is *panel-independent*, which is exactly what a study with
differing add-on panels needs — a run's transcripts per cell partly reflects how
many genes its panel carried, but its negative-control rate does not. The sibling
`xenium-spatial` loader discards these control features at load, so this number
is currently unavailable there.

Unassigned and deprecated codewords are counted separately as ``background_rate``
rather than folded in. Which codewords fall in those classes is a property of the
panel design and the Ranger version, so a study spanning two versions would show
a "quality" difference that is really a nomenclature difference.

Two other things this module exists to surface:

* **Cells that a QC filter would remove, per run.** If one run loses 3% of cells
  at a 10-transcript floor and another loses 30%, the filtering step is itself a
  batch effect — the runs enter the analysis having been thinned differently.
  Worth seeing *before* a threshold is picked. (No filtering happens anywhere in
  the sibling pipeline today, so this is also a preview of what adding one would
  do.)
* **Outliers, quantitatively.** Every metric is scored against the across-run
  median using a MAD-based robust z, so "run 5 looks odd" becomes a number
  rather than an impression.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .findings import Findings
from .probe import RunProbe

logger = logging.getLogger(__name__)

#: Per-cell transcript floors previewed in the report.
DEFAULT_COUNT_THRESHOLDS: tuple[int, ...] = (10, 20, 50)

#: True negative controls: probes and codewords designed to bind nothing. Which
#: ones a panel carries is stable across designs, so a rate built on these is
#: comparable between runs — the property that makes it the right cross-run
#: quality metric in a study with differing add-on panels.
_STRICT_CONTROL_COLUMNS = (
    "control_probe_counts",
    "control_codeword_counts",
    "genomic_control_counts",
)

#: Decoding background. Which codewords count as unassigned or deprecated is a
#: property of the panel design and the Ranger version, so this rate is NOT
#: comparable across runs built on different panels or software — folding it
#: into the quality number would let a version difference read as a quality
#: difference. Reported separately for that reason.
_BACKGROUND_CONTROL_COLUMNS = (
    "unassigned_codeword_counts",
    "deprecated_codeword_counts",
)

_CONTROL_COLUMNS = _STRICT_CONTROL_COLUMNS + _BACKGROUND_CONTROL_COLUMNS

#: |robust z| above which a run is called an outlier on a metric.
OUTLIER_Z = 3.0

#: Minimum relative deviation from the across-run median before an outlier is
#: reported. A robust z alone over-fires on tightly clustered replicates: when
#: eight runs sit within 1% of each other, a run 7% away scores five MADs out
#: while being of no practical consequence. Both bars must clear.
OUTLIER_MIN_RELATIVE_DEVIATION = 0.25

#: Strict negative-control rate above which a run is flagged. Healthy Xenium
#: runs sit far below this — a well-behaved run is typically a few hundredths of
#: a percent — so 1% is a generous ceiling rather than a target, and crossing it
#: points at background, decoding problems or over-segmentation.
CONTROL_RATE_WARN = 0.01

#: Metrics scored for outliers across runs, and whether high values are the
#: concerning direction.
_OUTLIER_METRICS = (
    "median_transcripts_per_cell",
    "control_rate",
    "median_cell_area",
    "frac_cells_below_10",
)


@dataclass
class CellQC:
    """Per-run QC table plus the outlier scoring."""

    per_run: pd.DataFrame
    robust_z: pd.DataFrame = field(default_factory=pd.DataFrame)
    thresholds: tuple[int, ...] = DEFAULT_COUNT_THRESHOLDS


def _num(cells: pd.DataFrame, col: str) -> pd.Series | None:
    if col not in cells.columns:
        return None
    return pd.to_numeric(cells[col], errors="coerce")


def _q(s: pd.Series | None, q: float) -> float | None:
    if s is None:
        return None
    s = s.dropna()
    if s.empty:
        return None
    return round(float(s.quantile(q)), 4)


def _run_metrics(
    probe: RunProbe,
    thresholds: Sequence[int],
) -> dict[str, Any]:
    """Compute one run's QC row."""
    row: dict[str, Any] = {
        "run_id": probe.run_id,
        "mouse_id": probe.mouse_id,
        "section_id": probe.section_id,
        "condition": probe.condition,
        "n_cells": probe.n_cells,
        "cells_source": probe.cells_source,
        "n_rna_targets": probe.n_rna,
        "n_control_features": probe.n_control_features,
    }
    for t in thresholds:
        row[f"frac_cells_below_{t}"] = None
    row.update(
        {
            "median_transcripts_per_cell": None,
            "q25_transcripts_per_cell": None,
            "q75_transcripts_per_cell": None,
            "mean_transcripts_per_cell": None,
            "total_transcripts": None,
            "control_counts": None,
            "control_rate": None,
            "background_rate": None,
            "total_control_rate": None,
            "median_cell_area": None,
            "median_nucleus_area": None,
            "frac_cells_no_nucleus": None,
            "frac_cells_zero_transcripts": None,
        }
    )

    cells = probe.cells
    if cells is None or cells.empty:
        return row

    tx = _num(cells, "transcript_counts")
    if tx is not None:
        row["median_transcripts_per_cell"] = _q(tx, 0.5)
        row["q25_transcripts_per_cell"] = _q(tx, 0.25)
        row["q75_transcripts_per_cell"] = _q(tx, 0.75)
        row["mean_transcripts_per_cell"] = round(float(tx.mean(skipna=True)), 4)
        row["total_transcripts"] = int(tx.fillna(0).sum())
        row["frac_cells_zero_transcripts"] = round(
            float((tx.fillna(0) <= 0).mean()), 6
        )
        for t in thresholds:
            row[f"frac_cells_below_{t}"] = round(float((tx.fillna(0) < t).mean()), 6)

    # Control rates. Denominator is RNA + every control, so each is a genuine
    # fraction of everything decoded rather than a ratio against RNA alone.
    present_controls = [c for c in _CONTROL_COLUMNS if c in cells.columns]
    if present_controls:
        def _sum(cols) -> float:
            present = [c for c in cols if c in cells.columns]
            if not present:
                return 0.0
            return float(
                sum(_num(cells, c).fillna(0).sum() for c in present)  # type: ignore[union-attr]
            )

        strict_total = _sum(_STRICT_CONTROL_COLUMNS)
        background_total = _sum(_BACKGROUND_CONTROL_COLUMNS)
        ctrl_total = strict_total + background_total
        rna_total = float(tx.fillna(0).sum()) if tx is not None else 0.0
        denom = ctrl_total + rna_total

        row["control_counts"] = int(ctrl_total)
        row["control_columns_used"] = ",".join(present_controls)
        if denom > 0:
            # control_rate is the strict one — it is the number compared across
            # runs, and the one the outlier check scores.
            row["control_rate"] = round(strict_total / denom, 8)
            row["background_rate"] = round(background_total / denom, 8)
            row["total_control_rate"] = round(ctrl_total / denom, 8)

    ca = _num(cells, "cell_area")
    if ca is not None:
        row["median_cell_area"] = _q(ca, 0.5)
    na = _num(cells, "nucleus_area")
    if na is not None:
        row["median_nucleus_area"] = _q(na, 0.5)
        row["frac_cells_no_nucleus"] = round(float((na.fillna(0) <= 0).mean()), 6)

    return row


def _robust_scale(s: pd.Series) -> float:
    """
    A spread estimate that does not collapse to zero on tight replicates.

    Plain MAD is the right first choice, but it degenerates exactly where this
    check is needed: when more than half the runs share an identical value —
    which is common for an integer metric like median transcripts per cell —
    the MAD is 0 and every outlier scores as undefined. So cascade:

        1.4826 x MAD        the robust default
        IQR / 1.349         still robust, survives up to 25% ties
        1.253 x mean|dev|   last resort; not robust to *several* outliers, but
                            it is only reached when the two above are exactly 0

    Returns 0.0 when every value is identical, which the caller reads as
    "no spread, so no outliers".
    """
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 2:
        return 0.0
    med = s.median()
    dev = (s - med).abs()

    mad = float(dev.median()) * 1.4826
    if np.isfinite(mad) and mad > 0:
        return mad

    iqr = float(s.quantile(0.75) - s.quantile(0.25)) / 1.349
    if np.isfinite(iqr) and iqr > 0:
        return iqr

    mean_ad = float(dev.mean()) * 1.253
    return mean_ad if np.isfinite(mean_ad) and mean_ad > 0 else 0.0


def _robust_z(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """
    Leave-one-out robust z per metric, across runs.

    Each run is scored against the median and spread of the *other* runs. Scoring
    against all runs lets a single bad run inflate the scale enough to hide
    itself — the precise case of a study with tight replicates and one failure,
    which is what this check is for.
    """
    out = pd.DataFrame({"run_id": df["run_id"]})
    for col in columns:
        if col not in df.columns:
            continue
        v = pd.to_numeric(df[col], errors="coerce")
        scores = []
        for idx in v.index:
            others = v.drop(index=idx).dropna()
            value = v.loc[idx]
            scale = _robust_scale(others)
            if len(others) < 2 or scale <= 0 or not np.isfinite(value):
                scores.append(np.nan)
            else:
                scores.append(round(float((value - others.median()) / scale), 3))
        out[col] = scores
    return out


def audit_cell_qc(
    probes: list[RunProbe],
    findings: Findings | None = None,
    thresholds: Sequence[int] = DEFAULT_COUNT_THRESHOLDS,
) -> CellQC:
    """Compute per-run QC and flag outliers and cross-run imbalance."""
    f = findings if findings is not None else Findings()
    thresholds = tuple(thresholds)

    rows = [_run_metrics(p, thresholds) for p in probes]
    per_run = pd.DataFrame(rows)
    if per_run.empty:
        return CellQC(per_run=per_run, thresholds=thresholds)

    with_cells = per_run[per_run["n_cells"] > 0]
    if with_cells.empty:
        f.warning(
            "qc.no_cell_tables",
            "No run yielded a readable cells table; per-run QC is unavailable.",
        )
        return CellQC(per_run=per_run, thresholds=thresholds)

    z = _robust_z(with_cells, _OUTLIER_METRICS)

    # -- outliers --------------------------------------------------------
    for col in _OUTLIER_METRICS:
        if col not in z.columns:
            continue
        values = pd.to_numeric(with_cells[col], errors="coerce")
        median = values.median()
        if not np.isfinite(median) or median == 0:
            continue

        rel_dev = (values - median).abs() / abs(median)
        big_z = pd.to_numeric(z[col], errors="coerce").abs() > OUTLIER_Z
        big_rel = rel_dev > OUTLIER_MIN_RELATIVE_DEVIATION
        flagged = z.loc[(big_z & big_rel).fillna(False)]
        if flagged.empty:
            continue

        detail = {
            r.run_id: {
                "value": _safe_scalar(values.loc[r.Index]),
                "robust_z": getattr(r, col),
                "relative_deviation": _safe_scalar(rel_dev.loc[r.Index]),
            }
            for r in flagged.itertuples()
        }
        f.warning(
            "qc.outlier_run",
            f"{len(flagged)} run(s) are outliers on {col} "
            f"(|robust z| > {OUTLIER_Z} and more than "
            f"{OUTLIER_MIN_RELATIVE_DEVIATION:.0%} from the median): "
            + ", ".join(
                f"{k} ({v['value']}, {v['relative_deviation']:.0%} off, "
                f"z={v['robust_z']})"
                for k, v in detail.items()
            )
            + f". Study median is {_safe_scalar(median)}.",
            evidence={"metric": col, "median": _safe_scalar(median), "runs": detail},
            run_ids=sorted(detail),
        )

    # -- absolute control-rate check -------------------------------------
    if "control_rate" in with_cells.columns:
        cr = pd.to_numeric(with_cells["control_rate"], errors="coerce")
        high = with_cells.loc[cr > CONTROL_RATE_WARN, ["run_id", "control_rate"]]
        if not high.empty:
            f.warning(
                "qc.high_control_rate",
                f"{len(high)} run(s) have a negative-control rate above "
                f"{CONTROL_RATE_WARN:.2%}: "
                + ", ".join(
                    f"{r.run_id} ({float(r.control_rate):.2%})"
                    for r in high.itertuples()
                )
                + ". High negative-control counts point at background, "
                "decoding trouble or over-segmentation. This is the one quality "
                "metric that does not depend on which add-on panel a run "
                "carried, which is what makes it comparable here.",
                evidence={
                    r.run_id: float(r.control_rate) for r in high.itertuples()
                },
                run_ids=sorted(high["run_id"].tolist()),
            )
        if cr.notna().sum() >= 2:
            f.info(
                "qc.control_rate_range",
                f"Negative-control rate across runs: "
                f"{cr.min():.3%} - {cr.max():.3%} (median {cr.median():.3%}). "
                "This counts designed-negative probes and codewords only. "
                "Unassigned and deprecated codewords are reported separately as "
                "background_rate, because which codewords fall in those classes "
                "depends on the panel design and Ranger version and so is not "
                "comparable across runs.",
                evidence={
                    "min": _safe_scalar(cr.min()),
                    "max": _safe_scalar(cr.max()),
                    "median": _safe_scalar(cr.median()),
                },
            )
        elif cr.notna().sum() == 0:
            f.info(
                "qc.control_rate_unavailable",
                "No control-count columns in any cells table, so the "
                "panel-independent control rate could not be computed. It is the "
                "cleanest way to compare run quality across differing panels — "
                "worth re-exporting cells.parquet from Xenium Ranger if you can.",
            )

    # -- filtering-induced imbalance --------------------------------------
    # If a QC floor would remove very different fractions across runs, the
    # filter itself introduces a between-run difference.
    for t in thresholds:
        col = f"frac_cells_below_{t}"
        if col not in with_cells.columns:
            continue
        v = pd.to_numeric(with_cells[col], errors="coerce").dropna()
        if len(v) < 2:
            continue
        spread = float(v.max() - v.min())
        if spread >= 0.10:
            worst = with_cells.loc[
                pd.to_numeric(with_cells[col], errors="coerce").idxmax(), "run_id"
            ]
            best = with_cells.loc[
                pd.to_numeric(with_cells[col], errors="coerce").idxmin(), "run_id"
            ]
            f.warning(
                "qc.filter_imbalance",
                f"A {t}-transcript floor would remove {v.min():.1%} of cells in "
                f"{best} but {v.max():.1%} in {worst} (spread {spread:.1%}). "
                "Runs would enter the analysis thinned by different amounts, so "
                "the filter becomes a batch effect of its own — choose the "
                "threshold knowing this.",
                evidence={
                    "threshold": t,
                    "min": _safe_scalar(v.min()),
                    "max": _safe_scalar(v.max()),
                    "spread": round(spread, 4),
                    "per_run": {
                        r.run_id: _safe_scalar(getattr(r, col))
                        for r in with_cells.itertuples()
                    },
                },
            )
            break  # one threshold's worth of warning is enough

    # -- empty cells -------------------------------------------------------
    if "frac_cells_zero_transcripts" in with_cells.columns:
        zero = pd.to_numeric(
            with_cells["frac_cells_zero_transcripts"], errors="coerce"
        )
        bad = with_cells.loc[zero > 0.05, ["run_id", "frac_cells_zero_transcripts"]]
        if not bad.empty:
            f.warning(
                "qc.zero_count_cells",
                f"{len(bad)} run(s) have more than 5% of cells with zero "
                "transcripts: "
                + ", ".join(
                    f"{r.run_id} ({float(r.frac_cells_zero_transcripts):.1%})"
                    for r in bad.itertuples()
                )
                + ". Empty segmentation artefacts survive into normalisation "
                "unless they are filtered — no cell filtering exists in the "
                "xenium-spatial pipeline today.",
                evidence={
                    r.run_id: float(r.frac_cells_zero_transcripts)
                    for r in bad.itertuples()
                },
                run_ids=sorted(bad["run_id"].tolist()),
            )

    return CellQC(per_run=per_run, robust_z=z, thresholds=thresholds)


def _safe_scalar(v):
    """Coerce a pandas/NumPy scalar (or 1-element Series) to plain Python."""
    if isinstance(v, pd.Series):
        if v.empty:
            return None
        v = v.iloc[0]
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(fv) else round(fv, 6)
