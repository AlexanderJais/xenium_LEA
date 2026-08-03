"""
metrics.py
----------
Read ``metrics_summary.csv`` — Xenium Ranger's own per-run QC sheet.

One row, ~50 columns, a few kB. It is the densest description of a run that
exists, and it is available long before anyone finishes copying multi-GB
bundles around, so the audit treats it as a first-class metadata source rather
than an afterthought:

* **Panel identity** — ``panel_design_id`` distinguishes two custom add-on
  designs built on the same predesigned base. Runs can share ``panel_name`` and
  ``predesigned_panel_id`` and still carry different add-on genes.
* **Segmentation, quantified** — ``segmented_cell_stain_frac`` and
  ``segmented_cell_nuc_expansion_frac`` are the instrument's own tally. A run
  with ``stain_frac = 0.0`` and ``nuc_expansion_frac = 1.0`` did not use the
  staining kit; there is nothing to infer.
* **10x's own control rates** — ``adjusted_negative_control_probe_rate`` and
  ``adjusted_negative_control_codeword_rate``, computed per control feature, so
  they are comparable across panels carrying different numbers of controls.
* **Technical covariates that never reach the count matrix** — section
  thickness, transcript density per area, total segmented cell area, the
  fraction of transcripts assigned to a cell. Each of these shifts downstream
  numbers, and none of them is visible from ``cells.parquet``.

A run directory holding nothing but this file is enough for the inventory, the
segmentation call and the separability verdict. Only the add-on *gene lists*
need the bundle.

Fields are read defensively — column names have drifted across Ranger versions,
and a missing column is a ``None``, never an exception.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

#: metrics_summary.csv column -> the canonical name used elsewhere in the audit.
#: Names on the right that also appear in ``probe._EXPERIMENT_FIELDS`` are
#: deliberate: the two sources describe the same things and are cross-checked.
_METRICS_FIELDS: dict[str, tuple[str, ...]] = {
    "run_name":              ("run_name",),
    "region_name":           ("region_name",),
    "cassette_name":         ("cassette_name",),
    "panel_name":            ("panel_name",),
    "panel_design_id":       ("panel_design_id",),
    "panel_predesigned_id":  ("predesigned_panel_id", "panel_predesigned_id"),
    "num_cells":             ("num_cells_detected", "num_cells"),
    "transcripts_per_cell":  ("median_transcripts_per_cell",),
    "median_genes_per_cell": ("median_genes_per_cell",),
    "region_area":           ("region_area",),
    "fraction_transcripts_assigned": ("fraction_transcripts_assigned",),
    # -- segmentation, stated outright
    "segmentation_stain":    ("stain_definition",),
    "frac_stain":            ("segmented_cell_stain_frac",),
    "frac_boundary_stain":   ("segmented_cell_boundary_frac",),
    "frac_interior_stain":   ("segmented_cell_interior_frac",),
    "frac_nuc_expansion":    ("segmented_cell_nuc_expansion_frac",),
    "frac_imported_cells":   ("segmented_cell_imported_frac",),
    # -- technical covariates invisible in the count matrix
    "total_cell_area":       ("total_cell_area",),
    "transcript_density":    ("decoded_transcripts_per_100um2",),
    "nuclear_transcript_density": ("nuclear_transcripts_per_100um2",),
    "section_thickness":     ("thickness_transcripts_high_quality",),
    "frac_empty_cells":      ("fraction_empty_cells",),
    "cells_per_100um2":      ("cells_per_100um2",),
    "fraction_decoded_q20":  ("fraction_transcripts_decoded_q20",),
    # -- 10x's own control rates, per control feature
    "declared_neg_control_probe_rate": ("adjusted_negative_control_probe_rate",),
    "declared_neg_control_codeword_rate": (
        "adjusted_negative_control_codeword_rate",
    ),
    "declared_genomic_control_rate": ("adjusted_genomic_control_probe_rate",),
    "declared_false_positives_per_cell": (
        "estimated_number_of_false_positive_transcripts_per_cell",
    ),
}

#: Numeric canonical fields, coerced on read.
_NUMERIC = {
    "num_cells", "transcripts_per_cell", "median_genes_per_cell", "region_area",
    "fraction_transcripts_assigned", "frac_stain", "frac_boundary_stain",
    "frac_interior_stain", "frac_nuc_expansion", "frac_imported_cells",
    "total_cell_area", "transcript_density", "nuclear_transcript_density",
    "section_thickness", "frac_empty_cells", "cells_per_100um2",
    "fraction_decoded_q20", "declared_neg_control_probe_rate",
    "declared_neg_control_codeword_rate", "declared_genomic_control_rate",
    "declared_false_positives_per_cell",
}

METRICS_FILENAME = "metrics_summary.csv"


def read_metrics_summary(path: Path | str) -> dict[str, Any]:
    """
    Parse ``metrics_summary.csv`` into canonical fields.

    Returns a dict with every key of :data:`_METRICS_FIELDS` (absent columns are
    ``None``), plus ``raw_keys`` so an unrecognised column still surfaces in the
    report, and ``parse_error`` when the file is unreadable.
    """
    path = Path(path)
    result: dict[str, Any] = {k: None for k in _METRICS_FIELDS}
    result["raw_keys"] = []
    result["parse_error"] = None

    try:
        df = pd.read_csv(path)
    except Exception as e:
        result["parse_error"] = f"{type(e).__name__}: {e}"
        return result

    if df.empty:
        result["parse_error"] = "file has a header but no data row"
        return result

    if len(df) > 1:
        # Multi-region runs write one row per region. Without knowing which
        # region this run directory holds, taking the first row silently would
        # be a guess — say so instead.
        logger.warning(
            "%s has %d rows; using the first. If this bundle covers several "
            "regions, point the manifest at the per-region output instead.",
            path.name, len(df),
        )

    row = df.iloc[0]
    lower = {str(c).strip().lower(): c for c in df.columns}
    result["raw_keys"] = sorted(lower)

    for canonical, candidates in _METRICS_FIELDS.items():
        for cand in candidates:
            col = lower.get(cand)
            if col is None:
                continue
            value = row[col]
            if pd.isna(value):
                break
            if canonical in _NUMERIC:
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    break
            else:
                value = str(value).strip()
                if not value:
                    break
            result[canonical] = value
            break

    return result


def segmentation_from_metrics(metrics: dict[str, Any]) -> str | None:
    """
    The segmentation method as ``metrics_summary.csv`` states it.

    ``stain_frac == 0`` with ``nuc_expansion_frac == 1`` is an unambiguous
    "no staining kit" — the distinction the whole segmentation audit is trying
    to establish, given directly.
    """
    stain = metrics.get("frac_stain")
    expansion = metrics.get("frac_nuc_expansion")
    if stain is None and expansion is None:
        return None
    stain = stain or 0.0
    expansion = expansion or 0.0
    if stain > expansion:
        return "stain_kit"
    if expansion > stain:
        return "nucleus_expansion"
    return None


def find_metrics_summary(run_dir: Path) -> Path | None:
    """Locate the metrics sheet in a run directory, if it has one."""
    run_dir = Path(run_dir)
    for name in (METRICS_FILENAME, "metrics_summary.csv.gz"):
        p = run_dir / name
        if p.exists():
            return p
    # A directory holding only the sheet, renamed per region.
    matches = sorted(run_dir.glob("metrics_summary*.csv"))
    return matches[0] if matches else None
