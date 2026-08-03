"""
audit.py
--------
The orchestrator: manifest in, :class:`~xenium_lea.report.AuditResult` out.

Order matters. The panel audit produces the panel groups and the safe gene set;
the segmentation audit produces the kit calls; both feed the design analysis,
which decides whether anything measured afterwards can be acted on. The deep
pass runs last and only when asked, because it is the only step that opens a
count matrix.

Every stage is total: a run that fails to read becomes a finding and the sweep
continues. The audit never modifies anything on disk inside a run directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from .batch_metrics import aggregate_to_mouse, analyse_batch, build_pseudobulk
from .cell_qc import DEFAULT_COUNT_THRESHOLDS, audit_cell_qc
from .design import TECHNICAL_FACTORS, audit_design, build_factor_table
from .findings import Findings
from .manifest import RunManifest
from .panel_audit import audit_panels, load_base_panel
from .probe import probe_all
from .report import AuditResult
from .segmentation_audit import audit_segmentation

logger = logging.getLogger(__name__)

#: Factors carried into the PC-association analysis. ``condition`` is included
#: deliberately: seeing the biological factor alongside the technical ones is
#: what makes the comparison interpretable.
_ASSOCIATION_FACTORS = ("condition", "mouse_id", *TECHNICAL_FACTORS)


def run_audit(
    manifest: RunManifest,
    base_panel_csv: Path | str,
    deep: bool = False,
    min_counts_per_cell: int = 0,
    qc_thresholds: Sequence[int] = DEFAULT_COUNT_THRESHOLDS,
    cache_dir: Path | None = None,
) -> AuditResult:
    """Run the full audit."""
    findings = Findings()

    logger.info("Validating manifest (%d runs)", len(manifest))
    manifest.validate(findings)

    logger.info("Tier 0: probing runs")
    probes = probe_all(manifest, findings=findings, load_cells=True)

    logger.info("Auditing gene panels")
    base_panel = load_base_panel(base_panel_csv)
    panel = audit_panels(probes, base_panel, findings=findings)

    logger.info("Auditing segmentation")
    segmentation, _calls = audit_segmentation(probes, findings=findings)

    logger.info("Auditing per-cell quality")
    cell_qc = audit_cell_qc(probes, findings=findings, thresholds=qc_thresholds)

    logger.info("Analysing design separability")
    factor_table = build_factor_table(
        probes,
        panel_groups=panel.panel_groups,
        segmentation=segmentation,
        manifest=manifest,
    )
    design = audit_design(factor_table, findings=findings)

    result = AuditResult(
        findings=findings,
        manifest_frame=manifest.to_frame(),
        panel=panel,
        cell_qc=cell_qc,
        segmentation=segmentation,
        design=design,
    )

    if not deep:
        findings.info(
            "deep.skipped",
            "Deep pass not run. The separability verdict and every panel, "
            "segmentation and QC result above come from metadata alone. Re-run "
            "with --deep to quantify the size of the batch effect from the count "
            "matrices.",
        )
        return result

    if not panel.safe_genes:
        findings.warning(
            "deep.no_safe_genes",
            "No gene is present in every run, so there is no set on which a "
            "cross-run comparison would be apples-to-apples. The deep pass is "
            "skipped — running it anyway would measure the panel difference and "
            "report it as a batch effect.",
        )
        return result

    logger.info("Tier 1: building pseudobulk over %d safe genes", len(panel.safe_genes))
    counts, skipped = build_pseudobulk(
        probes,
        panel.safe_genes,
        findings=findings,
        min_counts_per_cell=min_counts_per_cell,
        cache_dir=cache_dir,
    )
    result.metrics_section = None
    result.metrics_mouse = None

    if counts.empty:
        findings.warning(
            "deep.no_pseudobulk",
            "No run produced a readable count matrix; the deep pass found "
            "nothing to analyse.",
        )
        return result

    meta = factor_table.set_index("run_id").reindex(counts.index)

    result.metrics_section = analyse_batch(
        counts, meta, factors=_ASSOCIATION_FACTORS,
        findings=findings, level="section",
    )

    mouse_counts, mouse_meta = aggregate_to_mouse(counts, meta)
    if len(mouse_counts) >= 3 and len(mouse_counts) < len(counts):
        result.metrics_mouse = analyse_batch(
            mouse_counts, mouse_meta, factors=_ASSOCIATION_FACTORS,
            findings=findings, level="mouse",
        )
    elif len(mouse_counts) == len(counts):
        findings.info(
            "deep.mouse_level_identical",
            "Each mouse contributed one run, so the mouse-level view is "
            "identical to the section-level one and is not repeated.",
        )

    return result
