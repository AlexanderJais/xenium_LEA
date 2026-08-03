"""
report.py
---------
Turn the audit into files a human reads and a pipeline can gate on.

Writes CSVs (for your own downstream use), ``findings.json`` and
``separability_verdict.json`` (machine-readable), and a self-contained
``report.html`` with figures embedded as base64 PNGs — no external assets, so it
opens from a Mac Finder double-click with no server and no network.

The report leads with the separability verdict, because that is the finding that
determines whether any of the other numbers should be acted on.
"""

from __future__ import annotations

import base64
import html
import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .design import (
    ALIASED,
    CROSSED,
    NESTED,
    OVERALL_BLOCKED,
    OVERALL_CAUTION,
    PARTIAL,
    DesignAudit,
)
from .findings import Findings

logger = logging.getLogger(__name__)

_VERDICT_STYLE = {
    OVERALL_BLOCKED: ("#8B1A1A", "#fdecea", "Not separable"),
    OVERALL_CAUTION: ("#8a6100", "#fff8e1", "Separable, with caveats"),
    "OK": ("#1b5e20", "#e8f5e9", "Separable"),
}

_SEVERITY_STYLE = {
    "error": ("#8B1A1A", "#fdecea"),
    "warning": ("#8a6100", "#fff8e1"),
    "info": ("#1B4F8A", "#e8f0fa"),
}


@dataclass
class AuditResult:
    """Everything the audit produced, ready to write."""

    findings: Findings
    manifest_frame: pd.DataFrame
    panel: Any = None
    cell_qc: Any = None
    segmentation: pd.DataFrame | None = None
    design: DesignAudit | None = None
    metrics_section: Any = None
    metrics_mouse: Any = None


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _apply_style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "figure.facecolor": "white",
        }
    )
    return plt


def fig_panel_overlap(panel) -> str | None:
    """Add-on genes x runs presence map — where the panels diverge."""
    if panel is None or panel.overlap.empty:
        return None
    plt = _apply_style()

    m = panel.overlap.astype(int)
    height = max(2.2, min(10.0, 0.13 * len(m) + 1.2))
    fig, ax = plt.subplots(figsize=(max(4.0, 0.55 * m.shape[1] + 2.0), height))
    ax.imshow(m.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1,
              interpolation="nearest")
    ax.set_xticks(range(m.shape[1]))
    ax.set_xticklabels(m.columns, rotation=90)
    if len(m) <= 60:
        ax.set_yticks(range(len(m)))
        ax.set_yticklabels(m.index, fontsize=6)
    else:
        ax.set_yticks([])
        ax.set_ylabel(f"{len(m)} add-on genes")
    ax.set_title("Add-on gene presence (filled = present in run)", fontsize=9)
    return _fig_to_b64(fig)


def fig_qc_bars(cell_qc) -> str | None:
    """Per-run transcripts/cell and the panel-independent control rate."""
    if cell_qc is None or cell_qc.per_run.empty:
        return None
    df = cell_qc.per_run
    if df["n_cells"].sum() == 0:
        return None
    plt = _apply_style()

    panels = [
        ("median_transcripts_per_cell", "Median transcripts / cell", "#1B4F8A"),
        ("control_rate", "Negative-control rate", "#B44A1E"),
        ("n_cells", "Cells", "#4C7A34"),
    ]
    available = [p for p in panels if p[0] in df.columns
                 and pd.to_numeric(df[p[0]], errors="coerce").notna().any()]
    if not available:
        return None

    fig, axes = plt.subplots(
        1, len(available), figsize=(3.4 * len(available), 2.6 + 0.1 * len(df))
    )
    axes = np.atleast_1d(axes)
    for ax, (col, title, colour) in zip(axes, available):
        v = pd.to_numeric(df[col], errors="coerce")
        ax.barh(df["run_id"].astype(str), v, color=colour, height=0.65)
        ax.set_title(title, fontsize=9)
        ax.invert_yaxis()
        ax.tick_params(labelsize=7)
        med = v.median()
        if np.isfinite(med):
            ax.axvline(med, color="0.35", ls="--", lw=0.8)
    fig.tight_layout()
    return _fig_to_b64(fig)


def fig_segmentation(seg: pd.DataFrame | None) -> str | None:
    """
    Morphological fingerprint per run.

    Top-left (tight nucleus/cell coupling, low ratio spread) is the signature of
    nucleus-expansion segmentation; the opposite corner is a traced membrane.
    """
    if seg is None or seg.empty:
        return None
    if "nucleus_cell_area_spearman" not in seg.columns:
        return None
    x = pd.to_numeric(seg.get("area_ratio_cv"), errors="coerce")
    y = pd.to_numeric(seg["nucleus_cell_area_spearman"], errors="coerce")
    if y.notna().sum() == 0:
        return None
    plt = _apply_style()

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    colours = {
        "nucleus_expansion": "#B44A1E",
        "stain_kit": "#1B4F8A",
        "unknown": "#999999",
    }
    for kit, sub in seg.groupby(seg["segmentation_kit"].astype(str)):
        ax.scatter(
            pd.to_numeric(sub.get("area_ratio_cv"), errors="coerce"),
            pd.to_numeric(sub["nucleus_cell_area_spearman"], errors="coerce"),
            s=48, label=kit, color=colours.get(kit, "#666666"),
            edgecolors="black", linewidths=0.5, zorder=3,
        )
    for r in seg.itertuples():
        xv = getattr(r, "area_ratio_cv", None)
        yv = getattr(r, "nucleus_cell_area_spearman", None)
        if xv is not None and yv is not None and np.isfinite(
            pd.to_numeric(xv, errors="coerce")
        ):
            ax.annotate(str(r.run_id), (xv, yv), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("CV of nucleus/cell area ratio")
    ax.set_ylabel("Spearman(nucleus area, cell area)")
    ax.set_title("Segmentation morphology fingerprint", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return _fig_to_b64(fig)


def fig_pca(metrics, colour_by: str = "condition", shape_by: str = "segmentation_kit") -> str | None:
    """Pseudobulk PC1/PC2, coloured by biology and shaped by segmentation."""
    if metrics is None or metrics.pca_scores.empty or metrics.pca_scores.shape[1] < 2:
        return None
    plt = _apply_style()

    scores = metrics.pca_scores
    meta = metrics.sample_meta.reindex(scores.index)
    ev = metrics.explained_variance

    fig, ax = plt.subplots(figsize=(4.8, 3.9))
    palette = ["#1B4F8A", "#D55E00", "#4C7A34", "#8A4FA0", "#B44A1E", "#666666"]
    markers = ["o", "s", "^", "D", "v", "P"]

    cvals = (meta[colour_by].astype(str) if colour_by in meta.columns
             else pd.Series(["all"] * len(scores), index=scores.index))
    svals = (meta[shape_by].astype(str) if shape_by in meta.columns
             else pd.Series([""] * len(scores), index=scores.index))
    cmap = {v: palette[i % len(palette)] for i, v in enumerate(sorted(cvals.unique()))}
    smap = {v: markers[i % len(markers)] for i, v in enumerate(sorted(svals.unique()))}

    for cv in sorted(cvals.unique()):
        for sv in sorted(svals.unique()):
            m = (cvals == cv) & (svals == sv)
            if not m.any():
                continue
            ax.scatter(
                scores.loc[m, "PC1"], scores.loc[m, "PC2"],
                c=cmap[cv], marker=smap[sv], s=64,
                edgecolors="black", linewidths=0.5, zorder=3,
                label=f"{cv} / {sv}" if sv else str(cv),
            )
    for sid in scores.index:
        ax.annotate(str(sid), (scores.loc[sid, "PC1"], scores.loc[sid, "PC2"]),
                    fontsize=6, xytext=(4, 3), textcoords="offset points")

    ax.axhline(0, color="0.85", lw=0.6, zorder=0)
    ax.axvline(0, color="0.85", lw=0.6, zorder=0)
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)" if ev else "PC1")
    ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)" if len(ev) > 1 else "PC2")
    ax.set_title(
        f"Pseudobulk PCA ({metrics.level} level, {metrics.n_genes_used} safe genes)",
        fontsize=9,
    )
    ax.legend(frameon=False, fontsize=6.5, loc="best")
    fig.tight_layout()
    return _fig_to_b64(fig)


def fig_associations(metrics) -> str | None:
    """PC x factor eta-squared: which factor drives which axis."""
    if metrics is None or metrics.associations.empty:
        return None
    plt = _apply_style()

    pivot = metrics.associations.pivot_table(
        index="factor", columns="pc", values="eta_squared", aggfunc="first"
    )
    if pivot.empty:
        return None
    pivot = pivot.reindex(
        pivot.mean(axis=1).sort_values(ascending=False).index
    )

    fig, ax = plt.subplots(
        figsize=(0.75 * pivot.shape[1] + 2.6, 0.36 * pivot.shape[0] + 1.7)
    )
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="RdPu",
                   vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=7)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="eta$^2$")
    ax.set_title(
        f"Variance of each PC explained by each factor ({metrics.level} level)",
        fontsize=9,
    )
    fig.tight_layout()
    return _fig_to_b64(fig)


def fig_correlation(metrics) -> str | None:
    """Sample-sample correlation on the safe gene set."""
    if metrics is None or metrics.correlations.empty:
        return None
    plt = _apply_style()

    c = metrics.correlations
    fig, ax = plt.subplots(figsize=(0.42 * len(c) + 2.4, 0.42 * len(c) + 2.0))
    im = ax.imshow(c.to_numpy(dtype=float), cmap="viridis", interpolation="nearest")
    ax.set_xticks(range(len(c)))
    ax.set_xticklabels(c.columns, rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(c)))
    ax.set_yticklabels(c.index, fontsize=6.5)
    fig.colorbar(im, ax=ax, shrink=0.75, label="Pearson r")
    ax.set_title(f"Sample correlation ({metrics.level} level)", fontsize=9)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

_CSS = """
:root { color-scheme: light; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
       margin: 0; padding: 0 0 4rem; color: #1c1c1c; background: #fff; line-height: 1.5; }
.wrap { max-width: 1080px; margin: 0 auto; padding: 0 1.5rem; }
header { background: #12293f; color: #fff; padding: 1.8rem 0 1.4rem; margin-bottom: 1.5rem; }
header h1 { margin: 0 0 .3rem; font-size: 1.45rem; font-weight: 600; }
header .sub { opacity: .8; font-size: .85rem; }
h2 { font-size: 1.05rem; margin: 2.2rem 0 .6rem; padding-bottom: .3rem;
     border-bottom: 1px solid #e3e3e3; }
h3 { font-size: .92rem; margin: 1.4rem 0 .4rem; }
p, li { font-size: .87rem; }
.verdict { border-radius: 6px; padding: 1rem 1.1rem; margin: 1rem 0 1.4rem;
           border-left: 5px solid; }
.verdict .label { font-weight: 700; letter-spacing: .04em; text-transform: uppercase;
                  font-size: .78rem; }
.verdict h2 { border: 0; margin: .2rem 0 .4rem; font-size: 1.2rem; }
.finding { border-radius: 4px; padding: .6rem .8rem; margin: .45rem 0;
           border-left: 4px solid; font-size: .85rem; }
.finding .code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                 font-size: .74rem; opacity: .75; }
.finding .runs { font-size: .74rem; opacity: .7; }
table { border-collapse: collapse; width: 100%; font-size: .78rem; margin: .6rem 0 1rem; }
th, td { border-bottom: 1px solid #e8e8e8; padding: .35rem .5rem; text-align: left;
         vertical-align: top; }
th { background: #f6f7f9; font-weight: 600; position: sticky; top: 0; }
tr:hover td { background: #fafbfc; }
.scroll { overflow-x: auto; max-width: 100%; }
img { max-width: 100%; height: auto; display: block; margin: .5rem 0 1.2rem; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .8em;
       background: #f3f4f6; padding: .1em .3em; border-radius: 3px; }
.tag { display: inline-block; padding: .1rem .45rem; border-radius: 3px; font-size: .72rem;
       font-weight: 600; }
.muted { color: #666; font-size: .82rem; }
.note { background: #f6f7f9; border-left: 3px solid #c8ccd2; padding: .6rem .8rem;
        margin: .8rem 0; font-size: .83rem; }
"""

_FACTOR_TAG = {
    ALIASED: ("#8B1A1A", "#fdecea"),
    NESTED: ("#8a6100", "#fff8e1"),
    PARTIAL: ("#8a6100", "#fff8e1"),
    CROSSED: ("#1b5e20", "#e8f5e9"),
    "CONSTANT": ("#444", "#eceff1"),
}

#: Worst-first ordering for the factor table.
_FACTOR_ORDER = {ALIASED: 0, NESTED: 1, PARTIAL: 2, CROSSED: 3}


def _esc(v: Any) -> str:
    return html.escape("" if v is None else str(v))


def _table(df: pd.DataFrame | None, max_rows: int = 200) -> str:
    if df is None or df.empty:
        return '<p class="muted">No data.</p>'
    shown = df.head(max_rows)
    head = "".join(f"<th>{_esc(c)}</th>" for c in shown.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(v)}</td>" for v in row) + "</tr>"
        for row in shown.itertuples(index=False)
    )
    extra = (
        f'<p class="muted">Showing {max_rows} of {len(df)} rows; '
        "the full table is in the CSV alongside this report.</p>"
        if len(df) > max_rows
        else ""
    )
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead>' \
           f"<tbody>{body}</tbody></table></div>{extra}"


def _findings_html(findings: Findings) -> str:
    if len(findings) == 0:
        return '<p class="muted">No findings.</p>'
    out = []
    for f in findings.sorted():
        fg, bg = _SEVERITY_STYLE.get(f.severity, ("#333", "#f4f4f4"))
        runs = (
            f'<div class="runs">runs: {_esc(", ".join(f.run_ids))}</div>'
            if f.run_ids else ""
        )
        out.append(
            f'<div class="finding" style="border-color:{fg};background:{bg}">'
            f'<span class="code">{_esc(f.severity.upper())} &middot; {_esc(f.code)}</span>'
            f"<div>{_esc(f.message)}</div>{runs}</div>"
        )
    return "".join(out)


def _coverage_note(
    shown: Iterable[str], all_runs: Iterable[str], why: str, fix: str = ""
) -> str:
    """
    State a section's coverage whenever it is less than the whole study.

    A table silently listing 1 of 14 runs reads as "the other 13 are missing",
    which is alarming and wrong. Saying which runs a section covers, and why the
    rest are absent, costs one line and removes the ambiguity.
    """
    shown, all_runs = sorted(set(shown)), sorted(set(all_runs))
    missing = [r for r in all_runs if r not in shown]
    if not missing or not all_runs:
        return ""
    listed = ", ".join(missing[:10]) + (
        f" and {len(missing) - 10} more" if len(missing) > 10 else ""
    )
    return (
        f'<div class="note"><b>Covers {len(shown)} of {len(all_runs)} runs.</b> '
        f"{_esc(why)} Not covered here: {_esc(listed)}."
        + (f" {_esc(fix)}" if fix else "")
        + "</div>"
    )


def _img(b64: str | None, caption: str = "") -> str:
    if not b64:
        return ""
    cap = f'<p class="muted">{_esc(caption)}</p>' if caption else ""
    return f'<img alt="{_esc(caption)}" src="data:image/png;base64,{b64}">{cap}'


def _verdict_block(design: DesignAudit | None) -> str:
    if design is None:
        return ""
    fg, bg, label = _VERDICT_STYLE.get(design.overall, ("#333", "#f4f4f4", design.overall))
    aliased = [k for k, v in design.verdicts.items() if v.verdict == ALIASED]
    partial = [k for k, v in design.verdicts.items() if v.verdict == PARTIAL]

    if design.overall == OVERALL_BLOCKED:
        detail = (
            f"<p><b>{_esc(', '.join(aliased))}</b> "
            f"{'is' if len(aliased) == 1 else 'are'} perfectly confounded with "
            "condition. A difference measured between conditions is a mixture of "
            "biology and this technical factor, and no correction method can "
            "separate them — correcting removes the biology with the batch "
            "effect.</p>"
        )
    elif design.overall == OVERALL_CAUTION:
        nested = [k for k, v in design.verdicts.items() if v.verdict == NESTED]
        bits = []
        if partial:
            bits.append(
                f"<b>{_esc(', '.join(partial))}</b> "
                f"{'is' if len(partial) == 1 else 'are'} unbalanced across "
                "conditions — adjust for "
                f"{'it' if len(partial) == 1 else 'them'} explicitly rather than "
                "relying on unsupervised integration"
            )
        if nested:
            bits.append(
                f"<b>{_esc(', '.join(nested))}</b> "
                f"{'is' if len(nested) == 1 else 'are'} nested within condition "
                "— not correctable as a fixed effect, but replicated inside each "
                "condition, so test the condition difference against that "
                "variation using a random effect"
            )
        detail = (
            "<p>No technical factor is perfectly confounded with condition, so "
            "the biological contrast is recoverable. "
            + "; and ".join(bits)
            + ".</p>"
        )
    else:
        detail = (
            "<p>Every technical factor is either constant across the study or "
            "varies within every condition. Batch effects here affect statistical "
            "efficiency, not validity.</p>"
        )

    reps = "".join(
        f"<li><b>{_esc(c)}</b>: {v['n_mice']} mice, {v['n_sections']} sections</li>"
        for c, v in sorted(design.replicates.items())
    )
    return (
        f'<div class="verdict" style="border-color:{fg};background:{bg}">'
        f'<div class="label" style="color:{fg}">Separability verdict</div>'
        f'<h2 style="color:{fg}">{_esc(design.overall)} &mdash; {_esc(label)}</h2>'
        f"{detail}"
        f"<p class=\"muted\">Replicate structure: <ul>{reps}</ul></p>"
        "</div>"
    )


def _factor_table_html(design: DesignAudit | None) -> str:
    if design is None or not design.verdicts:
        return '<p class="muted">No design analysis available.</p>'
    rows = []
    for v in sorted(design.verdicts.values(), key=lambda x: (
        _FACTOR_ORDER.get(x.verdict, 9), x.factor
    )):
        fg, bg = _FACTOR_TAG.get(v.verdict, ("#333", "#eee"))
        rows.append(
            f"<tr><td><code>{_esc(v.factor)}</code></td>"
            f'<td><span class="tag" style="color:{fg};background:{bg}">'
            f"{_esc(v.verdict)}</span></td>"
            f"<td>{v.n_levels}</td>"
            f"<td>{_esc('; '.join(v.levels[:5]))}"
            f"{' &hellip;' if len(v.levels) > 5 else ''}</td>"
            f"<td>{v.cramers_v_vs_condition:.2f}</td>"
            f"<td>{_esc(v.message)}</td></tr>"
        )
    return (
        '<div class="scroll"><table><thead><tr>'
        "<th>Factor</th><th>Verdict</th><th>Levels</th><th>Values</th>"
        "<th>Cram&eacute;r's V vs condition</th><th>What it means</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def build_html(result: AuditResult, title: str = "Xenium dataset audit") -> str:
    """Render the full self-contained HTML report."""
    design = result.design
    panel = result.panel
    counts = result.findings.counts()

    sections: list[str] = []

    # 1. Verdict + findings
    sections.append(_verdict_block(design))
    sections.append(
        f"<h2>Findings "
        f'<span class="muted">({counts["error"]} error, {counts["warning"]} warning, '
        f'{counts["info"]} info)</span></h2>'
        + _findings_html(result.findings)
    )

    # 2. Design
    if design is not None:
        sections.append("<h2>Design &mdash; is each technical factor separable?</h2>")
        sections.append(_factor_table_html(design))
        if design.within_mouse:
            sections.append(
                "<h3>Within-mouse technical contrasts</h3>"
                '<div class="note">Sections of the same animal that differ in a '
                "technical factor. Same biology, differing only technically &mdash; "
                "the cleanest available measurement of a technical effect, even "
                "where it cannot be removed from the condition comparison.</div>"
                + _table(
                    pd.DataFrame(
                        [
                            {
                                "mouse_id": c["mouse_id"],
                                "condition": c["condition"],
                                "factor": c["factor"],
                                "levels": ", ".join(c["levels"]),
                                "runs": ", ".join(
                                    f"{k}={v}" for k, v in c["runs"].items()
                                ),
                            }
                            for c in design.within_mouse
                        ]
                    )
                )
            )
        pairs = pd.DataFrame(design.factor_pairs)
        if not pairs.empty and pairs["aliased"].any():
            sections.append(
                "<h3>Factors aliased with each other</h3>"
                + _table(pairs[pairs["aliased"]])
            )

    # 3. Inventory
    sections.append("<h2>Inventory</h2>")
    sections.append(
        _table(design.factor_table if design is not None else result.manifest_frame)
    )

    # 4. Panel
    all_runs = (
        design.factor_table["run_id"].tolist()
        if design is not None and "run_id" in design.factor_table.columns
        else result.manifest_frame.get("run_id", pd.Series(dtype=str)).tolist()
    )
    if panel is not None and not panel.per_run.empty:
        sections.append("<h2>Gene panels</h2>")
        sections.append(
            _coverage_note(
                panel.per_run["run_id"],
                all_runs,
                "Gene-level panel contents need a run's feature list, which only "
                "the full bundle carries — metrics_summary.csv names the panel "
                "design but not its genes.",
                "Those runs still appear in the inventory, segmentation and QC "
                "sections, and their panel_design_id is used in the design "
                "analysis. Add one full bundle per distinct panel_design_id to "
                "compare add-on genes.",
            )
        )
        sections.append(_table(panel.per_run))
        sections.append(_img(fig_panel_overlap(panel),
                             "Add-on gene presence across runs."))
        if not panel.retention.empty:
            sections.append(
                "<h3>Add-on retention by threshold</h3>"
                '<div class="note">What each "keep an add-on gene present in at '
                "least k runs\" rule buys, and what it costs in zero-filled "
                "columns. Only genes present in <i>every</i> run "
                f"({len(panel.safe_genes)} of them) are used for the cross-run "
                "comparisons below.</div>"
                + _table(panel.retention)
            )

    # 5. Segmentation
    if result.segmentation is not None and not result.segmentation.empty:
        sections.append("<h2>Segmentation</h2>")
        sections.append(
            _coverage_note(
                result.segmentation["run_id"], all_runs,
                "A run needs at least one readable metadata source to be called.",
            )
        )
        sections.append(_table(result.segmentation))
        sections.append(_img(fig_segmentation(result.segmentation)))

    # 6. QC
    if result.cell_qc is not None and not result.cell_qc.per_run.empty:
        sections.append("<h2>Per-run quality</h2>")
        sections.append(
            _coverage_note(
                result.cell_qc.per_run["run_id"], all_runs,
                "Per-run quality needs a cells table or a metrics summary.",
            )
        )
        _qc = result.cell_qc.per_run
        if "qc_source" in _qc.columns and _qc["qc_source"].nunique(dropna=True) > 1:
            sections.append(
                '<div class="note">Runs differ in where their QC comes from '
                "(<code>qc_source</code>): per-cell values are recomputed from "
                "the cells table, run-level ones are taken from Ranger's "
                "metrics_summary.csv. The two are not interchangeable — compare "
                "within a source, not across.</div>"
            )
        sections.append(_img(fig_qc_bars(result.cell_qc),
                             "Dashed line marks the across-run median."))
        sections.append(_table(result.cell_qc.per_run))

    # 7. Deep metrics
    for metrics, heading in (
        (result.metrics_section, "Batch diagnostics &mdash; section level"),
        (result.metrics_mouse, "Batch diagnostics &mdash; mouse level"),
    ):
        if metrics is None or metrics.pca_scores.empty:
            continue
        sections.append(f"<h2>{heading}</h2>")
        sections.append(
            _coverage_note(
                metrics.pseudobulk.index, all_runs,
                "The deep pass needs a count matrix, which only the full bundle "
                "carries.",
            )
            if metrics.level == "section" else ""
        )
        sections.append(
            '<div class="note">Descriptive only &mdash; nothing here is '
            "corrected. Restricted to the "
            f"{metrics.n_genes_used} genes present in every run, so a panel "
            "difference cannot masquerade as a batch effect.</div>"
        )
        sections.append(_img(fig_pca(metrics)))
        sections.append(_img(fig_associations(metrics)))
        sections.append(_img(fig_correlation(metrics)))

    body = "".join(sections)
    return (
        f"<title>{_esc(title)}</title>"
        f"<style>{_CSS}</style>"
        f'<header><div class="wrap"><h1>{_esc(title)}</h1>'
        f'<div class="sub">Read-only audit &mdash; no data is modified and no '
        f"batch correction is applied.</div></div></header>"
        f'<div class="wrap">{body}</div>'
    )


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def write_report(result: AuditResult, out_dir: Path | str) -> dict[str, Path]:
    """Write every output file. Returns a map of name -> path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    def _csv(name: str, df: pd.DataFrame | None):
        if df is None or df.empty:
            return
        p = out / name
        df.to_csv(p, index=isinstance(df.index, pd.MultiIndex) or df.index.name is not None)
        written[name] = p

    design = result.design
    _csv("inventory.csv",
         design.factor_table if design is not None else result.manifest_frame)

    if result.panel is not None:
        _csv("panel_per_run.csv", result.panel.per_run)
        _csv("panel_overlap.csv", result.panel.overlap.astype(int))
        _csv("panel_addon_categories.csv", result.panel.gene_categories())
        _csv("panel_retention.csv", result.panel.retention)
        missing = pd.DataFrame(
            [
                {"run_id": r, "n_missing": len(g), "missing_base_genes": ";".join(g)}
                for r, g in sorted(result.panel.missing_base.items())
            ]
        )
        _csv("panel_missing_base.csv", missing)
        if result.panel.safe_genes:
            p = out / "safe_gene_set.txt"
            p.write_text("\n".join(result.panel.safe_genes) + "\n")
            written["safe_gene_set.txt"] = p

    if result.cell_qc is not None:
        _csv("cell_qc.csv", result.cell_qc.per_run)
        _csv("cell_qc_robust_z.csv", result.cell_qc.robust_z)

    _csv("segmentation_audit.csv", result.segmentation)

    if design is not None:
        _csv("design_factors.csv", design.verdict_frame())
        _csv("design_factor_pairs.csv", pd.DataFrame(design.factor_pairs))
        p = out / "separability_verdict.json"
        p.write_text(json.dumps(design.to_dict(), indent=2, default=str))
        written["separability_verdict.json"] = p

    for metrics, prefix in (
        (result.metrics_section, "batch_section"),
        (result.metrics_mouse, "batch_mouse"),
    ):
        if metrics is None or metrics.pca_scores.empty:
            continue
        _csv(f"{prefix}_pca_scores.csv", metrics.pca_scores.reset_index(names="sample"))
        _csv(f"{prefix}_associations.csv", metrics.associations)
        _csv(f"{prefix}_correlation.csv",
             metrics.correlations.reset_index(names="sample"))

    written["findings.json"] = result.findings.write_json(out / "findings.json")

    html_path = out / "report.html"
    html_path.write_text(build_html(result), encoding="utf-8")
    written["report.html"] = html_path

    return written
