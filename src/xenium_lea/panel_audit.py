"""
panel_audit.py
--------------
What is on each panel, and which genes are actually comparable across runs.

Every run carries one matrix holding base-panel and add-on genes together;
Xenium does not record which is which. The split is recovered by comparing gene
symbols against the base-panel CSV (``Xenium_mBrain_v1_1_metadata.csv``, 247
genes), the same approach the sibling `xenium-spatial` pipeline uses.

Three outputs matter downstream:

``panel_group``
    Runs sharing an identical add-on gene set get the same label (``P1``,
    ``P2``, ...). Panel heterogeneity is only a threat to the analysis when it
    lines up with condition, and this label is what ``design.py`` tests for that.

``safe_genes``
    Genes present in *every* run. This is the only set on which a cross-run
    number is apples-to-apples, and it is what ``batch_metrics.py`` is restricted
    to. Anything outside it is zero-filled somewhere, and a zero-filled gene is
    indistinguishable from a genuinely non-expressed one — so including it makes
    the *panel difference* register as a batch effect. That confusion is the
    single easiest way to conclude "we have a batch problem" from what is really
    a panel bookkeeping artefact.

``missing_base``
    Base-panel genes absent from a run. Worth checking independently: in
    `xenium-spatial`, ``panel_registry.py`` hard-codes ``zero_filled=False`` for
    every base gene, so a base gene missing from one run is zero-filled and
    never flagged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .findings import Findings
from .probe import RunProbe

logger = logging.getLogger(__name__)

_CSV_GENE_COL = "Genes"
_CSV_ENSEMBL = "Ensembl_ID"
_CSV_ANNOTATION = "Annotation"


def load_base_panel(csv_path: Path | str) -> pd.DataFrame:
    """Load the base-panel metadata CSV, keyed by gene symbol."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if _CSV_GENE_COL not in df.columns:
        raise ValueError(
            f"Column '{_CSV_GENE_COL}' not found in {csv_path}. "
            f"Columns present: {list(df.columns)}"
        )
    df = df.drop_duplicates(subset=[_CSV_GENE_COL]).reset_index(drop=True)
    return df.set_index(_CSV_GENE_COL)


@dataclass
class PanelAudit:
    """Result of the cross-run panel comparison."""

    base_genes: set[str]
    per_run: pd.DataFrame                       # one row per run
    overlap: pd.DataFrame                       # add-on genes x runs, bool
    panel_groups: dict[str, str]                # run_id -> "P1" | "P2" | ...
    group_members: dict[str, list[str]]         # "P1" -> [run_id, ...]
    safe_genes: list[str]                       # present in every run
    retention: pd.DataFrame                     # add-on genes kept by threshold
    missing_base: dict[str, list[str]]          # run_id -> missing base genes
    base_annotation: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def n_panel_groups(self) -> int:
        return len(self.group_members)

    def gene_categories(self) -> pd.DataFrame:
        """
        One row per add-on gene: how many runs carry it, and which.

        ``category`` is ``shared_all`` (in every run), ``shared_partial``
        (in 2..n-1) or ``unique`` (exactly one run).
        """
        if self.overlap.empty:
            return pd.DataFrame(
                columns=["gene", "n_runs", "runs_present", "category"]
            )
        n = self.overlap.shape[1]
        n_runs = self.overlap.sum(axis=1)
        runs_present = self.overlap.apply(
            lambda row: ",".join(c for c in self.overlap.columns if row[c]), axis=1
        )

        def _cat(k: int) -> str:
            if k == n:
                return "shared_all"
            return "shared_partial" if k > 1 else "unique"

        return (
            pd.DataFrame(
                {
                    "gene": self.overlap.index,
                    "n_runs": n_runs.values,
                    "runs_present": runs_present.values,
                }
            )
            .assign(category=lambda d: d["n_runs"].map(_cat))
            .sort_values(["n_runs", "gene"], ascending=[False, True])
            .reset_index(drop=True)
        )


def audit_panels(
    probes: list[RunProbe],
    base_panel: pd.DataFrame,
    findings: Findings | None = None,
) -> PanelAudit:
    """Compare panels across runs and decide what is safely comparable."""
    f = findings if findings is not None else Findings()
    base_genes = set(base_panel.index.astype(str))

    usable = [p for p in probes if p.rna_genes]
    if not usable:
        # A study described only by metrics_summary.csv has no gene lists, and
        # that is a known, stated limitation rather than a failure: panel
        # *identity* still comes through as panel_design_id, and the whole
        # design analysis runs on it. Only add-on gene membership is missing.
        described = [p for p in probes if p.metrics]
        if described:
            f.warning(
                "panel.no_gene_lists",
                f"No run carries a gene list, so add-on panel membership and the "
                f"safe gene set are unavailable. {len(described)}/{len(probes)} "
                "run(s) do report a panel identity in metrics_summary.csv, and "
                "the design analysis uses that — but two runs sharing a "
                "panel_design_id is an assumption here rather than something "
                "verified gene by gene. Add one full bundle per distinct "
                "panel_design_id to check it.",
                evidence={
                    "design_ids": sorted(
                        {
                            str(p.metrics.get("panel_design_id"))
                            for p in described
                            if p.metrics.get("panel_design_id")
                        }
                    )
                },
            )
        else:
            f.error(
                "panel.no_usable_runs",
                "No run yielded a readable gene list or a metrics summary; the "
                "panel audit cannot run.",
            )
        return PanelAudit(
            base_genes=base_genes,
            per_run=pd.DataFrame(),
            overlap=pd.DataFrame(),
            panel_groups={},
            group_members={},
            safe_genes=[],
            retention=pd.DataFrame(),
            missing_base={},
            base_annotation=base_panel,
        )

    run_ids = [p.run_id for p in usable]
    n = len(usable)

    # -- per-run composition -------------------------------------------
    rows = []
    missing_base: dict[str, list[str]] = {}
    for p in usable:
        genes = p.gene_set
        addon = sorted(genes - base_genes)
        missing = sorted(base_genes - genes)
        missing_base[p.run_id] = missing
        rows.append(
            {
                "run_id": p.run_id,
                "mouse_id": p.mouse_id,
                "condition": p.condition,
                "n_rna_total": len(genes),
                "n_base": len(genes & base_genes),
                "n_addon": len(addon),
                "n_base_missing": len(missing),
                "base_complete": len(missing) == 0,
                "n_control_features": p.n_control_features,
                "panel_name": p.experiment.get("panel_name"),
                "panel_design_id": p.experiment.get("panel_design_id"),
            }
        )
    per_run = pd.DataFrame(rows)

    # -- base-panel completeness ---------------------------------------
    incomplete = {r: g for r, g in missing_base.items() if g}
    if incomplete:
        f.error(
            "panel.base_incomplete",
            f"{len(incomplete)} run(s) are missing base-panel genes: "
            + "; ".join(
                f"{r} ({len(g)} missing: {', '.join(g[:5])}"
                f"{' ...' if len(g) > 5 else ''})"
                for r, g in sorted(incomplete.items())
            )
            + ". The base panel is meant to be identical everywhere, so this is "
            "either a different panel version or a corrupted bundle. Note that "
            "a missing base gene is zero-filled and *not* flagged by the "
            "harmonisation step in xenium-spatial, so it would pass silently "
            "into the analysis.",
            evidence={r: g for r, g in sorted(incomplete.items())},
            run_ids=sorted(incomplete),
        )
    else:
        f.info(
            "panel.base_complete",
            f"All {n} runs carry the complete {len(base_genes)}-gene base panel.",
            evidence={"n_base_genes": len(base_genes), "n_runs": n},
        )

    # A Xenium add-on panel has two identities, and they mean different things.
    # panel_predesigned_id names the catalogue base panel and must be shared for
    # the base genes to be comparable at all. panel_design_id names the custom
    # add-on built on top of it, and legitimately differs between orders.
    predesigned = {
        str(p.experiment.get("panel_predesigned_id"))
        for p in usable
        if p.experiment.get("panel_predesigned_id")
    }
    design_ids = {
        str(p.experiment.get("panel_design_id"))
        for p in usable
        if p.experiment.get("panel_design_id")
    }

    if len(predesigned) > 1:
        f.error(
            "panel.multiple_base_panels",
            f"Runs were built on {len(predesigned)} different predesigned base "
            f"panels: {sorted(predesigned)}. The base panel is the part meant to "
            "be identical everywhere, so even the shared genes may not be "
            "measured by the same probes.",
            evidence={"predesigned_panel_ids": sorted(predesigned)},
        )
    elif predesigned:
        f.info(
            "panel.shared_base_design",
            f"All runs are built on the same predesigned base panel "
            f"({predesigned.pop()}), so the base genes are directly comparable.",
        )

    if len(design_ids) > 1:
        f.warning(
            "panel.multiple_addon_designs",
            f"Runs carry {len(design_ids)} different custom add-on designs: "
            f"{sorted(design_ids)}. Different design IDs generally mean "
            "different add-on gene lists — see the add-on overlap below for "
            "what they actually share. Whether this threatens the analysis "
            "depends on how the designs line up with condition.",
            evidence={"panel_design_ids": sorted(design_ids)},
        )

    # -- add-on overlap matrix -----------------------------------------
    all_addon: set[str] = set()
    for p in usable:
        all_addon |= p.gene_set - base_genes

    if all_addon:
        overlap = pd.DataFrame(
            {p.run_id: [g in p.gene_set for g in sorted(all_addon)] for p in usable},
            index=sorted(all_addon),
        )
        overlap = overlap.loc[
            overlap.sum(axis=1).sort_values(ascending=False, kind="stable").index
        ]
    else:
        overlap = pd.DataFrame(index=pd.Index([], name="gene"), columns=run_ids)

    # -- panel groups ---------------------------------------------------
    # Runs with byte-identical add-on sets are one group. Sorting by first
    # appearance keeps labels stable across re-runs of the same manifest.
    signature_to_group: dict[frozenset[str], str] = {}
    panel_groups: dict[str, str] = {}
    group_members: dict[str, list[str]] = {}
    for p in usable:
        sig = frozenset(p.gene_set - base_genes)
        if sig not in signature_to_group:
            signature_to_group[sig] = f"P{len(signature_to_group) + 1}"
        label = signature_to_group[sig]
        panel_groups[p.run_id] = label
        group_members.setdefault(label, []).append(p.run_id)

    per_run["panel_group"] = per_run["run_id"].map(panel_groups)

    if len(group_members) > 1:
        f.warning(
            "panel.heterogeneous_addons",
            f"{len(group_members)} distinct add-on panels across {n} runs: "
            + "; ".join(
                f"{g} = {', '.join(sorted(m))}"
                for g, m in sorted(group_members.items())
            )
            + ". Whether this threatens the analysis depends on how the groups "
            "line up with condition — see the separability verdict.",
            evidence={
                "groups": {g: sorted(m) for g, m in group_members.items()},
                "addon_sizes": {
                    g: len(next(
                        s for s, lbl in signature_to_group.items() if lbl == g
                    ))
                    for g in group_members
                },
            },
        )
    else:
        f.info(
            "panel.homogeneous_addons",
            f"All {n} runs share one add-on panel — panel differences cannot "
            "contribute a batch effect here.",
            evidence={"n_addon_genes": len(all_addon)},
        )

    # -- safe gene set ---------------------------------------------------
    safe = set(usable[0].gene_set)
    for p in usable[1:]:
        safe &= p.gene_set
    safe_genes = sorted(safe)

    n_addon_safe = len(safe - base_genes)
    f.info(
        "panel.safe_gene_set",
        f"{len(safe_genes)} genes are present in every run "
        f"({len(safe & base_genes)} base + {n_addon_safe} add-on). Cross-run "
        "comparisons are restricted to these; the remaining "
        f"{len(all_addon) - n_addon_safe} add-on gene(s) are zero-filled in at "
        "least one run and would otherwise register as a batch effect.",
        evidence={
            "n_safe": len(safe_genes),
            "n_safe_base": len(safe & base_genes),
            "n_safe_addon": n_addon_safe,
            "n_addon_unsafe": len(all_addon) - n_addon_safe,
        },
    )

    # -- retention by threshold ------------------------------------------
    # What each "keep an add-on gene if it appears in >= k runs" threshold buys,
    # so the coverage vs zero-inflation trade-off is explicit rather than a
    # default someone has to trust.
    cats = (
        pd.DataFrame(
            {"gene": sorted(all_addon),
             "n_runs": [int(overlap.loc[g].sum()) for g in sorted(all_addon)]}
        )
        if all_addon
        else pd.DataFrame(columns=["gene", "n_runs"])
    )
    ret_rows = []
    for k in range(1, n + 1):
        kept = int((cats["n_runs"] >= k).sum()) if not cats.empty else 0
        zero_filled = (
            int((n - cats.loc[cats["n_runs"] >= k, "n_runs"]).sum())
            if not cats.empty
            else 0
        )
        ret_rows.append(
            {
                "min_runs": k,
                "addon_genes_kept": kept,
                "zero_filled_columns_total": zero_filled,
                "zero_filled_per_run_avg": round(zero_filled / n, 2) if n else 0.0,
            }
        )
    retention = pd.DataFrame(ret_rows)

    unique_addons = cats[cats["n_runs"] == 1]["gene"].tolist() if not cats.empty else []
    if unique_addons:
        f.info(
            "panel.single_run_addons",
            f"{len(unique_addons)} add-on gene(s) appear in exactly one run "
            f"({', '.join(unique_addons[:8])}"
            f"{' ...' if len(unique_addons) > 8 else ''}). They carry no "
            "cross-run information and are excluded from the safe gene set.",
            evidence={"genes": unique_addons},
        )

    annotation = base_panel.copy()
    if _CSV_ANNOTATION in annotation.columns:
        annotation = annotation[[
            c for c in (_CSV_ENSEMBL, _CSV_ANNOTATION) if c in annotation.columns
        ]]

    return PanelAudit(
        base_genes=base_genes,
        per_run=per_run,
        overlap=overlap,
        panel_groups=panel_groups,
        group_members=group_members,
        safe_genes=safe_genes,
        retention=retention,
        missing_base=missing_base,
        base_annotation=annotation,
    )
