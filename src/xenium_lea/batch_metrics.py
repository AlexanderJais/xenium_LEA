"""
batch_metrics.py
----------------
Tier 1: how large is the run effect, and which factor drives which axis?

Opt-in (``--deep``), because this is the only part of the audit that opens a
count matrix. Everything here is **descriptive**. Nothing is corrected, no
integrated embedding is written, and no "batch-corrected" object is produced —
by design. Whether correction is even permissible is settled in ``design.py``,
not here.

Two decisions shape the numbers:

**Restricted to the safe gene set.** Only genes present in *every* run are used.
Include a gene that one run's panel lacked and its zero column is
indistinguishable from genuine non-expression, so the pseudobulk PCA cleanly
separates runs by panel and calls it a batch effect. That artefact is entirely
avoidable and would be the most confidently wrong result the audit could
produce, so the restriction is enforced rather than offered as an option.

**Effect sizes, not p-values, lead.** With 6-16 pseudobulk samples, a
Kruskal-Wallis p-value is coarse and a variance-component model is not
identifiable. eta-squared — the share of a PC's variance explained by a factor —
is computable, comparable across factors, and honest about being descriptive.

Pseudobulk is built by streaming each matrix once and reducing to per-gene
totals, then caching them, so a re-run costs nothing.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .findings import Findings
from .probe import RNA_FEATURE_TYPE, RunProbe, _read_features

logger = logging.getLogger(__name__)

#: PCs carried into the association analysis.
N_PCS_REPORTED = 5

#: eta-squared above which a factor is called a dominant driver of a PC.
ETA2_STRONG = 0.5

#: How far eta-squared must exceed its own null expectation before the factor is
#: called a driver. eta-squared is biased upward by the number of levels: under
#: the null a k-level factor explains (k-1)/(n-1) of any variable by arithmetic
#: alone, so with 8 samples a 6-level factor "explains" 71% of every PC while
#: meaning nothing. Comparing against that baseline is what keeps the finding
#: honest at the sample sizes this audit runs at.
ETA2_EXCESS_MIN = 0.25

#: Factors that describe the biology rather than the technology. A biological
#: factor dominating a PC is the wanted outcome, not a batch effect, and is
#: reported as such.
BIOLOGICAL_FACTORS = frozenset({"condition", "mouse_id"})


@dataclass
class BatchMetrics:
    """Descriptive batch diagnostics at section and mouse level."""

    pseudobulk: pd.DataFrame                      # samples x safe genes (raw sums)
    sample_meta: pd.DataFrame                     # one row per pseudobulk sample
    pca_scores: pd.DataFrame = field(default_factory=pd.DataFrame)
    explained_variance: list[float] = field(default_factory=list)
    associations: pd.DataFrame = field(default_factory=pd.DataFrame)
    correlations: pd.DataFrame = field(default_factory=pd.DataFrame)
    level: str = "section"
    n_genes_used: int = 0
    skipped_runs: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pseudobulk
# ---------------------------------------------------------------------------

def _cache_key(matrix_path: Path, min_counts: int) -> str:
    st = matrix_path.stat()
    raw = f"{matrix_path.resolve()}|{st.st_size}|{int(st.st_mtime)}|{min_counts}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def run_pseudobulk(
    probe: RunProbe,
    min_counts_per_cell: int = 0,
    cache_dir: Path | None = None,
) -> tuple[pd.Series, dict[str, Any]] | None:
    """
    Per-gene total counts for one run.

    ``min_counts_per_cell`` drops cells below a transcript floor before summing.
    It defaults to 0 (no filtering) so the audit reports the data as it is; set
    it to see how sensitive the picture is to a QC threshold, which matters
    because runs lose different fractions of cells to the same floor.

    Returns ``(gene_totals, info)`` or ``None`` when the matrix is unavailable.
    """
    matrix_path = probe.run_dir / "cell_feature_matrix" / "matrix.mtx.gz"
    features_path = probe.run_dir / "cell_feature_matrix" / "features.tsv.gz"
    if not matrix_path.exists() or not features_path.exists():
        return None

    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{probe.run_id}.{_cache_key(matrix_path, min_counts_per_cell)}.npz"
        if cache_path.exists():
            try:
                z = np.load(cache_path, allow_pickle=True)
                totals = pd.Series(z["totals"], index=[str(g) for g in z["genes"]])
                info = {
                    "n_cells_total": int(z["n_cells_total"]),
                    "n_cells_used": int(z["n_cells_used"]),
                    "cached": True,
                }
                return totals, info
            except Exception as e:  # pragma: no cover - cache is best-effort
                logger.debug("Cache miss for %s: %s", probe.run_id, e)

    import scipy.io
    import scipy.sparse as sp

    # MTX is genes x cells. Reducing straight to per-gene totals keeps only a
    # length-n_genes vector, whatever the cell count.
    mat = scipy.io.mmread(matrix_path)
    mat = sp.csc_matrix(mat)

    feats = _read_features(features_path)
    if mat.shape[0] != len(feats):
        raise ValueError(
            f"Run {probe.run_id}: matrix has {mat.shape[0]} rows but "
            f"features.tsv.gz lists {len(feats)} features."
        )

    n_cells_total = mat.shape[1]
    if min_counts_per_cell > 0:
        rna_rows = (feats["feature_type"] == RNA_FEATURE_TYPE).to_numpy()
        per_cell = np.asarray(mat[rna_rows, :].sum(axis=0)).ravel()
        keep = per_cell >= min_counts_per_cell
        mat = mat[:, keep]
    n_cells_used = mat.shape[1]

    totals_all = np.asarray(mat.sum(axis=1)).ravel()

    rna_mask = (feats["feature_type"] == RNA_FEATURE_TYPE).to_numpy()
    genes = feats.loc[rna_mask, "gene_name"].astype(str).to_numpy()
    values = totals_all[rna_mask]

    # Duplicate symbols are summed rather than dropped: splitting one gene's
    # counts across two columns would understate it in every downstream ratio.
    totals = pd.Series(values, index=genes).groupby(level=0).sum()

    info = {
        "n_cells_total": int(n_cells_total),
        "n_cells_used": int(n_cells_used),
        "cached": False,
    }

    if cache_path is not None:
        try:
            np.savez_compressed(
                cache_path,
                totals=totals.to_numpy(),
                genes=np.array(totals.index, dtype=object),
                n_cells_total=info["n_cells_total"],
                n_cells_used=info["n_cells_used"],
            )
        except Exception as e:  # pragma: no cover - cache is best-effort
            logger.debug("Could not cache %s: %s", probe.run_id, e)

    return totals, info


def build_pseudobulk(
    probes: Sequence[RunProbe],
    safe_genes: Sequence[str],
    findings: Findings | None = None,
    min_counts_per_cell: int = 0,
    cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Assemble a runs x safe-genes matrix of raw pseudobulk counts."""
    f = findings if findings is not None else Findings()
    safe = list(safe_genes)
    rows: dict[str, pd.Series] = {}
    skipped: list[str] = []

    for p in probes:
        try:
            result = run_pseudobulk(
                p, min_counts_per_cell=min_counts_per_cell, cache_dir=cache_dir
            )
        except Exception as e:
            skipped.append(p.run_id)
            f.warning(
                "deep.pseudobulk_failed",
                f"Could not build pseudobulk for {p.run_id}: "
                f"{type(e).__name__}: {e}",
                run_ids=[p.run_id],
            )
            continue
        if result is None:
            skipped.append(p.run_id)
            continue
        totals, info = result
        rows[p.run_id] = totals.reindex(safe).fillna(0.0)
        logger.info(
            "Pseudobulk %s: %d/%d cells used%s",
            p.run_id, info["n_cells_used"], info["n_cells_total"],
            " (cached)" if info["cached"] else "",
        )

    if skipped:
        f.info(
            "deep.runs_skipped",
            f"{len(skipped)} run(s) had no readable count matrix and are absent "
            f"from the deep analysis: {', '.join(sorted(skipped))}. The Tier-0 "
            "audit still covers them.",
            run_ids=sorted(skipped),
        )

    if not rows:
        return pd.DataFrame(columns=safe), skipped
    return pd.DataFrame(rows).T.reindex(columns=safe).fillna(0.0), skipped


# ---------------------------------------------------------------------------
# Normalisation + PCA
# ---------------------------------------------------------------------------

def normalise_pseudobulk(counts: pd.DataFrame, target_sum: float = 1e6) -> pd.DataFrame:
    """
    CPM + log1p — the same recipe the sibling pipeline uses for its sample PCA,
    so the numbers here are directly comparable to what you see there.
    """
    # copy=True: a zero-copy view of the frame is read-only, and the empty-library
    # guard below writes into it.
    lib = counts.sum(axis=1).to_numpy(dtype=float, copy=True)
    lib[lib == 0] = 1.0
    cpm = counts.to_numpy(dtype=float) / lib[:, None] * target_sum
    return pd.DataFrame(np.log1p(cpm), index=counts.index, columns=counts.columns)


def _eta_squared(values: np.ndarray, groups: Sequence[str]) -> float:
    """
    Share of variance in ``values`` explained by the grouping.

    One-way eta-squared. Preferred over a p-value here: with a handful of
    pseudobulk samples an effect size is interpretable where significance is
    not.
    """
    g = pd.Series(list(groups), dtype="object")
    v = pd.Series(values, dtype=float)
    if g.nunique() < 2 or len(v) < 3:
        return float("nan")
    grand = v.mean()
    ss_total = float(((v - grand) ** 2).sum())
    if ss_total == 0:
        return float("nan")
    ss_between = 0.0
    for _, idx in g.groupby(g).groups.items():
        sub = v.loc[list(idx)]
        ss_between += len(sub) * (float(sub.mean()) - grand) ** 2
    return float(ss_between / ss_total)


def _kruskal_p(values: np.ndarray, groups: Sequence[str]) -> float:
    g = pd.Series(list(groups), dtype="object")
    samples = [values[(g == lvl).to_numpy()] for lvl in g.unique()]
    samples = [s for s in samples if len(s) > 0]
    if len(samples) < 2 or sum(len(s) > 0 for s in samples) < 2:
        return float("nan")
    if all(len(s) == 1 for s in samples):
        return float("nan")
    try:
        return float(stats.kruskal(*samples).pvalue)
    except ValueError:
        return float("nan")


def analyse_batch(
    counts: pd.DataFrame,
    meta: pd.DataFrame,
    factors: Iterable[str],
    findings: Findings | None = None,
    level: str = "section",
    n_pcs: int = N_PCS_REPORTED,
) -> BatchMetrics:
    """
    PCA the pseudobulk and associate every PC with every factor.

    ``meta`` must be indexed by the same sample ids as ``counts``.
    """
    from sklearn.decomposition import PCA

    f = findings if findings is not None else Findings()
    metrics = BatchMetrics(
        pseudobulk=counts, sample_meta=meta, level=level, n_genes_used=counts.shape[1]
    )

    if counts.shape[0] < 3:
        f.info(
            "deep.too_few_samples",
            f"Only {counts.shape[0]} pseudobulk sample(s) at the {level} level — "
            "too few for a meaningful PCA. Batch diagnostics are skipped at this "
            "level.",
        )
        return metrics

    logn = normalise_pseudobulk(counts)
    # Drop genes with no variance; they contribute nothing and make the PCA
    # numerically awkward.
    keep = logn.var(axis=0) > 0
    logn = logn.loc[:, keep]
    metrics.n_genes_used = int(keep.sum())

    if metrics.n_genes_used < 2:
        f.warning(
            "deep.no_variable_genes",
            f"Fewer than 2 genes vary across samples at the {level} level; "
            "PCA is not possible.",
        )
        return metrics

    x = logn.to_numpy(dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    n_comp = int(min(n_pcs, x.shape[0] - 1, x.shape[1]))
    pca = PCA(n_components=n_comp, svd_solver="full", random_state=0)
    scores = pca.fit_transform(x)

    metrics.pca_scores = pd.DataFrame(
        scores,
        index=counts.index,
        columns=[f"PC{i + 1}" for i in range(n_comp)],
    )
    metrics.explained_variance = [
        round(float(v) * 100, 2) for v in pca.explained_variance_ratio_
    ]

    # -- PC x factor associations ----------------------------------------
    rows = []
    n_samples = len(metrics.pca_scores)
    usable_factors: list[str] = []
    degenerate: list[str] = []
    for c in factors:
        if c not in meta.columns:
            continue
        k = meta.loc[metrics.pca_scores.index, c].astype(str).nunique()
        if k <= 1:
            continue
        if k >= n_samples:
            # One level per sample (e.g. mouse_id when every mouse gave one
            # section). Such a factor explains 100% of every PC by construction,
            # which is arithmetic rather than evidence.
            degenerate.append(c)
            continue
        usable_factors.append(c)

    if degenerate:
        f.info(
            "deep.factor_degenerate",
            f"Excluded from the association analysis at the {level} level "
            f"because each sample has its own level: {', '.join(degenerate)}. "
            "Such a factor explains every PC completely by construction, so the "
            "number would be meaningless rather than informative.",
            evidence={"factors": degenerate, "n_samples": n_samples},
        )
    for pc in metrics.pca_scores.columns:
        vals = metrics.pca_scores[pc].to_numpy(dtype=float)
        for col in usable_factors:
            groups = meta.loc[metrics.pca_scores.index, col].astype(str).tolist()
            k = len(set(groups))
            eta2 = _eta_squared(vals, groups)
            # What a k-level factor explains by arithmetic alone at this n.
            null = (k - 1) / (n_samples - 1) if n_samples > 1 else float("nan")
            rows.append(
                {
                    "pc": pc,
                    "pc_variance_pct": metrics.explained_variance[
                        metrics.pca_scores.columns.get_loc(pc)
                    ],
                    "factor": col,
                    "n_levels": k,
                    "eta_squared": round(eta2, 4),
                    "eta_squared_null": round(null, 4),
                    "eta_squared_excess": round(eta2 - null, 4),
                    "kruskal_p": _round_p(_kruskal_p(vals, groups)),
                }
            )
    metrics.associations = pd.DataFrame(rows)

    # -- sample correlation ----------------------------------------------
    metrics.correlations = pd.DataFrame(
        np.corrcoef(logn.to_numpy(dtype=float)),
        index=counts.index,
        columns=counts.index,
    ).round(4)

    _report_associations(metrics, f, level)
    return metrics


def _round_p(p: float) -> float | None:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return None
    return float(f"{p:.3g}")


def _report_associations(metrics: BatchMetrics, f: Findings, level: str) -> None:
    """Turn the association table into readable findings."""
    assoc = metrics.associations
    if assoc.empty:
        return

    # Both bars must clear: a high absolute share, and a share clearly beyond
    # what the factor's level count buys for free.
    strong = assoc[
        (pd.to_numeric(assoc["eta_squared"], errors="coerce") >= ETA2_STRONG)
        & (
            pd.to_numeric(assoc["eta_squared_excess"], errors="coerce")
            >= ETA2_EXCESS_MIN
        )
    ].sort_values(["pc", "eta_squared"], ascending=[True, False])

    if strong.empty:
        f.info(
            "deep.no_dominant_factor",
            f"At the {level} level, no factor explains more than "
            f"{ETA2_STRONG:.0%} of a leading PC beyond what its number of levels "
            "would give by chance. There is no dominant technical axis in the "
            "pseudobulk.",
            evidence={"n_genes_used": metrics.n_genes_used},
        )
        return

    for pc, sub in strong.groupby("pc", sort=True):
        drivers = ", ".join(
            f"{r.factor} (eta2={r.eta_squared:.2f} vs "
            f"{r.eta_squared_null:.2f} expected by chance)"
            for r in sub.itertuples()
        )
        var_pct = float(sub["pc_variance_pct"].iloc[0])
        technical = sorted(set(sub["factor"]) - BIOLOGICAL_FACTORS)
        f.add(
            "warning" if technical else "info",
            "deep.pc_driver",
            f"At the {level} level, {pc} ({var_pct:.1f}% of variance) is "
            f"dominated by: {drivers}."
            + (
                " A technical factor dominating a leading axis is the batch "
                "effect made visible. Whether it can be removed is the "
                "separability verdict's call, not this number's."
                if technical
                else " Those are biological factors — a leading axis reflecting "
                "the biology is the wanted outcome, not a batch effect."
            ),
            evidence={
                "pc": pc,
                "variance_pct": var_pct,
                "technical_drivers": technical,
                "drivers": sub[
                    ["factor", "eta_squared", "eta_squared_null",
                     "eta_squared_excess", "kruskal_p"]
                ].to_dict("records"),
                "level": level,
            },
        )


def aggregate_to_mouse(
    counts: pd.DataFrame, meta: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sum section pseudobulk to the mouse level.

    Sections of one animal are not independent replicates, so the mouse-level
    view is the one that matches the analysis unit. Both are reported: a factor
    that dominates at the section level but vanishes at the mouse level is a
    within-animal technical effect rather than a study-level one.
    """
    if counts.empty or "mouse_id" not in meta.columns:
        return counts, meta

    mouse_of = meta.loc[counts.index, "mouse_id"].astype(str)
    summed = counts.groupby(mouse_of.to_numpy()).sum()
    summed.index.name = "mouse_id"

    # Keep only factors that are constant within a mouse; anything varying
    # within an animal has no single value at this level.
    agg_meta = (
        meta.assign(_mouse=meta["mouse_id"].astype(str))
        .groupby("_mouse")
        .agg(lambda s: s.iloc[0] if s.nunique() == 1 else "mixed")
    )
    agg_meta.index.name = "mouse_id"
    return summed, agg_meta.reindex(summed.index)
