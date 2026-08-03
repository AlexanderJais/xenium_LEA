"""
Synthetic Xenium run bundles.

Real Xenium output is several GB per run and cannot be committed, so the tests
build minimal but *structurally faithful* bundles on disk: the same file names,
the same MTX layout (genes x cells, 1-based coordinates), features.tsv.gz with
control rows interleaved among the RNA targets, a cells table carrying the real
column names, and an experiment.xenium whose key layout matches what Xenium
Ranger writes.

The two segmentation generations are modelled explicitly, because the
morphological fingerprint in ``segmentation_audit`` is what distinguishes them —
and the modelling has to match how real runs actually behave, not how they might
be imagined to:

``KIT_EXPANSION``  the nucleus is dilated a fixed distance but stops early at
                   neighbours, so nucleus and cell area end up only *loosely*
                   coupled. What survives the clipping is a hard ceiling: no
                   cell's boundary sits further from its nucleus than the
                   configured distance.
``KIT_STAIN``      a traced membrane, independent of the nucleus and with a
                   heavy right tail, because nothing caps how far a boundary may
                   sit from its nucleus.

The ceiling is the discriminator. An earlier fixture modelled expansion as a
deterministic function of nucleus area, which made the two trivially separable
on correlation — and hid that real expansion is not deterministic at all.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

EXPANSION = "expansion"
STAIN = "stain"

#: True negative controls — designed to bind nothing. These drive the strict
#: negative-control rate, the metric that is comparable across panel versions.
STRICT_CONTROL_FEATURES = [
    ("NegControlProbe_00042", "NegControlProbe_00042", "Negative Control Probe"),
    ("NegControlProbe_00043", "NegControlProbe_00043", "Negative Control Probe"),
    ("NegControlCodeword_0500", "NegControlCodeword_0500", "Negative Control Codeword"),
]

#: Decoding background — which codewords are unassigned or deprecated depends on
#: the panel design and Ranger version, so these are counted separately.
BACKGROUND_CONTROL_FEATURES = [
    ("UnassignedCodeword_0498", "UnassignedCodeword_0498", "Unassigned Codeword"),
    ("DeprecatedCodeword_0334", "DeprecatedCodeword_0334", "Deprecated Codeword"),
]

CONTROL_FEATURES = STRICT_CONTROL_FEATURES + BACKGROUND_CONTROL_FEATURES

#: Xenium Ranger v6 zarr uses a different vocabulary for the same things.
V6_TYPE_VOCAB = {
    "Gene Expression": "gene",
    "Negative Control Probe": "negative_control_probe",
    "Negative Control Codeword": "negative_control_codeword",
    "Unassigned Codeword": "unassigned_codeword",
    "Deprecated Codeword": "deprecated_codeword",
}


def base_gene_names(n: int = 40) -> list[str]:
    """Stand-in base panel, matching the real CSV's leading genes in spirit."""
    return [f"Base{i:03d}" for i in range(1, n + 1)]


def write_base_panel_csv(path: Path, genes: Sequence[str]) -> Path:
    """Write a base-panel metadata CSV with the real column names."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "Genes": list(genes),
            "Ensembl_ID": [f"ENSMUSG{i:011d}" for i in range(len(genes))],
            "Num_Probesets": [8] * len(genes),
            "Codewords": [1] * len(genes),
            "Annotation": ["Neurons"] * len(genes),
        }
    ).to_csv(path, index=False)
    return path


def _write_mtx(path: Path, matrix: np.ndarray) -> None:
    """Write a dense array as a gzipped MatrixMarket coordinate file."""
    rows, cols = np.nonzero(matrix)
    lines = [
        "%%MatrixMarket matrix coordinate integer general",
        "%",
        f"{matrix.shape[0]} {matrix.shape[1]} {len(rows)}",
    ]
    lines += [
        f"{r + 1} {c + 1} {int(matrix[r, c])}" for r, c in zip(rows, cols)
    ]
    with gzip.open(path, "wt") as fh:
        fh.write("\n".join(lines) + "\n")


def make_run(
    run_dir: Path,
    genes: Sequence[str],
    n_cells: int = 400,
    segmentation: str = STAIN,
    seed: int = 0,
    control_rate: float = 0.01,
    background_rate: float = 0.005,
    mean_counts_per_cell: float = 60.0,
    cells_format: str = "parquet",
    include_experiment: bool = True,
    analysis_sw_version: str | None = None,
    run_start_time: str = "2024-03-01T10:00:00",
    instrument_sn: str = "XETG00101",
    panel_name: str = "Xenium Mouse Brain Panel v1.1",
    gene_scale: dict[str, float] | None = None,
    zero_transcript_frac: float = 0.0,
    stain_frac: float = 0.93,
) -> Path:
    """
    Write one synthetic Xenium bundle.

    ``gene_scale`` multiplies specific genes' expression, which is how the batch
    tests plant a known effect on known genes.
    """
    run_dir = Path(run_dir)
    (run_dir / "cell_feature_matrix").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    genes = list(genes)

    # -- features.tsv.gz (controls interleaved, as Xenium writes them) ----
    feature_rows = [(f"ENSMUSG_{g}", g, "Gene Expression") for g in genes]
    feature_rows += CONTROL_FEATURES
    with gzip.open(run_dir / "cell_feature_matrix" / "features.tsv.gz", "wt") as fh:
        for gid, name, ftype in feature_rows:
            fh.write(f"{gid}\t{name}\t{ftype}\n")

    # -- barcodes.tsv.gz --------------------------------------------------
    with gzip.open(run_dir / "cell_feature_matrix" / "barcodes.tsv.gz", "wt") as fh:
        for i in range(1, n_cells + 1):
            fh.write(f"{i}\n")

    # -- counts -----------------------------------------------------------
    n_features = len(feature_rows)
    per_gene = np.full(len(genes), mean_counts_per_cell / max(len(genes), 1))
    if gene_scale:
        for g, s in gene_scale.items():
            if g in genes:
                per_gene[genes.index(g)] *= s

    matrix = np.zeros((n_features, n_cells), dtype=np.int64)
    matrix[: len(genes), :] = rng.poisson(
        per_gene[:, None], size=(len(genes), n_cells)
    )

    n_zero = int(round(zero_transcript_frac * n_cells))
    if n_zero:
        matrix[: len(genes), :n_zero] = 0

    n_strict = len(STRICT_CONTROL_FEATURES)
    n_background = len(BACKGROUND_CONTROL_FEATURES)
    strict_lambda = max(control_rate * mean_counts_per_cell / n_strict, 1e-9)
    background_lambda = max(
        background_rate * mean_counts_per_cell / n_background, 1e-9
    )
    strict_start = len(genes)
    background_start = strict_start + n_strict
    matrix[strict_start:background_start, :] = rng.poisson(
        strict_lambda, size=(n_strict, n_cells)
    )
    matrix[background_start:, :] = rng.poisson(
        background_lambda, size=(n_background, n_cells)
    )

    _write_mtx(run_dir / "cell_feature_matrix" / "matrix.mtx.gz", matrix)

    # -- cells table -------------------------------------------------------
    rna_counts = matrix[: len(genes), :].sum(axis=0)
    control_counts = matrix[len(genes):, :].sum(axis=0)

    nucleus_area = rng.gamma(shape=6.0, scale=6.0, size=n_cells) + 8.0
    if segmentation == EXPANSION:
        # Nucleus dilated by a fixed distance, but stopping early wherever it
        # meets a neighbour. Real runs are dominated by that clipping: it makes
        # cell size depend on local density as much as on nucleus size, so
        # nucleus and cell area end up only loosely coupled. What survives is
        # the *ceiling* -- no cell exceeds the configured distance -- and that
        # is what the fingerprint reads.
        d_max = 5.0
        achieved = d_max * rng.beta(5.0, 2.0, size=n_cells)  # clipped, capped
        radius = np.sqrt(nucleus_area / np.pi)
        cell_area = np.pi * (radius + achieved) ** 2
        seg_method = np.array(["Segmented by nucleus expansion of 5.0\u00b5m"] * n_cells)
    else:
        # Traced membrane: area independent of the nucleus, and crucially with a
        # heavy right tail. Nothing caps how far a boundary may sit from its
        # nucleus, so a few cells run far out -- that unbounded tail is exactly
        # what distinguishes a stain run from a dilated one.
        cell_area = np.exp(rng.normal(np.log(70.0), 0.85, size=n_cells)) + 20.0
        # Real kit runs resolve most cells by *interior* stain, a minority by
        # boundary stain, and fall back to nucleus expansion for the rest.
        seg_method = np.array(
            ["Segmented by interior stain (18S)"] * n_cells, dtype=object
        )
        n_boundary = int(0.02 * n_cells)
        n_fallback = int(0.07 * n_cells)
        seg_method[:n_boundary] = (
            "Segmented by boundary stain (ATP1A1+CD45+E-Cadherin)"
        )
        seg_method[n_boundary: n_boundary + n_fallback] = (
            "Segmented by nucleus expansion of 5.0\u00b5m"
        )

    cells = pd.DataFrame(
        {
            "cell_id": [str(i) for i in range(1, n_cells + 1)],
            "x_centroid": rng.uniform(0, 4000, n_cells).astype(np.float32),
            "y_centroid": rng.uniform(0, 4000, n_cells).astype(np.float32),
            "transcript_counts": rna_counts.astype(np.int32),
            "control_probe_counts": matrix[strict_start: strict_start + 2, :]
            .sum(axis=0)
            .astype(np.int32),
            "control_codeword_counts": matrix[strict_start + 2: background_start, :]
            .sum(axis=0)
            .astype(np.int32),
            "unassigned_codeword_counts": matrix[
                background_start: background_start + 1, :
            ]
            .sum(axis=0)
            .astype(np.int32),
            "deprecated_codeword_counts": matrix[background_start + 1:, :]
            .sum(axis=0)
            .astype(np.int32),
            "total_counts": (rna_counts + control_counts).astype(np.int32),
            "cell_area": cell_area.astype(np.float32),
            "nucleus_area": nucleus_area.astype(np.float32),
        }
    )

    if segmentation == STAIN:
        # Only the newer, kit-based generation writes these.
        cells["nucleus_count"] = np.ones(n_cells, dtype=np.int32)
        cells["segmentation_method"] = seg_method

    if cells_format == "parquet":
        cells.to_parquet(run_dir / "cells.parquet", index=False)
    else:
        cells.to_csv(run_dir / "cells.csv.gz", index=False, compression="gzip")

    # -- experiment.xenium --------------------------------------------------
    if include_experiment:
        if analysis_sw_version is None:
            analysis_sw_version = (
                "xenium-3.0.0" if segmentation == STAIN else "xenium-1.7.1"
            )
        meta = {
            "run_name": run_dir.name,
            "run_start_time": run_start_time,
            "region_name": "Region 1",
            "preservation_method": "FFPE",
            "instrument_sn": instrument_sn,
            "instrument_sw_version": "2.0.0.6",
            "analysis_sw_version": analysis_sw_version,
            "panel_name": panel_name,
            "panel_design_id": "MBv1.1",
            "panel_organism": "Mouse",
            "panel_tissue_type": "Brain",
            "panel_num_targets_predesigned": sum(
                1 for g in genes if g.startswith("Base")
            ),
            "panel_num_targets_custom": sum(
                1 for g in genes if not g.startswith("Base")
            ),
            "num_cells": n_cells,
            "pixel_size": 0.2125,
        }
        if segmentation == STAIN:
            meta["segmentation_stain"] = "Xenium Multi-Tissue Stain"
            meta["segmented_cell_stain_frac"] = stain_frac
            meta["segmented_cell_boundary_frac"] = round(stain_frac * 0.023, 6)
            meta["segmented_cell_interior_frac"] = round(stain_frac * 0.977, 6)
            meta["segmented_cell_nuc_expansion_frac"] = round(1.0 - stain_frac, 6)
            meta["imported_cell_frac"] = 0.0
        else:
            meta["segmentation_method"] = "nucleus expansion"
            meta["nucleus_expansion_distance"] = 5.0
            meta["segmented_cell_stain_frac"] = 0.0
            meta["segmented_cell_nuc_expansion_frac"] = 1.0
        (run_dir / "experiment.xenium").write_text(json.dumps(meta, indent=2))

    return run_dir


def write_zarr_features(
    path: Path,
    genes: Sequence[str],
    n_cells: int = 100,
    include_aggregate: bool = True,
) -> Path:
    """
    Write a minimal ``cell_feature_matrix.zarr.zip`` carrying only the feature
    list, in the v6 layout and vocabulary.

    The audit reads the panel from ``cell_features/.zattrs``, which is plain JSON
    inside the zip, so no zarr library is involved and a metadata-only fixture is
    enough to exercise that path. ``include_aggregate`` appends the synthetic
    ``Total transcripts`` row that v6 adds and the audit must drop.
    """
    import zipfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ids = [f"ENSMUSG_{g}" for g in genes]
    keys = list(genes)
    types = ["gene"] * len(genes)

    for fid, name, ftype in CONTROL_FEATURES:
        ids.append(fid)
        keys.append(name)
        types.append(V6_TYPE_VOCAB[ftype])

    if include_aggregate:
        ids.append("Total transcripts")
        keys.append("Total transcripts")
        types.append("aggregate_gene")

    attrs = {
        "feature_ids": ids,
        "feature_keys": keys,
        "feature_types": types,
        "major_version": 4,
        "minor_version": 1,
        "number_cells": n_cells,
        "number_features": len(ids),
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(".zgroup", json.dumps({"zarr_format": 2}))
        z.writestr("cell_features/.zgroup", json.dumps({"zarr_format": 2}))
        z.writestr("cell_features/.zattrs", json.dumps(attrs, indent=2))
    return path


#: metrics_summary.csv as Xenium Ranger actually writes it. Values are taken
#: from a real run so the parser is exercised against real formatting, including
#: the empty columns a no-stain run leaves behind.
_METRICS_TEMPLATE = {
    "run_name": "20250626_Xv1_Lea_Droppman_Run1",
    "cassette_name": "Slide1",
    "region_name": "F536_1",
    "panel_name": "mBrain_50g",
    "panel_design_id": "7ZBFXR",
    "predesigned_panel_id": "mBrain_v1.1",
    "region_area": 58980160.716,
    "total_cell_area": 21340410.528,
    "total_high_quality_decoded_transcripts": 40498440,
    "fraction_transcripts_decoded_q20": 0.9616,
    "nuclear_transcripts_per_100um2": 272.55,
    "decoded_transcripts_per_100um2": 149.39,
    "adjusted_negative_control_probe_rate": 0.00153,
    "adjusted_negative_control_codeword_rate": 0.000405,
    "adjusted_genomic_control_probe_rate": "",
    "estimated_number_of_false_positive_transcripts_per_cell": 0.3234,
    "num_cells_detected": 123654,
    "fraction_empty_cells": 0.000817,
    "cells_per_100um2": 0.2097,
    "fraction_transcripts_assigned": 0.7872,
    "median_genes_per_cell": 61,
    "median_transcripts_per_cell": 172,
    "thickness_transcripts_high_quality": 5.554,
    "stain_definition": "",
    "segmented_cell_stain_frac": 0.0,
    "segmented_cell_boundary_frac": 0.0,
    "segmented_cell_interior_frac": 0.0,
    "segmented_cell_nuc_expansion_frac": 1.0,
    "segmented_cell_imported_frac": "",
}


def write_metrics_summary(
    path: Path,
    region_name: str = "F536_1",
    run_name: str = "20250626_Xv1_Lea_Droppman_Run1",
    panel_design_id: str = "7ZBFXR",
    stain_frac: float = 0.0,
    **overrides,
) -> Path:
    """
    Write a ``metrics_summary.csv`` for one run.

    ``stain_frac`` drives the segmentation columns coherently: a kit run reports
    a stain definition and splits the remainder into nucleus expansion, while a
    no-kit run reports 100% expansion and leaves the stain columns blank — which
    is exactly what distinguishes the two generations in the real data.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    row = dict(_METRICS_TEMPLATE)
    row["region_name"] = region_name
    row["run_name"] = run_name
    row["panel_design_id"] = panel_design_id
    row["segmented_cell_stain_frac"] = stain_frac
    row["segmented_cell_nuc_expansion_frac"] = round(1.0 - stain_frac, 6)
    if stain_frac > 0:
        row["stain_definition"] = "xenium_cell_segmentation_stains_v1"
        row["segmented_cell_boundary_frac"] = round(stain_frac * 0.06, 6)
        row["segmented_cell_interior_frac"] = round(stain_frac * 0.94, 6)
    row.update(overrides)

    pd.DataFrame([row]).to_csv(path, index=False)
    return path


def build_metrics_study(root: Path, design: Sequence[tuple]) -> tuple[Path, Path]:
    """
    Build a study of metrics-only run directories.

    Each entry is ``(run_id, mouse_id, condition, run_name, panel_design_id,
    stain_frac)``. Returns ``(manifest_path, base_panel_csv)``.
    """
    root = Path(root)
    panel_csv = write_base_panel_csv(root / "base_panel.csv", base_gene_names(20))

    rows = []
    for run_id, mouse_id, condition, run_name, design_id, stain in design:
        d = root / "runs" / run_id
        d.mkdir(parents=True, exist_ok=True)
        write_metrics_summary(
            d / "metrics_summary.csv",
            region_name=run_id,
            run_name=run_name,
            panel_design_id=design_id,
            stain_frac=stain,
        )
        rows.append(
            {"run_id": run_id, "mouse_id": mouse_id,
             "section_id": run_id.rpartition("_")[2] or "s1",
             "condition": condition, "run_dir": str(d)}
        )

    return write_manifest(root / "manifest.csv", rows), panel_csv


def write_manifest(
    path: Path,
    rows: Sequence[dict],
    extra_columns: Sequence[str] = (),
) -> Path:
    """Write a manifest CSV from dicts."""
    path = Path(path)
    cols = ["run_id", "mouse_id", "section_id", "condition", "run_dir", *extra_columns]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df[cols].to_csv(path, index=False)
    return path


def build_study(
    root: Path,
    design: Sequence[dict],
    base_genes: Sequence[str] | None = None,
    n_cells: int = 400,
    **run_kwargs,
) -> tuple[Path, Path, list[str]]:
    """
    Build a whole study from a compact design spec.

    Each entry needs ``run_id, mouse_id, condition`` and may set ``addons``
    (extra gene names), ``segmentation``, and any :func:`make_run` keyword.

    Returns ``(manifest_path, base_panel_csv, base_genes)``.
    """
    root = Path(root)
    base = list(base_genes or base_gene_names())
    panel_csv = write_base_panel_csv(root / "base_panel.csv", base)

    rows = []
    for i, spec in enumerate(design):
        spec = dict(spec)
        addons = spec.pop("addons", [])
        run_id = spec.pop("run_id")
        mouse_id = spec.pop("mouse_id")
        condition = spec.pop("condition")
        section_id = spec.pop("section_id", f"s{i + 1}")
        run_dir = root / run_id
        make_run(
            run_dir,
            genes=base + list(addons),
            n_cells=spec.pop("n_cells", n_cells),
            seed=spec.pop("seed", i),
            **{**run_kwargs, **spec},
        )
        rows.append(
            {
                "run_id": run_id,
                "mouse_id": mouse_id,
                "section_id": section_id,
                "condition": condition,
                "run_dir": str(run_dir),
            }
        )

    manifest_path = write_manifest(root / "manifest.csv", rows)
    return manifest_path, panel_csv, base
