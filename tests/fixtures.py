"""
Synthetic Xenium run bundles.

Real Xenium output is several GB per run and cannot be committed, so the tests
build minimal but *structurally faithful* bundles on disk: the same file names,
the same MTX layout (genes x cells, 1-based coordinates), features.tsv.gz with
control rows interleaved among the RNA targets, a cells table carrying the real
column names, and an experiment.xenium whose key layout matches what Xenium
Ranger writes.

The two segmentation generations are modelled explicitly, because the
morphological fingerprint in ``segmentation_audit`` is what distinguishes them:

``KIT_EXPANSION``  cell_area is derived from nucleus_area by dilating a fixed
                   distance, so the two are near-deterministically coupled.
``KIT_STAIN``      cell_area comes from a traced membrane, independent of the
                   nucleus, so the areas decouple.
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

CONTROL_FEATURES = [
    ("NegControlProbe_00042", "NegControlProbe_00042", "Negative Control Probe"),
    ("NegControlProbe_00043", "NegControlProbe_00043", "Negative Control Probe"),
    ("NegControlCodeword_0500", "NegControlCodeword_0500", "Negative Control Codeword"),
    ("BLANK_0001", "BLANK_0001", "Blank Codeword"),
    ("BLANK_0002", "BLANK_0002", "Blank Codeword"),
]


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
    mean_counts_per_cell: float = 60.0,
    cells_format: str = "parquet",
    include_experiment: bool = True,
    analysis_sw_version: str | None = None,
    run_start_time: str = "2024-03-01T10:00:00",
    instrument_sn: str = "XETG00101",
    panel_name: str = "Xenium Mouse Brain Panel v1.1",
    gene_scale: dict[str, float] | None = None,
    zero_transcript_frac: float = 0.0,
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

    n_control = len(CONTROL_FEATURES)
    control_lambda = max(control_rate * mean_counts_per_cell / n_control, 1e-6)
    matrix[len(genes):, :] = rng.poisson(control_lambda, size=(n_control, n_cells))

    _write_mtx(run_dir / "cell_feature_matrix" / "matrix.mtx.gz", matrix)

    # -- cells table -------------------------------------------------------
    rna_counts = matrix[: len(genes), :].sum(axis=0)
    control_counts = matrix[len(genes):, :].sum(axis=0)

    nucleus_area = rng.gamma(shape=6.0, scale=6.0, size=n_cells) + 8.0
    if segmentation == EXPANSION:
        # Cell boundary = nucleus dilated by a fixed distance. Treating the
        # nucleus as roughly circular, area maps through radius + d.
        d = 5.0
        radius = np.sqrt(nucleus_area / np.pi)
        cell_area = np.pi * (radius + d) ** 2
        cell_area *= rng.normal(1.0, 0.01, size=n_cells)  # tiny measurement noise
        seg_method = np.array(["Interior - Nucleus expansion"] * n_cells)
    else:
        # Traced membrane: area independent of the nucleus.
        cell_area = rng.gamma(shape=4.0, scale=45.0, size=n_cells) + 40.0
        seg_method = np.array(["Boundary"] * n_cells)

    cells = pd.DataFrame(
        {
            "cell_id": [str(i) for i in range(1, n_cells + 1)],
            "x_centroid": rng.uniform(0, 4000, n_cells).astype(np.float32),
            "y_centroid": rng.uniform(0, 4000, n_cells).astype(np.float32),
            "transcript_counts": rna_counts.astype(np.int32),
            "control_probe_counts": matrix[len(genes): len(genes) + 2, :]
            .sum(axis=0)
            .astype(np.int32),
            "control_codeword_counts": matrix[len(genes) + 2: len(genes) + 3, :]
            .sum(axis=0)
            .astype(np.int32),
            "unassigned_codeword_counts": matrix[len(genes) + 3:, :]
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
            meta["segmentation"] = {
                "method": "multimodal cell segmentation",
                "boundary_stain": "ATP1A1/CD45/E-Cadherin",
                "interior_stain": "18S",
                "nucleus_expansion_distance": 0,
            }
        else:
            meta["segmentation_method"] = "nucleus expansion"
            meta["nucleus_expansion_distance"] = 5.0
        (run_dir / "experiment.xenium").write_text(json.dumps(meta, indent=2))

    return run_dir


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
