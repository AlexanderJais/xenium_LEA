"""
probe.py
--------
Tier-0 reader: everything the audit needs *without* opening a count matrix.

Three small files per run answer questions 1-6 of the audit:

    cell_feature_matrix/features.tsv.gz   what is on the panel
    experiment.xenium                     what the instrument says it did
    cells.parquet                         what the segmentation produced

A Xenium bundle is mostly ``transcripts.parquet`` and morphology images, which
together are several GB and which this audit never reads. Tier 0 touches a few
MB per run and runs in seconds.

Two deliberate departures from how the sibling `xenium-spatial` loader reads the
same files:

* **Control features are kept.** That loader slices the matrix to
  ``feature_type == "Gene Expression"`` and discards blank codewords and negative
  controls. Those controls are the cleanest *panel-independent* measure of run
  quality — precisely what you need to compare runs carrying different add-on
  panels — so here they are counted and retained.
* **``experiment.xenium`` is actually parsed.** That loader stores the JSON
  verbatim and never looks inside. Panel identity, software version and the
  declared segmentation settings all live in there.

Parsing is defensive. Key names drift between Xenium Ranger generations, so each
canonical field is resolved against a list of candidate keys over a flattened
view of the JSON, and the full raw key set is retained so anything unrecognised
still surfaces in the report instead of vanishing.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .findings import Findings
from .manifest import RunEntry

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# features.tsv.gz
# --------------------------------------------------------------------------

#: The feature_type value carrying real RNA targets. Everything else is control.
RNA_FEATURE_TYPE = "Gene Expression"

#: Control feature types, in the order they are reported.
CONTROL_FEATURE_TYPES = (
    "Negative Control Probe",
    "Negative Control Codeword",
    "Unassigned Codeword",
    "Genomic Control",
    "Blank Codeword",
    "Deprecated Codeword",
)

# --------------------------------------------------------------------------
# cells.parquet
# --------------------------------------------------------------------------

#: Per-cell count columns. ``transcript_counts`` is gene-expression only; the
#: control columns are counted separately, which is what makes a control *rate*
#: computable per cell.
CELL_COUNT_COLUMNS = (
    "transcript_counts",
    "control_probe_counts",
    "control_codeword_counts",
    "unassigned_codeword_counts",
    "genomic_control_counts",
    "deprecated_codeword_counts",
    "total_counts",
)

#: Per-cell morphology columns. Their presence, and the relationship between
#: them, is the morphological fingerprint of the segmentation method.
CELL_MORPHOLOGY_COLUMNS = (
    "cell_area",
    "nucleus_area",
    "nucleus_count",
    "segmentation_method",
)

_CENTROID_ALIASES = {
    "x_centroid": "centroid_x", "centroid_x": "centroid_x",
    "x": "centroid_x", "x_um": "centroid_x", "centroid_x_um": "centroid_x",
    "y_centroid": "centroid_y", "centroid_y": "centroid_y",
    "y": "centroid_y", "y_um": "centroid_y", "centroid_y_um": "centroid_y",
}

# --------------------------------------------------------------------------
# experiment.xenium — canonical field -> candidate keys (lower-cased)
# --------------------------------------------------------------------------

_EXPERIMENT_FIELDS: dict[str, tuple[str, ...]] = {
    "run_name":          ("run_name", "runname", "experiment_name"),
    "run_start_time":    ("run_start_time", "run_starttime", "start_time", "date"),
    "region_name":       ("region_name", "region"),
    "panel_name":        ("panel_name", "panel"),
    "panel_design_id":   ("panel_design_id", "panel_id", "design_id"),
    "panel_organism":    ("panel_organism", "organism"),
    "panel_tissue_type": ("panel_tissue_type", "tissue_type"),
    "panel_n_predesigned": (
        "panel_num_targets_predesigned", "num_targets_predesigned",
        "panel_num_predesigned",
    ),
    "panel_n_custom": (
        "panel_num_targets_custom", "num_targets_custom", "panel_num_custom",
    ),
    "instrument_sn":         ("instrument_sn", "instrument_serial_number", "instrument"),
    "instrument_sw_version": ("instrument_sw_version", "instrument_software_version"),
    "analysis_sw_version":   ("analysis_sw_version", "analysis_software_version",
                              "software_version", "pipeline_version"),
    "preservation_method":   ("preservation_method", "preservation", "sample_prep"),
    "cassette_name":         ("cassette_name", "cassette"),
    "slide_id":              ("slide_id", "slide", "slide_serial_number"),
    "num_cells":             ("num_cells", "n_cells", "cell_count"),
    "transcripts_per_cell":  ("transcripts_per_cell", "median_transcripts_per_cell"),
    "pixel_size":            ("pixel_size", "pixel_size_um"),
}

#: Any flattened key matching this is captured under ``segmentation_keys`` —
#: 10x has moved and renamed these across releases, so pattern-match rather
#: than enumerate.
_SEGMENTATION_KEY_RE = re.compile(
    r"segment|nucleus_expansion|expansion_distance|boundary_stain|"
    r"cell_stain|multimodal|interior_stain|membrane",
    re.IGNORECASE,
)


def _flatten_json(obj: Any, prefix: str = "") -> dict[str, Any]:
    """
    Flatten nested dicts to ``a.b.c`` keys.

    Segmentation settings have lived at the top level in some Xenium Ranger
    versions and nested under a ``segmentation`` object in others; flattening
    makes one lookup work for both.
    """
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                out.update(_flatten_json(v, key))
            else:
                out[key] = v
    elif prefix:
        out[prefix] = obj
    return out


def parse_experiment_metadata(path: Path | str) -> dict[str, Any]:
    """
    Parse ``experiment.xenium`` into canonical fields.

    Returns a dict with the keys of :data:`_EXPERIMENT_FIELDS` (missing ones set
    to ``None``), plus:

        ``segmentation_keys``  every flattened key/value whose name looks
                               segmentation-related, whatever it is called
        ``raw_keys``           the full flattened key list, so an unrecognised
                               field is visible in the report rather than lost
        ``parse_error``        set when the file is unreadable/not JSON
    """
    path = Path(path)
    result: dict[str, Any] = {k: None for k in _EXPERIMENT_FIELDS}
    result["segmentation_keys"] = {}
    result["raw_keys"] = []
    result["parse_error"] = None

    try:
        raw = json.loads(path.read_text())
    except Exception as e:
        result["parse_error"] = f"{type(e).__name__}: {e}"
        return result

    flat = _flatten_json(raw)
    result["raw_keys"] = sorted(flat)

    # Match on the last path segment so nesting does not defeat the lookup.
    by_leaf: dict[str, Any] = {}
    for key, value in flat.items():
        by_leaf.setdefault(key.rsplit(".", 1)[-1].lower(), value)
        by_leaf.setdefault(key.lower(), value)

    for canonical, candidates in _EXPERIMENT_FIELDS.items():
        for cand in candidates:
            if cand in by_leaf and by_leaf[cand] not in (None, ""):
                result[canonical] = by_leaf[cand]
                break

    result["segmentation_keys"] = {
        k: v for k, v in flat.items() if _SEGMENTATION_KEY_RE.search(k)
    }
    return result


# --------------------------------------------------------------------------
# The probe record
# --------------------------------------------------------------------------

@dataclass
class RunProbe:
    """Everything Tier 0 knows about one run."""

    run_id: str
    mouse_id: str
    section_id: str
    condition: str
    run_dir: Path

    # -- panel (features.tsv.gz)
    rna_genes: list[str] = field(default_factory=list)
    feature_type_counts: dict[str, int] = field(default_factory=dict)
    duplicate_gene_symbols: list[str] = field(default_factory=list)

    # -- run metadata (experiment.xenium)
    experiment: dict[str, Any] = field(default_factory=dict)

    # -- cells (cells.parquet)
    cells: pd.DataFrame | None = None
    cells_source: str | None = None
    cell_columns_available: list[str] = field(default_factory=list)

    # -- bookkeeping
    files_present: dict[str, bool] = field(default_factory=dict)
    ok: bool = True

    # -- convenience ----------------------------------------------------

    @property
    def n_rna(self) -> int:
        return len(self.rna_genes)

    @property
    def n_cells(self) -> int:
        return 0 if self.cells is None else len(self.cells)

    @property
    def n_control_features(self) -> int:
        return sum(
            v for k, v in self.feature_type_counts.items() if k != RNA_FEATURE_TYPE
        )

    @property
    def gene_set(self) -> set[str]:
        return set(self.rna_genes)

    def has_cell_column(self, name: str) -> bool:
        return self.cells is not None and name in self.cells.columns


def _read_features(path: Path) -> pd.DataFrame:
    """
    Read features.tsv.gz.

    Xenium writes three unnamed columns (gene_id, gene_name, feature_type). Some
    exports carry a header; detect it by checking whether the first row's third
    field is a known feature type.
    """
    with gzip.open(path, "rt") as fh:
        first = fh.readline().rstrip("\n").split("\t")

    known = {RNA_FEATURE_TYPE, *CONTROL_FEATURE_TYPES}
    has_header = len(first) < 3 or first[2] not in known

    df = pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        header=0 if has_header else None,
        dtype=str,
    )
    if has_header:
        df.columns = [str(c).strip().lower() for c in df.columns]
        rename = {}
        for c in df.columns:
            if "name" in c or c in ("gene", "symbol"):
                rename[c] = "gene_name"
            elif c in ("id", "gene_id", "ensembl_id", "feature_id"):
                rename[c] = "gene_id"
            elif "type" in c:
                rename[c] = "feature_type"
        df = df.rename(columns=rename)
    else:
        df.columns = (["gene_id", "gene_name", "feature_type"] +
                      [f"extra_{i}" for i in range(len(df.columns) - 3)])

    for col in ("gene_id", "gene_name", "feature_type"):
        if col not in df.columns:
            df[col] = ""
    return df[["gene_id", "gene_name", "feature_type"]]


def _read_cells(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """
    Read only the QC/morphology columns from the cells table.

    Reading selected columns from parquet avoids pulling boundary or embedding
    columns into memory for what is often a million-row table.
    """
    wanted = [
        "cell_id",
        *CELL_COUNT_COLUMNS,
        *CELL_MORPHOLOGY_COLUMNS,
        *_CENTROID_ALIASES,
    ]

    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        available = list(pq.ParquetFile(path).schema.names)
        cols = [c for c in dict.fromkeys(wanted) if c in available]
        df = pd.read_parquet(path, columns=cols or None)
    else:
        compression = "gzip" if path.name.endswith(".gz") else None
        head = pd.read_csv(path, nrows=0, compression=compression)
        available = list(head.columns)
        cols = [c for c in dict.fromkeys(wanted) if c in available]
        df = pd.read_csv(path, compression=compression, usecols=cols or None)

    df = df.rename(
        columns={c: _CENTROID_ALIASES[c] for c in df.columns if c in _CENTROID_ALIASES}
    )
    return df, available


def probe_run(
    entry: RunEntry,
    findings: Findings | None = None,
    load_cells: bool = True,
) -> RunProbe:
    """
    Run the Tier-0 probe on one run. Never raises for bad data — a failure
    becomes a finding and the probe is returned with ``ok=False``.
    """
    f = findings if findings is not None else Findings()

    probe = RunProbe(
        run_id=entry.run_id,
        mouse_id=entry.mouse_id,
        section_id=entry.section_id,
        condition=entry.condition,
        run_dir=entry.run_dir,
        files_present=entry.file_inventory() if entry.run_dir.exists() else {},
    )

    if not entry.run_dir.exists():
        probe.ok = False
        return probe

    # -- features -------------------------------------------------------
    features_path = entry.features_path()
    if features_path is not None:
        try:
            feats = _read_features(features_path)
            probe.feature_type_counts = (
                feats["feature_type"].value_counts().to_dict()
            )
            rna = feats.loc[feats["feature_type"] == RNA_FEATURE_TYPE, "gene_name"]

            # A duplicate symbol is not cosmetic: after the analysis pipeline's
            # var_names_make_unique() the copy becomes "Foo-1", which no longer
            # matches the base panel by name and is silently reclassified as a
            # custom gene. Surface it here.
            dup = rna[rna.duplicated()].unique().tolist()
            probe.duplicate_gene_symbols = sorted(dup)
            probe.rna_genes = sorted(set(rna.tolist()))

            if dup:
                f.warning(
                    "panel.duplicate_gene_symbol",
                    f"{len(dup)} duplicated gene symbol(s) in features.tsv.gz "
                    f"({', '.join(map(str, dup[:5]))}"
                    f"{' ...' if len(dup) > 5 else ''}). A de-duplicated copy is "
                    "renamed 'Foo-1' downstream and then no longer matches the "
                    "base panel by name — it becomes a phantom add-on gene.",
                    evidence={"duplicates": sorted(dup)},
                    run_ids=[entry.run_id],
                )
            if not probe.rna_genes:
                probe.ok = False
                f.error(
                    "panel.no_rna_features",
                    f"features.tsv.gz in {entry.run_dir} contains no "
                    f"'{RNA_FEATURE_TYPE}' rows.",
                    evidence={"feature_types": probe.feature_type_counts},
                    run_ids=[entry.run_id],
                )
        except Exception as e:
            probe.ok = False
            f.error(
                "panel.features_unreadable",
                f"Could not read {features_path}: {type(e).__name__}: {e}",
                evidence={"path": str(features_path)},
                run_ids=[entry.run_id],
            )
    else:
        probe.ok = False

    # -- experiment.xenium ----------------------------------------------
    exp_path = entry.experiment_path()
    if exp_path is not None:
        probe.experiment = parse_experiment_metadata(exp_path)
        if probe.experiment.get("parse_error"):
            f.warning(
                "run.experiment_unparseable",
                f"experiment.xenium in {entry.run_dir} could not be parsed "
                f"({probe.experiment['parse_error']}).",
                evidence={"path": str(exp_path)},
                run_ids=[entry.run_id],
            )

    # -- cells ----------------------------------------------------------
    if load_cells:
        cells_path = entry.cells_path()
        if cells_path is not None:
            try:
                cells, available = _read_cells(cells_path)
                probe.cells = cells
                probe.cells_source = cells_path.name
                probe.cell_columns_available = available

                if not any(c in cells.columns for c in CELL_COUNT_COLUMNS):
                    f.warning(
                        "qc.no_count_columns",
                        f"{cells_path.name} carries none of the expected per-cell "
                        f"count columns {list(CELL_COUNT_COLUMNS)}. Per-cell QC "
                        "for this run is limited to what is present: "
                        f"{available}.",
                        evidence={"columns": available},
                        run_ids=[entry.run_id],
                    )
            except Exception as e:
                f.warning(
                    "qc.cells_unreadable",
                    f"Could not read {cells_path}: {type(e).__name__}: {e}. "
                    "Per-cell QC and the morphology fingerprint are unavailable "
                    "for this run.",
                    evidence={"path": str(cells_path)},
                    run_ids=[entry.run_id],
                )

    # Cross-check the declared panel size against what is actually in the file.
    declared = probe.experiment.get("panel_n_predesigned")
    declared_custom = probe.experiment.get("panel_n_custom")
    if declared is not None and declared_custom is not None and probe.rna_genes:
        try:
            total_declared = int(declared) + int(declared_custom)
            if total_declared != probe.n_rna:
                f.warning(
                    "panel.declared_size_mismatch",
                    f"experiment.xenium declares {total_declared} targets "
                    f"({declared} predesigned + {declared_custom} custom) but "
                    f"features.tsv.gz lists {probe.n_rna} unique RNA targets.",
                    evidence={
                        "declared_predesigned": int(declared),
                        "declared_custom": int(declared_custom),
                        "observed_rna": probe.n_rna,
                    },
                    run_ids=[entry.run_id],
                )
        except (TypeError, ValueError):
            pass

    return probe


def probe_all(
    manifest,
    findings: Findings | None = None,
    load_cells: bool = True,
) -> list[RunProbe]:
    """Probe every run in a manifest, in manifest order."""
    f = findings if findings is not None else Findings()
    probes = []
    for entry in manifest:
        logger.info("Probing run %s (%s)", entry.run_id, entry.run_dir)
        probes.append(probe_run(entry, findings=f, load_cells=load_cells))
    return probes
