"""
manifest.py
-----------
The study manifest: which Xenium run belongs to which mouse.

Why this file exists at all
---------------------------
Everything else the audit needs — panel contents, segmentation kit, software
version, run date, cell counts — is read off disk. Exactly one fact cannot be:
**which animal a run came from**. Xenium writes nothing that links two sections
of the same mouse, so the manifest is where you supply it.

That link matters more than it looks. With several sections per mouse, sections
are *not* independent replicates: treating them as such (the default in the
sibling `xenium-spatial` pipeline, where ``replicate`` falls back to
``slide_id``) pseudoreplicates and inflates significance. The manifest makes the
nesting explicit so the audit can count real replicates and, in ``design.py``,
find within-mouse technical contrasts.

CSV format
----------
Required columns: ``run_id, mouse_id, condition, run_dir``
Optional:         ``section_id`` (defaults to ``run_id``), plus any override
                  column (e.g. ``panel_group``, ``segmentation_kit``) which is
                  carried through and cross-checked against what is on disk.

Blank lines and ``#`` comment lines are skipped, so the shipped
``manifest_template.csv`` can be edited in place.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from .findings import Findings

REQUIRED_COLUMNS = ("run_id", "mouse_id", "condition", "run_dir")

#: Columns the manifest owns. Anything else is treated as a user override and
#: cross-checked against the value derived from the run directory.
_KNOWN_COLUMNS = set(REQUIRED_COLUMNS) | {"section_id"}

#: Files that make a directory recognisably a Xenium output bundle. The matrix
#: triple is only needed for the Tier-1 (--deep) pass, so its absence is a
#: warning rather than an error; ``cells`` and ``features`` drive Tier 0.
_MATRIX_FILES = (
    "cell_feature_matrix/matrix.mtx.gz",
    "cell_feature_matrix/barcodes.tsv.gz",
)
_FEATURES_FILE = "cell_feature_matrix/features.tsv.gz"
_CELLS_CANDIDATES = ("cells.parquet", "cells.csv.gz", "cells.csv")
_EXPERIMENT_FILE = "experiment.xenium"


@dataclass
class RunEntry:
    """One Xenium run."""

    run_id: str
    mouse_id: str
    condition: str
    run_dir: Path
    section_id: str = ""
    overrides: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.run_dir = Path(self.run_dir)
        if not self.section_id:
            self.section_id = self.run_id

    # -- what's on disk -------------------------------------------------

    def cells_path(self) -> Path | None:
        """First existing cells table, mirroring Xenium's format generations."""
        for name in _CELLS_CANDIDATES:
            p = self.run_dir / name
            if p.exists():
                return p
        return None

    def features_path(self) -> Path | None:
        """
        First present feature-list container.

        Xenium Ranger has shipped the feature list as features.tsv.gz, inside
        cell_feature_matrix.zarr.zip and inside cell_feature_matrix.h5 across
        generations, so all are accepted — see ``features.py``.
        """
        from .features import FEATURE_SOURCES

        for rel, _ in FEATURE_SOURCES:
            p = self.run_dir / rel
            if p.exists():
                return p
        return None

    def experiment_path(self) -> Path | None:
        p = self.run_dir / _EXPERIMENT_FILE
        return p if p.exists() else None

    def metrics_path(self) -> Path | None:
        """Ranger's per-run QC sheet, if the bundle has one; see ``metrics.py``."""
        from .metrics import find_metrics_summary

        return find_metrics_summary(self.run_dir)

    def matrix_path(self) -> Path | None:
        """First present count-matrix container; see ``matrix.py``."""
        from .matrix import MATRIX_SOURCES

        for rel, _ in MATRIX_SOURCES:
            p = self.run_dir / rel
            if p.exists():
                return p
        return None

    def file_inventory(self) -> dict[str, bool]:
        """Which known Xenium files this bundle has. Presence is diagnostic."""
        from .features import FEATURE_SOURCES
        from .matrix import MATRIX_SOURCES

        names = {
            *(rel for rel, _ in FEATURE_SOURCES),
            *(rel for rel, _ in MATRIX_SOURCES),
            *_MATRIX_FILES,
            _EXPERIMENT_FILE,
            *_CELLS_CANDIDATES,
            "cell_boundaries.parquet",
            "nucleus_boundaries.parquet",
            "metrics_summary.csv",
            "analysis_summary.html",
        }
        return {name: (self.run_dir / name).exists() for name in sorted(names)}

    def to_dict(self) -> dict[str, Any]:
        d = {
            "run_id": self.run_id,
            "mouse_id": self.mouse_id,
            "section_id": self.section_id,
            "condition": self.condition,
            "run_dir": str(self.run_dir),
        }
        d.update(self.overrides)
        return d


class RunManifest:
    """Ordered collection of :class:`RunEntry`, with study-level validation."""

    def __init__(self, entries: list[RunEntry] | None = None):
        self._entries: list[RunEntry] = list(entries or [])

    # -- construction ---------------------------------------------------

    def add(
        self,
        run_id: str,
        mouse_id: str,
        condition: str,
        run_dir: Path | str,
        section_id: str = "",
        **overrides: Any,
    ) -> "RunManifest":
        self._entries.append(
            RunEntry(
                run_id=str(run_id).strip(),
                mouse_id=str(mouse_id).strip(),
                condition=str(condition).strip(),
                run_dir=Path(str(run_dir).strip()),
                section_id=str(section_id).strip(),
                overrides={k: v for k, v in overrides.items() if v not in (None, "")},
            )
        )
        return self

    @classmethod
    def from_csv(cls, csv_path: Path | str) -> "RunManifest":
        """
        Read a manifest CSV.

        A header row is required — unlike a positional format, named columns let
        the optional ``section_id`` and arbitrary override columns coexist
        without the caller having to remember an order.
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Manifest not found: {csv_path}")

        # Strip comment/blank lines before parsing so the shipped template can be
        # edited in place without deleting the explanatory header block.
        lines = [
            ln for ln in csv_path.read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        ]
        if not lines:
            raise ValueError(f"Manifest is empty (only comments/blank lines): {csv_path}")

        rows = list(csv.DictReader(lines))
        if not rows:
            raise ValueError(f"Manifest has a header but no run rows: {csv_path}")

        cols = {(c or "").strip().lower() for c in rows[0].keys()}
        missing = [c for c in REQUIRED_COLUMNS if c not in cols]
        if missing:
            raise ValueError(
                f"Manifest {csv_path} is missing required column(s): {missing}.\n"
                f"Columns present: {sorted(cols)}\n"
                f"Required: {list(REQUIRED_COLUMNS)}"
            )

        manifest = cls()
        for row in rows:
            clean = {
                (k or "").strip().lower(): (v.strip() if isinstance(v, str) else v)
                for k, v in row.items()
                if k is not None
            }
            overrides = {
                k: v for k, v in clean.items()
                if k not in _KNOWN_COLUMNS and v not in (None, "")
            }
            manifest.add(
                run_id=clean["run_id"],
                mouse_id=clean["mouse_id"],
                condition=clean["condition"],
                run_dir=clean["run_dir"],
                section_id=clean.get("section_id", "") or "",
                **overrides,
            )
        return manifest

    # -- access ---------------------------------------------------------

    def __iter__(self) -> Iterator[RunEntry]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, i) -> RunEntry:
        return self._entries[i]

    def get(self, run_id: str) -> RunEntry | None:
        return next((e for e in self._entries if e.run_id == run_id), None)

    @property
    def run_ids(self) -> list[str]:
        return [e.run_id for e in self._entries]

    @property
    def mouse_ids(self) -> list[str]:
        return [e.mouse_id for e in self._entries]

    @property
    def conditions(self) -> list[str]:
        return [e.condition for e in self._entries]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([e.to_dict() for e in self._entries])

    def sections_per_mouse(self) -> pd.DataFrame:
        """
        One row per mouse: condition, number of sections, the run_ids.

        This is the replicate structure. ``n_sections > 1`` anywhere means the
        analysis unit must be the mouse, not the section.
        """
        df = self.to_frame()
        if df.empty:
            return pd.DataFrame(
                columns=["mouse_id", "condition", "n_sections", "run_ids"]
            )
        g = (
            df.groupby("mouse_id", sort=True)
            .agg(
                condition=("condition", lambda s: ", ".join(sorted(set(s)))),
                n_sections=("run_id", "size"),
                run_ids=("run_id", lambda s: ", ".join(sorted(s))),
            )
            .reset_index()
        )
        return g

    def mice_per_condition(self) -> pd.DataFrame:
        """Real replicate counts: distinct mice per condition, not sections."""
        df = self.to_frame()
        if df.empty:
            return pd.DataFrame(columns=["condition", "n_mice", "n_sections"])
        return (
            df.groupby("condition", sort=True)
            .agg(n_mice=("mouse_id", "nunique"), n_sections=("run_id", "size"))
            .reset_index()
        )

    # -- validation -----------------------------------------------------

    def validate(self, findings: Findings | None = None) -> Findings:
        """
        Check the manifest itself, before any run is touched.

        Catches the mistakes that would otherwise surface as confusing results
        much later: duplicate run ids, missing directories, a mouse assigned to
        two conditions, and single-mouse conditions that cannot support any
        between-group inference.
        """
        f = findings if findings is not None else Findings()

        if len(self._entries) == 0:
            f.error("manifest.empty", "Manifest contains no runs.")
            return f

        # Duplicate run_id — downstream tables are keyed on it.
        dups = [r for r in set(self.run_ids) if self.run_ids.count(r) > 1]
        if dups:
            f.error(
                "manifest.duplicate_run_id",
                f"Duplicate run_id(s): {sorted(dups)}. run_id must be unique.",
                evidence={"duplicates": sorted(dups)},
            )

        # Duplicate run_dir — the same bundle listed twice inflates n.
        dirs = [str(e.run_dir) for e in self._entries]
        dup_dirs = sorted({d for d in dirs if dirs.count(d) > 1})
        if dup_dirs:
            f.warning(
                "manifest.duplicate_run_dir",
                f"{len(dup_dirs)} run directory/ies are listed more than once. "
                "The same bundle counted twice looks like extra replication but is not.",
                evidence={"duplicated_dirs": dup_dirs},
            )

        # A mouse in two conditions is a manifest error, not a design feature.
        by_mouse: dict[str, set[str]] = {}
        for e in self._entries:
            by_mouse.setdefault(e.mouse_id, set()).add(e.condition)
        crossed = {m: sorted(c) for m, c in by_mouse.items() if len(c) > 1}
        if crossed:
            f.error(
                "manifest.mouse_multiple_conditions",
                f"Mouse/mice assigned to more than one condition: {crossed}. "
                "One animal cannot be both — check mouse_id and condition.",
                evidence={"mice": crossed},
            )

        # Per-run directory checks. Availability gaps are collected and
        # reported once per category rather than once per run.
        metrics_only: list[str] = []
        missing_features: list[str] = []
        missing_cells: list[str] = []
        missing_experiment: list[str] = []
        missing_matrix: list[str] = []

        for e in self._entries:
            if not e.run_dir.exists():
                f.error(
                    "run.dir_missing",
                    f"Run directory does not exist: {e.run_dir}",
                    evidence={"run_dir": str(e.run_dir)},
                    run_ids=[e.run_id],
                )
                continue
            if not e.run_dir.is_dir():
                f.error(
                    "run.not_a_directory",
                    f"Run path is not a directory: {e.run_dir}",
                    evidence={"run_dir": str(e.run_dir)},
                    run_ids=[e.run_id],
                )
                continue

            # File availability is accumulated and reported once per category
            # below. Emitting it per run turns a partially-uploaded study into
            # dozens of near-identical warnings that bury the real findings.
            if e.features_path() is None:
                (missing_features if e.metrics_path() is None
                 else metrics_only).append(e.run_id)
            if e.cells_path() is None:
                missing_cells.append(e.run_id)
            if e.experiment_path() is None:
                missing_experiment.append(e.run_id)
            if e.matrix_path() is None:
                missing_matrix.append(e.run_id)

        n = len(self._entries)

        def _listing(ids: list[str], limit: int = 8) -> str:
            shown = ", ".join(sorted(ids)[:limit])
            return shown + (f" ... (+{len(ids) - limit} more)" if len(ids) > limit else "")

        if missing_features:
            f.error(
                "run.features_missing",
                f"{len(missing_features)}/{n} run(s) have no feature list and no "
                f"metrics_summary.csv, so nothing can be audited for them: "
                f"{_listing(missing_features)}. Looked for "
                "cell_feature_matrix/features.tsv.gz, cell_feature_matrix.zarr.zip, "
                "cell_feature_matrix.h5, gene_panel.json and metrics_summary.csv.",
                evidence={"runs": sorted(missing_features)},
                run_ids=sorted(missing_features),
            )

        if metrics_only:
            # The common state of a study mid-assembly, and a perfectly useful
            # one: the verdict does not need the bundles.
            f.warning(
                "run.metrics_only",
                f"{len(metrics_only)}/{n} run(s) are described by "
                f"metrics_summary.csv alone: {_listing(metrics_only)}. That is "
                "enough for the inventory, the segmentation call, run-level QC "
                "and the full separability verdict. It is not enough for the "
                "add-on gene lists, so the safe gene set and the --deep pass "
                "cover only the runs with full bundles.",
                evidence={"runs": sorted(metrics_only)},
                run_ids=sorted(metrics_only),
            )

        if missing_cells:
            f.warning(
                "run.cells_missing",
                f"{len(missing_cells)}/{n} run(s) have no cells table: "
                f"{_listing(missing_cells)}. Per-cell QC and the morphology "
                "fingerprint are unavailable for them; run-level QC from "
                "metrics_summary.csv still is.",
                evidence={"runs": sorted(missing_cells)},
                run_ids=sorted(missing_cells),
            )

        if missing_experiment:
            f.warning(
                "run.experiment_missing",
                f"{len(missing_experiment)}/{n} run(s) have no experiment.xenium: "
                f"{_listing(missing_experiment)}. Instrument serial, software "
                "versions and chemistry are unknown for them — so any factor "
                "built on those fields separates the runs by *what was uploaded* "
                "rather than by anything about the experiment. Such factors are "
                "excluded from the verdict.",
                evidence={"runs": sorted(missing_experiment)},
                run_ids=sorted(missing_experiment),
            )

        if missing_matrix:
            f.info(
                "run.matrix_missing",
                f"{len(missing_matrix)}/{n} run(s) have no count matrix: "
                f"{_listing(missing_matrix)}. Tier-0 is unaffected; the --deep "
                "pseudobulk pass skips them.",
                evidence={"runs": sorted(missing_matrix)},
                run_ids=sorted(missing_matrix),
            )

        # Replicate structure.
        spm = self.sections_per_mouse()
        multi = spm[spm["n_sections"] > 1]
        if not multi.empty:
            f.info(
                "design.sections_per_mouse",
                f"{len(multi)} mouse/mice contribute more than one section "
                f"({int(multi['n_sections'].sum())} sections from "
                f"{len(multi)} mice). Sections of one animal are not independent "
                "replicates — the analysis unit is the mouse. Their upside is "
                "that a technical factor differing between two sections of the "
                "same mouse gives a controlled comparison (see design.py).",
                evidence={"mice": multi.to_dict("records")},
            )

        mpc = self.mice_per_condition()
        thin = mpc[mpc["n_mice"] < 2]
        if not thin.empty:
            f.error(
                "design.single_mouse_condition",
                "Condition(s) with fewer than 2 mice: "
                + ", ".join(
                    f"{r.condition} (n={int(r.n_mice)} mice, "
                    f"{int(r.n_sections)} sections)"
                    for r in thin.itertuples()
                )
                + ". No between-group inference is possible at the animal level, "
                "however many sections were cut.",
                evidence={"per_condition": mpc.to_dict("records")},
            )

        if len(set(self.conditions)) < 2:
            f.warning(
                "design.single_condition",
                f"All runs share one condition ({self.conditions[0]!r}). "
                "The separability verdict is limited to describing technical "
                "structure — there is no biological contrast to protect.",
                evidence={"condition": self.conditions[0]},
            )

        return f
