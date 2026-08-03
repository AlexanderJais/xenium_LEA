"""
cli.py
------
``xenium-lea audit`` — the command-line entry point.

Exits non-zero when any error-severity finding is present, so the audit can gate
a pipeline run rather than merely inform one. Pass ``--no-fail`` to always exit
0 while still writing the report.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .audit import run_audit
from .cell_qc import DEFAULT_COUNT_THRESHOLDS
from .design import (
    ALIASED,
    NESTED,
    OVERALL_BLOCKED,
    OVERALL_CAUTION,
    PARTIAL,
)
from .manifest import RunManifest
from .report import write_report, write_stratified_report
from .stratify import run_stratified_audit

_PANEL_FILENAME = "Xenium_mBrain_v1_1_metadata.csv"


def default_base_panel() -> Path:
    """
    Locate the shipped base-panel CSV.

    Tries the repository checkout first (the usual case, an editable install),
    then the working directory, so running from a copied-out data folder also
    works. Returns the repo path unchanged when nothing is found, letting the
    caller emit one clear "not found" message.
    """
    candidates = [
        Path(__file__).resolve().parents[2] / "data" / _PANEL_FILENAME,
        Path.cwd() / "data" / _PANEL_FILENAME,
        Path.cwd() / _PANEL_FILENAME,
    ]
    return next((c for c in candidates if c.exists()), candidates[0])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="xenium-lea",
        description=(
            "Audit a multi-run Xenium study: inventory, gene panels, "
            "segmentation method, per-run quality, and whether batch effects "
            "are separable from the biological contrast. Read-only — no data is "
            "modified and no batch correction is applied."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("audit", help="Run the audit over a manifest.")
    a.add_argument(
        "--manifest", "-m", required=True, type=Path,
        help="Manifest CSV: run_id, mouse_id, condition, run_dir [, section_id].",
    )
    a.add_argument(
        "--out", "-o", type=Path, default=Path("audit_out"),
        help="Output directory (default: audit_out).",
    )
    a.add_argument(
        "--base-panel", type=Path, default=None,
        help=f"Base panel metadata CSV (default: {default_base_panel()}).",
    )
    a.add_argument(
        "--deep", action="store_true",
        help=(
            "Also read the count matrices to quantify the size of the batch "
            "effect (pseudobulk PCA restricted to genes present in every run). "
            "Slower; results are cached."
        ),
    )
    a.add_argument(
        "--min-counts-per-cell", type=int, default=0,
        help=(
            "Drop cells below this transcript floor before building pseudobulk "
            "(default 0 = no filtering). Runs lose different fractions of cells "
            "to the same floor, so re-run with a value to see how much the "
            "picture depends on it."
        ),
    )
    a.add_argument(
        "--qc-thresholds", type=int, nargs="+", default=list(DEFAULT_COUNT_THRESHOLDS),
        help="Transcript floors previewed in the QC table (default: 10 20 50).",
    )
    a.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Where to cache pseudobulk vectors (default: <out>/cache).",
    )
    a.add_argument(
        "--split-by", metavar="COLUMN", default=None,
        help=(
            "Audit each level of this manifest column separately (e.g. "
            "--split-by sex). Use when a covariate is confounded with a "
            "technical factor: inside a level that factor is constant, so the "
            "confound disappears and the whole panel becomes usable — at the "
            "cost of animals per group, and of any comparison between levels. "
            "The pooled audit is written alongside for comparison."
        ),
    )
    a.add_argument(
        "--no-fail", action="store_true",
        help="Always exit 0, even when error-severity findings are present.",
    )
    a.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    base_panel = args.base_panel or default_base_panel()
    if not Path(base_panel).exists():
        print(
            f"error: base panel CSV not found: {base_panel}\n"
            "Pass --base-panel with the path to Xenium_mBrain_v1_1_metadata.csv.",
            file=sys.stderr,
        )
        return 2

    try:
        manifest = RunManifest.from_csv(args.manifest)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    cache_dir = args.cache_dir or (args.out / "cache")

    if args.split_by:
        strat = run_stratified_audit(
            manifest,
            base_panel_csv=base_panel,
            split_by=args.split_by,
            deep=args.deep,
            min_counts_per_cell=args.min_counts_per_cell,
            qc_thresholds=tuple(args.qc_thresholds),
            cache_dir=cache_dir if args.deep else None,
        )
        written = write_stratified_report(strat, args.out)
        _print_stratified_summary(strat, written, args.out)
        has_errors = strat.findings.has_errors or any(
            s.result.findings.has_errors for s in strat.audited
        )
        return 1 if (has_errors and not args.no_fail) else 0

    result = run_audit(
        manifest,
        base_panel_csv=base_panel,
        deep=args.deep,
        min_counts_per_cell=args.min_counts_per_cell,
        qc_thresholds=tuple(args.qc_thresholds),
        cache_dir=cache_dir if args.deep else None,
    )

    written = write_report(result, args.out)
    _print_summary(result, written, args.out)

    if result.findings.has_errors and not args.no_fail:
        return 1
    return 0


def _print_stratified_summary(strat, written: dict[str, Path], out_dir: Path) -> None:
    import sys as _sys

    print("", file=_sys.stderr)
    print("=" * 72, file=_sys.stderr)
    print(f"STRATIFIED AUDIT — split by '{strat.split_by}'", file=_sys.stderr)
    print("-" * 72, file=_sys.stderr)
    for row in strat.comparison.to_dict("records"):
        print(
            f"  {str(row.get('stratum','')):22s} {str(row.get('verdict','')):12s}"
            f" runs={row.get('n_runs','?'):<3} mice={row.get('n_mice','?'):<3}"
            f" genes={row.get('n_safe_genes','?'):<5}"
            f" [{row.get('mice_per_condition','')}]",
            file=_sys.stderr,
        )
    print("-" * 72, file=_sys.stderr)
    for f in strat.findings.by_severity("error"):
        print(f"  ERROR  {f.code}: {f.message[:150]}", file=_sys.stderr)
    for f in strat.findings.by_severity("warning"):
        print(f"  WARN   {f.code}: {f.message[:150]}", file=_sys.stderr)
    print("-" * 72, file=_sys.stderr)
    print(f"Index:   {written.get('index.html', out_dir)}", file=_sys.stderr)
    print(f"Outputs: {len(written)} file(s) in {out_dir}", file=_sys.stderr)
    print("=" * 72, file=_sys.stderr)


def _print_summary(result, written: dict[str, Path], out_dir: Path) -> None:
    counts = result.findings.counts()
    design = result.design

    print("", file=sys.stderr)
    print("=" * 72, file=sys.stderr)
    if design is not None:
        print(f"SEPARABILITY VERDICT: {design.overall}", file=sys.stderr)
        by_verdict: dict[str, list[str]] = {}
        for k, v in design.verdicts.items():
            by_verdict.setdefault(v.verdict, []).append(k)

        if design.overall == OVERALL_BLOCKED:
            print(
                f"  Confounded with condition: "
                f"{', '.join(by_verdict.get(ALIASED, []))}\n"
                "  These effects cannot be separated from the biology by any "
                "correction method.",
                file=sys.stderr,
            )
        elif design.overall == OVERALL_CAUTION:
            if by_verdict.get(PARTIAL):
                print(
                    f"  Unbalanced across conditions: "
                    f"{', '.join(by_verdict[PARTIAL])}\n"
                    "  Recoverable, but adjust for these explicitly.",
                    file=sys.stderr,
                )
            if by_verdict.get(NESTED):
                print(
                    f"  Nested within condition: {', '.join(by_verdict[NESTED])}\n"
                    "  Not correctable as a fixed effect; test condition against "
                    "this variation as a random effect.",
                    file=sys.stderr,
                )
        else:
            print(
                "  No technical factor is confounded with the biological "
                "contrast.",
                file=sys.stderr,
            )
    print("-" * 72, file=sys.stderr)
    print(
        f"Findings: {counts['error']} error, {counts['warning']} warning, "
        f"{counts['info']} info",
        file=sys.stderr,
    )
    for f in result.findings.by_severity("error"):
        print(f"  ERROR  {f.code}: {f.message[:160]}", file=sys.stderr)
    print("-" * 72, file=sys.stderr)
    print(f"Report:  {written.get('report.html', out_dir)}", file=sys.stderr)
    print(f"Outputs: {len(written)} file(s) in {out_dir}", file=sys.stderr)
    print("=" * 72, file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
