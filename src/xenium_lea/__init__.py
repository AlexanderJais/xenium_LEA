"""
xenium_lea
==========
Read-only audit of a multi-run Xenium study.

Answers, in order: what runs do we have, is the base panel identical everywhere,
which add-on genes are comparable, which runs used the segmentation staining
kit, is run quality comparable — and then the question the rest serves:

    **Are the technical factors separable from the biological contrast?**

Nothing here corrects a batch effect or writes an integrated object. If a
technical factor turns out to be perfectly confounded with condition, the audit
says so; correcting that case would remove the biology along with the batch
effect.

Typical use::

    from xenium_lea import RunManifest, run_audit, write_report

    manifest = RunManifest.from_csv("manifest.csv")
    result = run_audit(manifest, "data/Xenium_mBrain_v1_1_metadata.csv", deep=True)
    write_report(result, "audit_out")

Or from the shell::

    xenium-lea audit --manifest manifest.csv --out audit_out --deep

Submodules are imported lazily so that ``import xenium_lea`` stays cheap and the
Tier-0 audit does not pull in matplotlib or scikit-learn until they are needed.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "Finding",
    "Findings",
    "RunManifest",
    "RunEntry",
    "RunProbe",
    "probe_run",
    "probe_all",
    "load_base_panel",
    "audit_panels",
    "audit_segmentation",
    "audit_cell_qc",
    "audit_design",
    "build_factor_table",
    "run_audit",
    "write_report",
    "write_stratified_report",
    "run_stratified_audit",
    "split_manifest",
    "build_html",
    "AuditResult",
]

_LAZY = {
    "Finding": "findings",
    "Findings": "findings",
    "RunManifest": "manifest",
    "RunEntry": "manifest",
    "RunProbe": "probe",
    "probe_run": "probe",
    "probe_all": "probe",
    "load_base_panel": "panel_audit",
    "audit_panels": "panel_audit",
    "audit_segmentation": "segmentation_audit",
    "audit_cell_qc": "cell_qc",
    "audit_design": "design",
    "build_factor_table": "design",
    "run_audit": "audit",
    "write_report": "report",
    "write_stratified_report": "report",
    "run_stratified_audit": "stratify",
    "split_manifest": "stratify",
    "build_html": "report",
    "AuditResult": "report",
}


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f".{module}", __name__), name)


def __dir__():
    return sorted(__all__)
