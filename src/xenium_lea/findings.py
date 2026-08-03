"""
findings.py
-----------
The audit's common result type.

Every check in this package reports through a ``Finding`` rather than raising or
printing. That keeps the audit *read-only and total*: a broken run does not abort
the sweep, it becomes a finding and the remaining runs are still audited.

Severity contract
-----------------
    error    Proceeding with the analysis as planned would produce a wrong
             answer. The CLI exits non-zero so the audit can gate a pipeline.
    warning  Real, needs a human decision, but not automatically invalidating.
    info     Worth knowing; no action implied.

``evidence`` carries the machine-readable numbers behind the message so the
report can render them and so tests can assert on them without string matching.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

Severity = Literal["error", "warning", "info"]

_SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}


@dataclass
class Finding:
    """A single audit result."""

    severity: Severity
    code: str
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)
    run_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:  # pragma: no cover - display only
        where = f" [{', '.join(self.run_ids)}]" if self.run_ids else ""
        return f"{self.severity.upper():7s} {self.code}{where}: {self.message}"


class Findings:
    """An ordered, appendable collection of :class:`Finding`."""

    def __init__(self, items: Iterable[Finding] | None = None):
        self._items: list[Finding] = list(items or [])

    # -- construction ---------------------------------------------------

    def add(
        self,
        severity: Severity,
        code: str,
        message: str,
        evidence: dict[str, Any] | None = None,
        run_ids: Iterable[str] | None = None,
    ) -> Finding:
        f = Finding(
            severity=severity,
            code=code,
            message=message,
            evidence=dict(evidence or {}),
            run_ids=list(run_ids or []),
        )
        self._items.append(f)
        return f

    def error(self, code: str, message: str, **kw) -> Finding:
        return self.add("error", code, message, **kw)

    def warning(self, code: str, message: str, **kw) -> Finding:
        return self.add("warning", code, message, **kw)

    def info(self, code: str, message: str, **kw) -> Finding:
        return self.add("info", code, message, **kw)

    def extend(self, other: "Findings | Iterable[Finding]") -> "Findings":
        self._items.extend(other._items if isinstance(other, Findings) else other)
        return self

    # -- access ---------------------------------------------------------

    def __iter__(self) -> Iterator[Finding]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i):
        return self._items[i]

    def by_severity(self, severity: Severity) -> list[Finding]:
        return [f for f in self._items if f.severity == severity]

    def by_code(self, code: str) -> list[Finding]:
        return [f for f in self._items if f.code == code]

    def has(self, code: str) -> bool:
        """True if any finding carries ``code``. Convenience for tests."""
        return any(f.code == code for f in self._items)

    @property
    def has_errors(self) -> bool:
        return any(f.severity == "error" for f in self._items)

    def sorted(self) -> list[Finding]:
        """Errors first, then warnings, then info; stable within a severity."""
        return sorted(self._items, key=lambda f: _SEVERITY_ORDER.get(f.severity, 9))

    def counts(self) -> dict[str, int]:
        return {
            sev: len(self.by_severity(sev))  # type: ignore[arg-type]
            for sev in ("error", "warning", "info")
        }

    # -- output ---------------------------------------------------------

    def to_list(self) -> list[dict[str, Any]]:
        return [f.to_dict() for f in self.sorted()]

    def write_json(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"counts": self.counts(), "findings": self.to_list()}
        path.write_text(json.dumps(payload, indent=2, default=str))
        return path
