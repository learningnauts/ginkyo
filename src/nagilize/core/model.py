"""Shared signal data model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class SeriesMeta:
    """Vibration-oriented metadata (M3): location, DOF, FRF reference, provenance."""

    quantity: str = "time"  # time | frf | spectrum | other
    point_id: str = ""
    point_name: str = ""
    dof: str = ""  # e.g. X | Y | Z | RX | …
    ref_point_id: str = ""
    ref_point_name: str = ""
    ref_dof: str = ""
    provenance: str = ""  # source memo (path or label); not required to open
    attrs: dict[str, Any] = field(default_factory=dict)

    def response_label(self) -> str:
        point = (self.point_name or self.point_id).strip()
        dof = self.dof.strip()
        if point and dof:
            return f"{point} · {dof}"
        return point or dof

    def reference_label(self) -> str:
        point = (self.ref_point_name or self.ref_point_id).strip()
        dof = self.ref_dof.strip()
        if point and dof:
            return f"{point} · {dof}"
        return point or dof

    def tree_row_text(self, series_name: str) -> str:
        """Label for project tree: name | point·dof | ← ref·dof."""
        parts = [series_name.strip() or "(unnamed)"]
        resp = self.response_label()
        if resp:
            parts.append(resp)
        ref = self.reference_label()
        if ref:
            parts.append(f"← {ref}")
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SeriesMeta:
        if not data:
            return cls()
        attrs = data.get("attrs") or {}
        if not isinstance(attrs, dict):
            attrs = {}
        return cls(
            quantity=str(data.get("quantity") or "time"),
            point_id=str(data.get("point_id") or ""),
            point_name=str(data.get("point_name") or ""),
            dof=str(data.get("dof") or ""),
            ref_point_id=str(data.get("ref_point_id") or ""),
            ref_point_name=str(data.get("ref_point_name") or ""),
            ref_dof=str(data.get("ref_dof") or ""),
            provenance=str(data.get("provenance") or ""),
            attrs=dict(attrs),
        )


@dataclass
class Channel:
    """One time-series channel."""

    name: str
    data: np.ndarray
    unit: str = ""


@dataclass
class Recording:
    """A multi-channel recording with a common sample rate."""

    sample_rate: float
    channels: list[Channel] = field(default_factory=list)
    source: str = ""
    time: np.ndarray | None = None

    @property
    def n_samples(self) -> int:
        if not self.channels:
            return 0
        return int(self.channels[0].data.shape[0])

    @property
    def duration_s(self) -> float:
        t = self.time_axis()
        if t.size == 0:
            return 0.0
        return float(t[-1] - t[0]) if t.size > 1 else 0.0

    def time_axis(self) -> np.ndarray:
        if self.time is not None:
            return np.asarray(self.time, dtype=float)
        n = self.n_samples
        if self.sample_rate <= 0:
            return np.arange(n, dtype=float)
        return np.arange(n, dtype=float) / self.sample_rate
