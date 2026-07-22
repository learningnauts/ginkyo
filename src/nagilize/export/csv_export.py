"""CSV export for a Recording (full recording for M1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nagilize.core.model import Recording


def channel_header(name: str, unit: str = "") -> str:
    """Column header; non-empty unit becomes ``Name [unit]``."""
    n = (name or "").strip() or "Ch"
    u = (unit or "").strip()
    if u:
        return f"{n} [{u}]"
    return n


def export_csv(recording: Recording, path: str | Path) -> None:
    """Write time + all channels to CSV (header row included)."""
    path = Path(path)
    if not recording.channels:
        raise ValueError("Recording has no channels to export")

    t = recording.time_axis()
    cols = [t] + [ch.data for ch in recording.channels]
    # Ensure equal length
    n = recording.n_samples
    for i, col in enumerate(cols):
        if col.shape[0] != n:
            raise ValueError(f"Column {i} length mismatch: {col.shape[0]} != {n}")

    header = ["time_s"] + [
        channel_header(ch.name, ch.unit) for ch in recording.channels
    ]
    stacked = np.column_stack(cols)
    np.savetxt(
        path,
        stacked,
        delimiter=",",
        header=",".join(header),
        comments="",
        fmt="%.10g",
    )
