"""CSV reader → Recording.

Accepted layouts (auto-detected):

1. Header with ``time`` / ``time_s`` / ``t`` as first column → remaining columns are channels.
   Sample rate is inferred from the median time step.
2. Header without time column → every column is a channel; ``sample_rate`` must be given
   (default 1.0 Hz with a warning via exception if not provided — we require sample_rate kwarg
   or a ``# sample_rate=...`` comment on the first line).
3. No header, numeric only → column 0 is time if it is strictly increasing; else all channels.
"""

from __future__ import annotations

from pathlib import Path
from io import StringIO

import numpy as np

from nagilize.core.model import Channel, Recording

_TIME_NAMES = {"time", "time_s", "t", "times", "timestamp"}


def read_csv(path: str | Path, *, sample_rate: float | None = None) -> Recording:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if not lines:
        raise ValueError(f"CSV is empty: {path}")

    comment_fs = _parse_sample_rate_comment(lines[0])
    data_lines = [ln for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    if not data_lines:
        raise ValueError(f"CSV has no data rows: {path}")

    header, body = _split_header(data_lines)
    table = np.genfromtxt(StringIO("\n".join(body)), delimiter=",", dtype=np.float64)
    if table.ndim == 1:
        table = table.reshape(-1, 1)
    if table.size == 0 or np.all(np.isnan(table)):
        raise ValueError(f"CSV could not be parsed as numbers: {path}")

    time_col = None
    names: list[str]
    if header is not None:
        names = [h.strip() for h in header]
        if names and names[0].lower() in _TIME_NAMES:
            time_col = 0
            ch_names = names[1:] or [f"Ch{i+1}" for i in range(table.shape[1] - 1)]
        else:
            ch_names = names or [f"Ch{i+1}" for i in range(table.shape[1])]
    else:
        # Heuristic: strictly increasing first column → time
        if table.shape[1] >= 2 and _looks_like_time(table[:, 0]):
            time_col = 0
            ch_names = [f"Ch{i+1}" for i in range(table.shape[1] - 1)]
        else:
            ch_names = [f"Ch{i+1}" for i in range(table.shape[1])]

    if time_col is not None:
        t = table[:, time_col]
        data = table[:, time_col + 1 :]
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        fs = _fs_from_time(t)
        if sample_rate is not None:
            fs = float(sample_rate)
        channels = [
            Channel(name=ch_names[i] if i < len(ch_names) else f"Ch{i+1}", data=data[:, i].copy())
            for i in range(data.shape[1])
        ]
        return Recording(
            sample_rate=fs,
            channels=channels,
            source=str(path.resolve()),
            time=t.astype(np.float64),
        )

    fs = float(sample_rate) if sample_rate is not None else comment_fs
    if fs is None:
        raise ValueError(
            "CSV has no time column; pass sample_rate=... or add a comment "
            "'# sample_rate=1000' on the first line"
        )
    channels = [
        Channel(name=ch_names[i] if i < len(ch_names) else f"Ch{i+1}", data=table[:, i].copy())
        for i in range(table.shape[1])
    ]
    return Recording(sample_rate=fs, channels=channels, source=str(path.resolve()))


def _parse_sample_rate_comment(line: str) -> float | None:
    s = line.strip()
    if not s.startswith("#"):
        return None
    # # sample_rate=1000 or # fs=1000
    lower = s.lower().replace(" ", "")
    for key in ("sample_rate=", "fs="):
        if key in lower:
            try:
                return float(lower.split(key, 1)[1].split(",")[0])
            except ValueError:
                return None
    return None


def _split_header(data_lines: list[str]) -> tuple[list[str] | None, list[str]]:
    first = data_lines[0]
    parts = [p.strip() for p in first.split(",")]
    # Header if any field is non-numeric
    def _is_number(tok: str) -> bool:
        try:
            float(tok)
            return True
        except ValueError:
            return False

    if parts and not all(_is_number(p) for p in parts if p != ""):
        return parts, data_lines[1:]
    return None, data_lines


def _looks_like_time(col: np.ndarray) -> bool:
    if col.size < 2 or np.any(np.isnan(col)):
        return False
    d = np.diff(col)
    return bool(np.all(d > 0))


def _fs_from_time(t: np.ndarray) -> float:
    d = np.diff(t)
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        raise ValueError("Cannot infer sample rate from time column")
    dt = float(np.median(d))
    if dt <= 0:
        raise ValueError("Invalid time step in CSV")
    return 1.0 / dt
