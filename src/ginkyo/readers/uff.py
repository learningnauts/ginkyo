"""UFF / UNV reader (dataset 58 time responses via pyuff)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ginkyo.core.model import Channel, Recording


def read_uff(path: str | Path) -> Recording:
    """Load UFF/UNV dataset-58 function sets as channels.

    Prefer ``func_type == 1`` (time response). If none are found, use all
    dataset-58 sets that look like real-valued even-spaced series.
    """
    import pyuff

    path = Path(path)
    uff = pyuff.UFF(str(path))
    sets = uff.read_sets()
    if not isinstance(sets, list):
        sets = [sets]

    ds58 = [s for s in sets if int(s.get("type", -1)) == 58]
    if not ds58:
        raise ValueError(f"No dataset 58 found in UFF: {path}")

    time_sets = [s for s in ds58 if int(s.get("func_type", -1)) == 1]
    chosen = time_sets or ds58

    channels: list[Channel] = []
    sample_rate: float | None = None
    time: np.ndarray | None = None

    for i, s in enumerate(chosen):
        y = np.asarray(s.get("data"), dtype=np.float64).reshape(-1)
        if y.size == 0:
            continue
        # Drop imaginary part if complex
        if np.iscomplexobj(y):
            y = np.real(y)

        name = _channel_name(s, i)
        unit = str(s.get("ordinate_axis_units_lab") or "").strip()
        channels.append(Channel(name=name, data=y, unit=unit))

        fs_i, t_i = _time_base(s, y.size)
        if sample_rate is None:
            sample_rate = fs_i
            time = t_i
        elif abs(fs_i - sample_rate) > 1e-9 * max(sample_rate, fs_i):
            # Keep first set's time base; still load channel data (same n preferred)
            pass

    if not channels:
        raise ValueError(f"No usable dataset-58 channels in UFF: {path}")
    if sample_rate is None or sample_rate <= 0:
        raise ValueError(f"Could not determine sample rate from UFF: {path}")

    # Truncate / pad to common length
    n = min(ch.data.shape[0] for ch in channels)
    for ch in channels:
        if ch.data.shape[0] != n:
            ch.data = ch.data[:n].copy()
    if time is not None and time.shape[0] != n:
        time = time[:n].copy()

    return Recording(
        sample_rate=float(sample_rate),
        channels=channels,
        source=str(path.resolve()),
        time=time,
    )


def _channel_name(s: dict, index: int) -> str:
    id1 = str(s.get("id1") or "").strip()
    node = s.get("rsp_node")
    direction = s.get("rsp_dir")
    parts = []
    if id1:
        parts.append(id1[:40])
    if node is not None:
        parts.append(f"N{node}")
    if direction is not None:
        parts.append(f"D{direction}")
    if parts:
        return "_".join(str(p) for p in parts)
    return f"Ch{index + 1}"


def _time_base(s: dict, n: int) -> tuple[float, np.ndarray | None]:
    spacing = int(s.get("abscissa_spacing", 1) or 1)
    x = s.get("x")
    if x is not None:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size >= 2:
            dt = float(np.median(np.diff(x)))
            if dt > 0:
                return 1.0 / dt, x[:n] if x.size >= n else x

    if spacing == 1:
        abscissa_inc = float(s.get("abscissa_inc") or 0.0)
        abscissa_min = float(s.get("abscissa_min") or 0.0)
        if abscissa_inc > 0:
            t = abscissa_min + np.arange(n, dtype=np.float64) * abscissa_inc
            return 1.0 / abscissa_inc, t

    # Fallback
    return 1.0, None
