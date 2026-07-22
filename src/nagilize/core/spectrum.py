"""FFT / spectrum helpers (M3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

WINDOW_NAMES = ("rectangular", "hann", "hamming", "flattop")
AMPLITUDE_SCALES = ("lin", "peak", "rms", "ptp")


@dataclass
class SpectrumParams:
    """User-facing FFT settings (stored on result provenance)."""

    window: str = "hann"
    nfft: int | None = None
    averaging: bool = False
    overlap: float = 0.5
    n_averages: int | None = None
    segment_length: int | None = None  # legacy; averaging uses nfft as segment length
    amplitude_scale: str = "peak"
    stft_step_s: float | None = None  # fixed Δt stepping; None → use overlap
    stft_step_mode: str = "overlap"  # overlap | fixed_dt | equal_angle | equal_rpm
    delta_theta_deg: float | None = None
    delta_rpm: float | None = None
    tacho_kind: str | None = None  # pulse | rpm
    pulses_per_rev: int | None = None
    tacho_series_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "nfft": self.nfft,
            "averaging": self.averaging,
            "overlap": self.overlap,
            "n_averages": self.n_averages,
            "segment_length": self.segment_length,
            "amplitude_scale": self.amplitude_scale,
            "stft_step_s": self.stft_step_s,
            "stft_step_mode": self.stft_step_mode,
            "delta_theta_deg": self.delta_theta_deg,
            "delta_rpm": self.delta_rpm,
            "tacho_kind": self.tacho_kind,
            "pulses_per_rev": self.pulses_per_rev,
            "tacho_series_id": self.tacho_series_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SpectrumParams:
        if not data:
            return cls()
        nfft = data.get("nfft")
        seg = data.get("segment_length")
        n_avg = data.get("n_averages")
        step_s = data.get("stft_step_s")
        dth = data.get("delta_theta_deg")
        drpm = data.get("delta_rpm")
        ppr = data.get("pulses_per_rev")
        return cls(
            window=str(data.get("window") or "hann"),
            nfft=None if nfft is None else int(nfft),
            averaging=bool(data.get("averaging", False)),
            overlap=float(data.get("overlap") if data.get("overlap") is not None else 0.5),
            n_averages=None if n_avg is None else int(n_avg),
            segment_length=None if seg is None else int(seg),
            amplitude_scale=str(data.get("amplitude_scale") or "peak"),
            stft_step_s=None if step_s is None else float(step_s),
            stft_step_mode=str(data.get("stft_step_mode") or "overlap"),
            delta_theta_deg=None if dth is None else float(dth),
            delta_rpm=None if drpm is None else float(drpm),
            tacho_kind=None if data.get("tacho_kind") is None else str(data.get("tacho_kind")),
            pulses_per_rev=None if ppr is None else int(ppr),
            tacho_series_id=(
                None
                if data.get("tacho_series_id") is None
                else str(data.get("tacho_series_id"))
            ),
        )


@dataclass
class Spectrum:
    """One-sided spectrum (complex mean of segments when averaging)."""

    frequency_hz: np.ndarray
    real: np.ndarray
    imag: np.ndarray
    magnitude: np.ndarray
    phase_rad: np.ndarray
    params: SpectrumParams = field(default_factory=SpectrumParams)
    n_segments: int = 1
    delta_f_hz: float = 0.0

    @property
    def phase_deg(self) -> np.ndarray:
        return np.degrees(self.phase_rad)


@dataclass
class StftResult:
    """Short-time FFT: magnitude/phase vs frequency and time."""

    frequency_hz: np.ndarray
    time_s: np.ndarray
    real: np.ndarray
    imag: np.ndarray
    magnitude: np.ndarray
    phase_rad: np.ndarray
    params: SpectrumParams = field(default_factory=SpectrumParams)
    n_frames: int = 0
    delta_f_hz: float = 0.0
    hop_samples: int = 1
    frame_len: int = 0
    angle_deg: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    rpm: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))

    @property
    def phase_deg(self) -> np.ndarray:
        return np.degrees(self.phase_rad)


def make_window(name: str, n: int) -> np.ndarray:
    """Return a length-``n`` analysis window."""
    key = (name or "hann").strip().lower()
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    if key in ("rect", "rectangular", "boxcar", "none"):
        return np.ones(n, dtype=np.float64)
    if key == "hann":
        return np.hanning(n).astype(np.float64, copy=False)
    if key == "hamming":
        return np.hamming(n).astype(np.float64, copy=False)
    if key in ("flattop", "flat_top", "flat-top"):
        # SciPy flattop coefficients (symmetric).
        a = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]
        k = np.arange(n, dtype=np.float64)
        w = a[0]
        for i, ai in enumerate(a[1:], start=1):
            w = w + ((-1) ** i) * ai * np.cos(2.0 * np.pi * i * k / max(n - 1, 1))
        return w.astype(np.float64, copy=False)
    raise ValueError(f"Unknown window: {name!r} (expected one of {WINDOW_NAMES})")


def frequency_resolution_hz(sample_rate: float, nfft: int) -> float:
    if sample_rate <= 0 or nfft <= 0:
        return 0.0
    return float(sample_rate) / float(nfft)


def nfft_from_delta_f(sample_rate: float, delta_f_hz: float) -> int:
    """Convert target frequency resolution to an FFT length (samples)."""
    if sample_rate <= 0 or delta_f_hz <= 0:
        return 2
    return max(2, int(round(float(sample_rate) / float(delta_f_hz))))


def resolve_nfft(
    n_samples: int,
    sample_rate: float,
    *,
    delta_f_hz: float | None = None,
    nfft: int | None = None,
    cap_to_signal_length: bool = False,
) -> int:
    """Resolve effective NFFT from Δf, explicit length, or auto rules."""
    n = max(int(n_samples), 1)
    requested: int | None = None
    if delta_f_hz is not None and delta_f_hz > 0:
        requested = nfft_from_delta_f(sample_rate, delta_f_hz)
    elif nfft is not None and int(nfft) > 0:
        requested = int(nfft)

    if cap_to_signal_length:
        if requested is None:
            return max(2, min(n, 1024))
        return max(2, min(requested, n))
    if requested is None:
        return max(n, 2)
    return max(2, requested)


def _apply_amplitude_scale(
    complex_spec: np.ndarray, *, scale: str, window_sum: float
) -> np.ndarray:
    """Scale complex rFFT result; returns same shape complex array."""
    denom = window_sum if window_sum > 0 else 1.0
    scaled = complex_spec / denom
    key = (scale or "peak").strip().lower()
    if key in ("lin", "linear", "abs_over_n"):
        return scaled
    # Single-sided peak (double non-DC / non-Nyquist bins).
    peak = scaled.copy()
    if peak.size > 2:
        peak[1:-1] *= 2.0
    elif peak.size == 2:
        peak[1] *= 2.0
    if key == "peak":
        return peak
    if key == "rms":
        return peak / np.sqrt(2.0)
    if key in ("ptp", "peak_to_peak", "peak-to-peak"):
        return peak * 2.0
    raise ValueError(f"Unknown amplitude_scale: {scale!r} (expected one of {AMPLITUDE_SCALES})")


def _segment_starts(n: int, seg_len: int, overlap: float) -> list[int]:
    if seg_len <= 0 or seg_len > n:
        return []
    step = max(1, int(round(seg_len * (1.0 - float(np.clip(overlap, 0.0, 0.95))))) )
    starts = list(range(0, n - seg_len + 1, step))
    if not starts:
        starts = [0]
    return starts


def max_averages(n: int, seg_len: int, overlap: float) -> int:
    """How many Welch segments fit for the given length / overlap."""
    return len(_segment_starts(n, seg_len, overlap))


def stft_hop_samples(
    *,
    frame_len: int,
    sample_rate: float,
    overlap: float | None = None,
    step_s: float | None = None,
) -> int:
    """Hop size in samples for STFT (from overlap % or fixed time step)."""
    frame_len = max(1, int(frame_len))
    if step_s is not None and step_s > 0 and sample_rate > 0:
        return max(1, int(round(float(step_s) * float(sample_rate))))
    ov = float(np.clip(overlap if overlap is not None else 0.5, 0.0, 0.95))
    return max(1, int(round(frame_len * (1.0 - ov))))


def stft_frame_count(n: int, frame_len: int, hop: int) -> int:
    """How many STFT frames fit for signal length, frame length, and hop."""
    if frame_len <= 0 or frame_len > n:
        return 0
    hop = max(1, int(hop))
    starts = list(range(0, n - frame_len + 1, hop))
    if not starts:
        return 1 if n >= frame_len else 0
    return len(starts)


def stft_overlap_equiv(frame_len: int, hop: int) -> float:
    """Equivalent overlap fraction for a given frame length and hop."""
    if frame_len <= 0:
        return 0.0
    return float(max(0.0, min(0.95, 1.0 - float(hop) / float(frame_len))))


def compute_spectrum(
    data: np.ndarray,
    sample_rate: float,
    params: SpectrumParams | None = None,
) -> Spectrum:
    """Compute one-sided spectrum with optional window, zero-pad, and Welch average.

    Without averaging: one rFFT of the (windowed, optionally zero-padded) signal.
    With averaging: coherent mean of complex rFFTs over overlapped segments.
    """
    y = np.asarray(data, dtype=np.float64).reshape(-1)
    p = params or SpectrumParams()
    empty = np.array([], dtype=np.float64)
    if y.size < 2:
        return Spectrum(empty, empty, empty, empty, empty, params=p, n_segments=0, delta_f_hz=0.0)
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")

    if p.averaging:
        if p.nfft:
            seg_len = max(2, min(int(p.nfft), y.size))
        elif p.segment_length:
            seg_len = max(2, min(int(p.segment_length), y.size))
        else:
            seg_len = max(2, min(y.size, 1024))
        nfft = seg_len
        starts = _segment_starts(y.size, seg_len, p.overlap)
        if not starts:
            starts = [0]
            seg_len = y.size
            nfft = seg_len
        if p.n_averages is not None and int(p.n_averages) > 0:
            starts = starts[: max(1, min(int(p.n_averages), len(starts)))]
        window = make_window(p.window, seg_len)
        wsum = float(np.sum(window))
        acc = None
        for start in starts:
            chunk = y[start : start + seg_len] * window
            spec = np.fft.rfft(chunk)
            if acc is None:
                acc = spec.astype(np.complex128, copy=True)
            else:
                acc += spec
        assert acc is not None
        n_seg = len(starts)
        mean_c = acc / float(n_seg)
        mean_c = _apply_amplitude_scale(mean_c, scale=p.amplitude_scale, window_sum=wsum)
        freq = np.fft.rfftfreq(nfft, d=1.0 / float(sample_rate))
        return Spectrum(
            frequency_hz=freq.astype(np.float64, copy=False),
            real=np.real(mean_c).astype(np.float64, copy=False),
            imag=np.imag(mean_c).astype(np.float64, copy=False),
            magnitude=np.abs(mean_c).astype(np.float64, copy=False),
            phase_rad=np.angle(mean_c).astype(np.float64, copy=False),
            params=p,
            n_segments=n_seg,
            delta_f_hz=frequency_resolution_hz(sample_rate, nfft),
        )

    # Single-shot
    if p.nfft:
        nfft = max(2, int(p.nfft))
        if nfft < y.size:
            y = y[:nfft]
    else:
        nfft = y.size
    window = make_window(p.window, y.size)
    wsum = float(np.sum(window))
    yw = y * window
    if nfft > y.size:
        padded = np.zeros(nfft, dtype=np.float64)
        padded[: y.size] = yw
        yw = padded
    spec = np.fft.rfft(yw)
    scaled = _apply_amplitude_scale(spec, scale=p.amplitude_scale, window_sum=wsum)
    freq = np.fft.rfftfreq(nfft, d=1.0 / float(sample_rate))
    return Spectrum(
        frequency_hz=freq.astype(np.float64, copy=False),
        real=np.real(scaled).astype(np.float64, copy=False),
        imag=np.imag(scaled).astype(np.float64, copy=False),
        magnitude=np.abs(scaled).astype(np.float64, copy=False),
        phase_rad=np.angle(scaled).astype(np.float64, copy=False),
        params=p,
        n_segments=1,
        delta_f_hz=frequency_resolution_hz(sample_rate, nfft),
    )


def _stft_frame_starts(n: int, frame_len: int, hop: int) -> list[int]:
    if frame_len <= 0 or frame_len > n:
        return []
    hop = max(1, int(hop))
    return list(range(0, n - frame_len + 1, hop))


def angle_from_pulse(
    data: np.ndarray,
    sample_rate: float,
    *,
    pulses_per_rev: int = 1,
    threshold: float | None = None,
) -> np.ndarray:
    """Build cumulative shaft angle (deg) from a tacho pulse train.

    Rising edges advance angle by ``360 / pulses_per_rev`` degrees.
    Values between edges are linearly interpolated in sample index.
    """
    y = np.asarray(data, dtype=np.float64).reshape(-1)
    n = int(y.size)
    if n == 0:
        return np.array([], dtype=np.float64)
    ppr = max(1, int(pulses_per_rev))
    step_deg = 360.0 / float(ppr)
    if threshold is None:
        y_min = float(np.min(y))
        y_max = float(np.max(y))
        threshold = 0.5 * (y_min + y_max)
    high = y >= float(threshold)
    edges = np.flatnonzero(high[1:] & ~high[:-1]) + 1
    theta = np.zeros(n, dtype=np.float64)
    if edges.size == 0:
        return theta
    edge_angles = np.arange(edges.size, dtype=np.float64) * step_deg
    # Interpolate from sample 0 (angle 0 before first edge) through edges to end.
    xp = np.concatenate(([0], edges.astype(np.float64), [n - 1]))
    fp = np.concatenate(([0.0], edge_angles, [edge_angles[-1]]))
    # Deduplicate xp for interp (n-1 may equal last edge).
    keep = np.ones(xp.size, dtype=bool)
    keep[1:] = np.diff(xp) > 0
    xp = xp[keep]
    fp = fp[keep]
    theta = np.interp(np.arange(n, dtype=np.float64), xp, fp)
    return theta.astype(np.float64, copy=False)


def angle_from_rpm(rpm: np.ndarray, sample_rate: float) -> np.ndarray:
    """Integrate RPM vs time into cumulative shaft angle (deg).

    Uses ``ω_deg/s = rpm * 6`` (since 360 deg/rev * rpm/60).
    """
    r = np.asarray(rpm, dtype=np.float64).reshape(-1)
    if r.size == 0:
        return np.array([], dtype=np.float64)
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    omega_deg = r * 6.0
    dt = 1.0 / float(sample_rate)
    # Cumulative trapezoid from t=0.
    theta = np.empty(r.size, dtype=np.float64)
    theta[0] = 0.0
    if r.size > 1:
        theta[1:] = np.cumsum(0.5 * (omega_deg[:-1] + omega_deg[1:]) * dt)
    return theta


def rpm_from_pulse(
    data: np.ndarray,
    sample_rate: float,
    *,
    pulses_per_rev: int = 1,
    threshold: float | None = None,
) -> np.ndarray:
    """Estimate instantaneous RPM from a tacho pulse train.

    Rising-edge intervals give RPM = ``60 * pulses_per_rev / Δt``.
    Values are held between edges and interpolated onto every sample.
    """
    y = np.asarray(data, dtype=np.float64).reshape(-1)
    n = int(y.size)
    if n == 0:
        return np.array([], dtype=np.float64)
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    ppr = max(1, int(pulses_per_rev))
    if threshold is None:
        y_min = float(np.min(y))
        y_max = float(np.max(y))
        threshold = 0.5 * (y_min + y_max)
    high = y >= float(threshold)
    edges = np.flatnonzero(high[1:] & ~high[:-1]) + 1
    if edges.size < 2:
        return np.zeros(n, dtype=np.float64)
    dt = np.diff(edges.astype(np.float64)) / float(sample_rate)
    dt = np.maximum(dt, 1.0 / float(sample_rate))
    rpm_edges = 60.0 * float(ppr) / dt
    # Assign RPM of interval [edge_i, edge_{i+1}) to the midpoint, then interp.
    mid = 0.5 * (edges[:-1].astype(np.float64) + edges[1:].astype(np.float64))
    xp = np.concatenate(([0.0], mid, [float(n - 1)]))
    fp = np.concatenate(([rpm_edges[0]], rpm_edges, [rpm_edges[-1]]))
    keep = np.ones(xp.size, dtype=bool)
    keep[1:] = np.diff(xp) > 0
    return np.interp(np.arange(n, dtype=np.float64), xp[keep], fp[keep]).astype(
        np.float64, copy=False
    )


def rpm_from_angle(theta_deg: np.ndarray, sample_rate: float) -> np.ndarray:
    """Convert cumulative shaft angle (deg) to instantaneous RPM via dθ/dt / 6."""
    th = np.asarray(theta_deg, dtype=np.float64).reshape(-1)
    if th.size == 0:
        return np.array([], dtype=np.float64)
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if th.size == 1:
        return np.zeros(1, dtype=np.float64)
    dth = np.diff(th) * float(sample_rate)  # deg/s at sample midpoints
    omega = np.empty(th.size, dtype=np.float64)
    omega[0] = dth[0]
    omega[-1] = dth[-1]
    if th.size > 2:
        omega[1:-1] = 0.5 * (dth[:-1] + dth[1:])
    return (omega / 6.0).astype(np.float64, copy=False)


def _equal_axis_frame_starts(
    axis: np.ndarray,
    *,
    frame_len: int,
    delta: float,
    sample_rate: float,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Place STFT frames at equal spacing along a monotone axis (θ or RPM)."""
    ax = np.asarray(axis, dtype=np.float64).reshape(-1)
    n = int(ax.size)
    fl = max(2, int(frame_len))
    step = float(delta)
    empty = np.array([], dtype=np.float64)
    if n < fl or step <= 0 or sample_rate <= 0:
        return [], empty, empty

    half = fl // 2
    i0 = half
    i1 = n - (fl - half)
    if i1 <= i0:
        return [], empty, empty

    lo = float(ax[i0])
    hi = float(ax[i1 - 1])
    if hi <= lo:
        return [], empty, empty

    centers = np.arange(lo, hi + 0.5 * step, step, dtype=np.float64)
    idx = np.searchsorted(ax, centers, side="left")
    idx = np.clip(idx, i0, i1 - 1)
    for k, target in enumerate(centers):
        j = int(idx[k])
        if j > i0 and abs(ax[j - 1] - target) < abs(ax[j] - target):
            idx[k] = j - 1

    starts: list[int] = []
    times: list[float] = []
    values: list[float] = []
    last_start = -1
    for j in idx.astype(int):
        start = int(j - half)
        if start < 0 or start + fl > n:
            continue
        if start == last_start:
            continue
        last_start = start
        center = start + frame_len / 2.0
        starts.append(start)
        times.append(center / float(sample_rate))
        values.append(float(ax[j]))
    return (
        starts,
        np.asarray(times, dtype=np.float64),
        np.asarray(values, dtype=np.float64),
    )


def equal_angle_frame_starts(
    theta_deg: np.ndarray,
    *,
    frame_len: int,
    delta_theta_deg: float,
    sample_rate: float,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Place STFT frames at equal angular spacing.

    Returns ``(starts, time_s_centers, angle_deg_centers)``.
    Frame length is fixed in samples; only centers follow Δθ.
    """
    return _equal_axis_frame_starts(
        theta_deg,
        frame_len=frame_len,
        delta=delta_theta_deg,
        sample_rate=sample_rate,
    )


def equal_rpm_frame_starts(
    rpm: np.ndarray,
    *,
    frame_len: int,
    delta_rpm: float,
    sample_rate: float,
    theta_deg: np.ndarray | None = None,
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    """Place STFT frames at equal RPM spacing.

    Returns ``(starts, time_s_centers, rpm_centers, angle_deg_centers)``.
    ``angle_deg_centers`` is empty unless ``theta_deg`` is provided.
    """
    starts, times, rpm_out = _equal_axis_frame_starts(
        rpm,
        frame_len=frame_len,
        delta=delta_rpm,
        sample_rate=sample_rate,
    )
    empty = np.array([], dtype=np.float64)
    if not starts or theta_deg is None:
        return starts, times, rpm_out, empty
    th = np.asarray(theta_deg, dtype=np.float64).reshape(-1)
    if th.size != np.asarray(rpm).size:
        return starts, times, rpm_out, empty
    half = max(2, int(frame_len)) // 2
    angles = np.asarray(
        [float(th[start + half]) for start in starts], dtype=np.float64
    )
    return starts, times, rpm_out, angles


def compute_stft(
    data: np.ndarray,
    sample_rate: float,
    params: SpectrumParams | None = None,
    *,
    delta_f_hz: float | None = None,
    theta_deg: np.ndarray | None = None,
    delta_theta_deg: float | None = None,
    rpm: np.ndarray | None = None,
    delta_rpm: float | None = None,
) -> StftResult:
    """Compute a one-sided STFT (complex spectrum per overlapping frame).

    Equal-angle: pass ``theta_deg`` + ``delta_theta_deg``.
    Equal-RPM: pass ``rpm`` + ``delta_rpm`` (optional ``theta_deg`` for angle axis).
    Frame length still comes from Δf / nfft in samples.
    """
    y = np.asarray(data, dtype=np.float64).reshape(-1)
    p = params or SpectrumParams()
    empty = np.array([], dtype=np.float64)
    empty_2d = empty.reshape(0, 0)
    if y.size < 2:
        return StftResult(
            empty,
            empty,
            empty_2d,
            empty_2d,
            empty_2d,
            empty_2d,
            params=p,
            n_frames=0,
            delta_f_hz=0.0,
            hop_samples=1,
            frame_len=0,
            angle_deg=empty,
            rpm=empty,
        )
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")

    frame_len = resolve_nfft(
        y.size,
        sample_rate,
        delta_f_hz=delta_f_hz,
        nfft=p.nfft,
        cap_to_signal_length=True,
    )

    mode = (p.stft_step_mode or "").strip()
    dth = delta_theta_deg if delta_theta_deg is not None else p.delta_theta_deg
    drpm = delta_rpm if delta_rpm is not None else p.delta_rpm

    if mode == "equal_rpm":
        use_rpm = rpm is not None and drpm is not None and drpm > 0
        use_angle = False
    elif mode == "equal_angle":
        use_angle = theta_deg is not None and dth is not None and dth > 0
        use_rpm = False
    else:
        use_rpm = False
        use_angle = False

    angle_out = empty
    rpm_out = empty
    hop = 1

    if use_rpm and rpm is not None and drpm is not None and drpm > 0:
        r = np.asarray(rpm, dtype=np.float64).reshape(-1)
        if r.size != y.size:
            raise ValueError(
                f"rpm length ({r.size}) must match signal length ({y.size})"
            )
        th = None if theta_deg is None else np.asarray(theta_deg, dtype=np.float64)
        starts, times, rpm_out, angle_out = equal_rpm_frame_starts(
            r,
            frame_len=frame_len,
            delta_rpm=float(drpm),
            sample_rate=sample_rate,
            theta_deg=th,
        )
        if len(starts) >= 2:
            hop = max(1, int(round(np.median(np.diff(starts)))))
        elif starts:
            hop = 1
    elif use_angle and theta_deg is not None and dth is not None and dth > 0:
        th = np.asarray(theta_deg, dtype=np.float64).reshape(-1)
        if th.size != y.size:
            raise ValueError(
                f"theta_deg length ({th.size}) must match signal length ({y.size})"
            )
        starts, times, angle_out = equal_angle_frame_starts(
            th, frame_len=frame_len, delta_theta_deg=float(dth), sample_rate=sample_rate
        )
        # Fill per-frame RPM from θ when possible.
        rpm_series = rpm_from_angle(th, sample_rate)
        if starts:
            half = frame_len // 2
            rpm_out = np.asarray(
                [float(rpm_series[start + half]) for start in starts],
                dtype=np.float64,
            )
        if len(starts) >= 2:
            hop = max(1, int(round(np.median(np.diff(starts)))))
        elif starts:
            hop = 1
    else:
        hop = stft_hop_samples(
            frame_len=frame_len,
            sample_rate=sample_rate,
            overlap=None if p.stft_step_s is not None else p.overlap,
            step_s=p.stft_step_s,
        )
        starts = _stft_frame_starts(y.size, frame_len, hop)
        times = (
            (np.asarray(starts, dtype=np.float64) + frame_len / 2.0) / float(sample_rate)
            if starts
            else empty
        )

    if not starts:
        return StftResult(
            empty,
            empty,
            empty_2d,
            empty_2d,
            empty_2d,
            empty_2d,
            params=p,
            n_frames=0,
            delta_f_hz=frequency_resolution_hz(sample_rate, frame_len),
            hop_samples=hop,
            frame_len=frame_len,
            angle_deg=empty,
            rpm=empty,
        )

    window = make_window(p.window, frame_len)
    wsum = float(np.sum(window))
    specs: list[np.ndarray] = []
    for start in starts:
        chunk = y[start : start + frame_len] * window
        spec = np.fft.rfft(chunk)
        specs.append(
            _apply_amplitude_scale(spec, scale=p.amplitude_scale, window_sum=wsum)
        )
    stacked = np.column_stack(specs).astype(np.complex128, copy=False)
    freq = np.fft.rfftfreq(frame_len, d=1.0 / float(sample_rate))
    return StftResult(
        frequency_hz=freq.astype(np.float64, copy=False),
        time_s=np.asarray(times, dtype=np.float64),
        real=np.real(stacked).astype(np.float64, copy=False),
        imag=np.imag(stacked).astype(np.float64, copy=False),
        magnitude=np.abs(stacked).astype(np.float64, copy=False),
        phase_rad=np.angle(stacked).astype(np.float64, copy=False),
        params=p,
        n_frames=len(starts),
        delta_f_hz=frequency_resolution_hz(sample_rate, frame_len),
        hop_samples=hop,
        frame_len=frame_len,
        angle_deg=np.asarray(angle_out, dtype=np.float64),
        rpm=np.asarray(rpm_out, dtype=np.float64),
    )
