"""FFT / spectrum helpers (M3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

WINDOW_NAMES = ("rectangular", "hann", "hamming", "flattop")
AMPLITUDE_SCALES = ("lin", "peak", "rms")


@dataclass
class SpectrumParams:
    """User-facing FFT settings (stored on result provenance)."""

    window: str = "hann"
    nfft: int | None = None
    averaging: bool = False
    overlap: float = 0.5
    segment_length: int | None = None
    amplitude_scale: str = "lin"

    def to_dict(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "nfft": self.nfft,
            "averaging": self.averaging,
            "overlap": self.overlap,
            "segment_length": self.segment_length,
            "amplitude_scale": self.amplitude_scale,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SpectrumParams:
        if not data:
            return cls()
        nfft = data.get("nfft")
        seg = data.get("segment_length")
        return cls(
            window=str(data.get("window") or "hann"),
            nfft=None if nfft is None else int(nfft),
            averaging=bool(data.get("averaging", False)),
            overlap=float(data.get("overlap") if data.get("overlap") is not None else 0.5),
            segment_length=None if seg is None else int(seg),
            amplitude_scale=str(data.get("amplitude_scale") or "lin"),
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


def _apply_amplitude_scale(
    complex_spec: np.ndarray, *, scale: str, window_sum: float
) -> np.ndarray:
    """Scale complex rFFT result; returns same shape complex array."""
    denom = window_sum if window_sum > 0 else 1.0
    scaled = complex_spec / denom
    key = (scale or "lin").strip().lower()
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
    raise ValueError(f"Unknown amplitude_scale: {scale!r} (expected one of {AMPLITUDE_SCALES})")


def _segment_starts(n: int, seg_len: int, overlap: float) -> list[int]:
    if seg_len <= 0 or seg_len > n:
        return []
    step = max(1, int(round(seg_len * (1.0 - float(np.clip(overlap, 0.0, 0.95))))) )
    starts = list(range(0, n - seg_len + 1, step))
    if not starts:
        starts = [0]
    return starts


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
        seg_len = int(p.segment_length) if p.segment_length else min(y.size, 1024)
        seg_len = max(2, min(seg_len, y.size))
        nfft = int(p.nfft) if p.nfft else seg_len
        nfft = max(seg_len, nfft)
        starts = _segment_starts(y.size, seg_len, p.overlap)
        if not starts:
            starts = [0]
            seg_len = y.size
            nfft = max(seg_len, int(p.nfft) if p.nfft else seg_len)
        window = make_window(p.window, seg_len)
        wsum = float(np.sum(window))
        acc = None
        for start in starts:
            chunk = y[start : start + seg_len] * window
            if nfft > seg_len:
                padded = np.zeros(nfft, dtype=np.float64)
                padded[:seg_len] = chunk
                chunk = padded
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
    window = make_window(p.window, y.size)
    wsum = float(np.sum(window))
    yw = y * window
    nfft = int(p.nfft) if p.nfft else y.size
    nfft = max(y.size, nfft)
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
