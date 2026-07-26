"""Spectrum measure helpers: dB display, band/overall RMS, peak pick."""

from __future__ import annotations

import numpy as np

_DB_FLOOR = 1e-12


def to_db(magnitude: np.ndarray | float) -> np.ndarray | float:
    """Convert linear magnitude to dB re 1 (``20·log10``), floored at 1e-12."""
    arr = np.asarray(magnitude, dtype=np.float64)
    out = 20.0 * np.log10(np.maximum(arr, _DB_FLOOR))
    if np.isscalar(magnitude) or (isinstance(magnitude, np.ndarray) and magnitude.ndim == 0):
        return float(out)
    return out


def level_db(rms_linear: float) -> float:
    """``20·log10`` of a non-negative linear RMS level (re 1)."""
    return float(to_db(max(float(rms_linear), 0.0)))


def to_rms_amplitudes(
    magnitude: np.ndarray,
    amplitude_scale: str = "peak",
) -> np.ndarray:
    """Convert spectrum bin magnitudes to RMS-equivalent amplitudes.

    - ``rms``: already RMS → unchanged
    - ``peak`` / ``lin``: DC unchanged; AC bins ÷ √2
    - ``ptp``: AC bins ÷ (2√2); DC ÷ 2

    ``lin`` is treated like ``peak`` for this conversion; it is not a
    strict engineering RMS scale (see Analysis amplitude-scale notes).
    """
    mag = np.asarray(magnitude, dtype=np.float64).reshape(-1)
    key = (amplitude_scale or "peak").strip().lower()
    out = mag.copy()
    if key == "rms":
        return out
    if key in ("ptp", "peak_to_peak", "peak-to-peak"):
        if out.size > 0:
            out[0] = out[0] / 2.0
        if out.size > 1:
            out[1:] = out[1:] / (2.0 * np.sqrt(2.0))
        return out
    # peak, lin, linear, abs_over_n, …
    if out.size > 1:
        out[1:] = out[1:] / np.sqrt(2.0)
    return out


def band_mask(
    frequency_hz: np.ndarray,
    f_lo: float | None = None,
    f_hi: float | None = None,
) -> np.ndarray:
    """Boolean mask for bins in ``[f_lo, f_hi]`` (open ends when None)."""
    f = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    mask = np.ones(f.shape, dtype=bool)
    if f_lo is not None and np.isfinite(f_lo):
        mask &= f >= float(f_lo)
    if f_hi is not None and np.isfinite(f_hi):
        mask &= f <= float(f_hi)
    return mask


def band_rms(
    frequency_hz: np.ndarray,
    magnitude: np.ndarray,
    *,
    f_lo: float | None = None,
    f_hi: float | None = None,
    amplitude_scale: str = "peak",
) -> float:
    """Overall or band RMS from discrete spectrum bins: ``√Σ A_rms²``.

    When both ``f_lo`` and ``f_hi`` are None, all bins are used (all-pass /
    overall). Empty band → 0.0.
    """
    freq = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    mag = np.asarray(magnitude, dtype=np.float64).reshape(-1)
    if freq.size == 0 or mag.size == 0 or freq.size != mag.size:
        return 0.0
    rms_amp = to_rms_amplitudes(mag, amplitude_scale)
    mask = band_mask(freq, f_lo, f_hi)
    if not np.any(mask):
        return 0.0
    return float(np.sqrt(np.sum(np.square(rms_amp[mask]))))


def find_spectrum_peaks(
    frequency_hz: np.ndarray,
    magnitude: np.ndarray,
    *,
    n_peaks: int = 5,
    min_height: float | None = None,
) -> list[tuple[float, float, int]]:
    """Find local maxima; return up to ``n_peaks`` as ``(f, mag, index)``.

    Peaks are ranked by magnitude (highest first), then sorted by frequency.
    """
    freq = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    mag = np.asarray(magnitude, dtype=np.float64).reshape(-1)
    n = int(max(0, n_peaks))
    if n == 0 or freq.size < 3 or mag.size != freq.size:
        return []
    left = mag[1:-1] > mag[:-2]
    right = mag[1:-1] >= mag[2:]
    candidates = np.where(left & right)[0] + 1
    if candidates.size == 0:
        return []
    if min_height is not None and np.isfinite(min_height):
        candidates = candidates[mag[candidates] >= float(min_height)]
        if candidates.size == 0:
            return []
    order = np.argsort(mag[candidates])[::-1]
    chosen = candidates[order][:n]
    chosen = np.sort(chosen)
    return [(float(freq[i]), float(mag[i]), int(i)) for i in chosen]


def series_amplitude_scale(attrs: dict | None) -> str:
    """Read ``amplitude_scale`` from spectrum series ``meta.attrs``."""
    if not attrs:
        return "peak"
    params = attrs.get("spectrum_params")
    if isinstance(params, dict):
        return str(params.get("amplitude_scale") or "peak")
    return "peak"
