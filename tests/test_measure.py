"""Unit tests for spectrum measure helpers (dB, band RMS, peaks)."""

from __future__ import annotations

import numpy as np

from ginkyo.core.measure import (
    band_rms,
    find_spectrum_peaks,
    level_db,
    series_amplitude_scale,
    to_db,
    to_rms_amplitudes,
)
from ginkyo.core.spectrum import SpectrumParams, compute_spectrum


def test_to_db_floor_and_monotonic() -> None:
    assert to_db(1.0) == 0.0
    assert to_db(10.0) == 20.0
    assert to_db(0.0) < -200.0
    xs = np.array([0.1, 1.0, 10.0])
    ys = to_db(xs)
    assert np.all(np.diff(ys) > 0)


def test_level_db_matches_to_db() -> None:
    assert abs(level_db(1.0) - 0.0) < 1e-12
    assert abs(level_db(0.1) - (-20.0)) < 1e-9


def test_to_rms_amplitudes_peak_and_rms() -> None:
    mag = np.array([2.0, 2.0, 4.0])  # DC, AC, AC
    peak_rms = to_rms_amplitudes(mag, "peak")
    assert abs(peak_rms[0] - 2.0) < 1e-12
    assert abs(peak_rms[1] - 2.0 / np.sqrt(2.0)) < 1e-12
    assert abs(peak_rms[2] - 4.0 / np.sqrt(2.0)) < 1e-12
    same = to_rms_amplitudes(mag, "rms")
    np.testing.assert_allclose(same, mag)


def test_band_rms_synthetic_bins() -> None:
    freq = np.array([0.0, 10.0, 20.0, 30.0])
    # peak scale: AC bins 2 and 2 → rms each √2 → overall √(0 + 2 + 2) wait
    # DC=0, A_peak=√2 at 10 and 20 → A_rms=1 each → overall √2
    mag_peak = np.array([0.0, np.sqrt(2.0), np.sqrt(2.0), 0.0])
    overall = band_rms(freq, mag_peak, amplitude_scale="peak")
    assert abs(overall - np.sqrt(2.0)) < 1e-9
    band = band_rms(freq, mag_peak, f_lo=15.0, f_hi=25.0, amplitude_scale="peak")
    assert abs(band - 1.0) < 1e-9
    empty = band_rms(freq, mag_peak, f_lo=100.0, f_hi=200.0, amplitude_scale="peak")
    assert empty == 0.0


def test_band_rms_sine_peak_scale() -> None:
    fs = 1024.0
    n = 1024
    t = np.arange(n) / fs
    amp = 2.0
    y = amp * np.sin(2.0 * np.pi * 64.0 * t)  # bin-centered tone
    spec = compute_spectrum(
        y,
        fs,
        SpectrumParams(
            window="rectangular",
            nfft=n,
            averaging=False,
            amplitude_scale="peak",
        ),
    )
    peak_bin = int(np.argmax(spec.magnitude))
    assert abs(spec.frequency_hz[peak_bin] - 64.0) < 1e-6
    assert abs(spec.magnitude[peak_bin] - amp) < 0.05
    overall = band_rms(
        spec.frequency_hz,
        spec.magnitude,
        amplitude_scale="peak",
    )
    expected = amp / np.sqrt(2.0)
    assert abs(overall - expected) / expected < 0.05
    # Band excluding the tone → near zero
    far = band_rms(
        spec.frequency_hz,
        spec.magnitude,
        f_lo=200.0,
        f_hi=300.0,
        amplitude_scale="peak",
    )
    assert far < 0.05 * expected


def test_band_rms_sine_rms_scale() -> None:
    fs = 1024.0
    n = 1024
    t = np.arange(n) / fs
    amp = 2.0
    y = amp * np.sin(2.0 * np.pi * 64.0 * t)
    spec = compute_spectrum(
        y,
        fs,
        SpectrumParams(
            window="rectangular",
            nfft=n,
            averaging=False,
            amplitude_scale="rms",
        ),
    )
    overall = band_rms(
        spec.frequency_hz,
        spec.magnitude,
        amplitude_scale="rms",
    )
    expected = amp / np.sqrt(2.0)
    assert abs(overall - expected) / expected < 0.05


def test_find_spectrum_peaks_two_tones() -> None:
    fs = 2048.0
    n = 2048
    t = np.arange(n) / fs
    y = (
        1.0 * np.sin(2.0 * np.pi * 80.0 * t)
        + 0.5 * np.sin(2.0 * np.pi * 200.0 * t)
    )
    spec = compute_spectrum(
        y,
        fs,
        SpectrumParams(
            window="rectangular",
            nfft=n,
            averaging=False,
            amplitude_scale="peak",
        ),
    )
    peaks = find_spectrum_peaks(spec.frequency_hz, spec.magnitude, n_peaks=2)
    assert len(peaks) == 2
    freqs = [p[0] for p in peaks]
    assert abs(freqs[0] - 80.0) < 2.0
    assert abs(freqs[1] - 200.0) < 2.0
    assert peaks[0][1] > peaks[1][1]


def test_series_amplitude_scale() -> None:
    assert series_amplitude_scale(None) == "peak"
    assert series_amplitude_scale({}) == "peak"
    assert (
        series_amplitude_scale({"spectrum_params": {"amplitude_scale": "rms"}})
        == "rms"
    )
