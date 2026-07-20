"""Unit tests for spectrum FFT helpers."""

from __future__ import annotations

import numpy as np

from nagilize.core.spectrum import SpectrumParams, compute_spectrum, frequency_resolution_hz, make_window


def test_make_window_lengths() -> None:
    for name in ("rectangular", "hann", "hamming", "flattop"):
        w = make_window(name, 128)
        assert w.shape == (128,)
        assert np.all(np.isfinite(w))


def test_single_tone_peak_near_bin() -> None:
    fs = 1000.0
    n = 1000
    t = np.arange(n) / fs
    y = np.sin(2.0 * np.pi * 50.0 * t)
    spec = compute_spectrum(y, fs, SpectrumParams(window="rectangular", amplitude_scale="peak"))
    assert abs(spec.delta_f_hz - 1.0) < 1e-9
    idx = int(np.argmax(spec.magnitude))
    assert abs(spec.frequency_hz[idx] - 50.0) < 1.5
    assert abs(spec.magnitude[idx] - 1.0) < 0.05


def test_averaging_runs() -> None:
    fs = 1000.0
    y = np.random.default_rng(0).normal(size=4096)
    spec = compute_spectrum(
        y,
        fs,
        SpectrumParams(
            window="hann",
            averaging=True,
            overlap=0.5,
            segment_length=512,
            amplitude_scale="lin",
        ),
    )
    assert spec.n_segments >= 2
    assert spec.frequency_hz.size == spec.magnitude.size
    assert abs(spec.delta_f_hz - frequency_resolution_hz(fs, 512)) < 1e-12


def test_project_spectrum_roundtrip(tmp_path) -> None:
    from nagilize.core.dummy import make_sine_with_noise
    from nagilize.core.project import Project

    project = Project()
    project.add_recording(make_sine_with_noise())
    series = project.get(project.series_order[0])
    assert series is not None
    spec = compute_spectrum(series.data, series.sample_rate, SpectrumParams(window="hann"))
    sid = project.add_spectrum_result(
        name="fft",
        frequency_hz=spec.frequency_hz,
        magnitude=spec.magnitude,
        phase_rad=spec.phase_rad,
        sample_rate=series.sample_rate,
        parent=series,
        params_dict=spec.params.to_dict(),
    )
    path = tmp_path / "t.nagproj"
    project.save(path)
    loaded = Project.open(path)
    result = loaded.get(sid)
    assert result is not None
    result.ensure_loaded()
    assert result.is_spectrum()
    assert result.phase_rad is not None
    assert result.phase_rad.shape == result.data.shape
