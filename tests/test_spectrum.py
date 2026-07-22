"""Unit tests for spectrum FFT helpers."""

from __future__ import annotations

import numpy as np

from nagilize.core.spectrum import (
    SpectrumParams,
    compute_spectrum,
    frequency_resolution_hz,
    make_window,
    nfft_from_delta_f,
    resolve_nfft,
)


def test_compute_stft_shape_and_frames() -> None:
    from nagilize.core.spectrum import compute_stft, stft_frame_count, stft_hop_samples

    fs = 1000.0
    n = 4096
    t = np.arange(n) / fs
    y = np.sin(2.0 * np.pi * 50.0 * t)
    params = SpectrumParams(window="hann", nfft=512, overlap=0.5, stft_step_s=None)
    hop = stft_hop_samples(frame_len=512, sample_rate=fs, overlap=0.5)
    stft = compute_stft(y, fs, params, delta_f_hz=fs / 512)
    assert stft.n_frames == stft_frame_count(n, 512, hop)
    assert stft.magnitude.shape == (257, stft.n_frames)
    assert stft.phase_rad.shape == stft.magnitude.shape
    assert stft.time_s.shape == (stft.n_frames,)
    assert abs(stft.delta_f_hz - fs / 512) < 1e-12
    peak_bin = int(np.argmax(stft.magnitude[:, stft.n_frames // 2]))
    assert abs(stft.frequency_hz[peak_bin] - 50.0) < 2.0


def test_compute_stft_fixed_dt() -> None:
    from nagilize.core.spectrum import compute_stft, stft_hop_samples

    fs = 1000.0
    y = np.random.default_rng(1).normal(size=4096)
    hop = stft_hop_samples(frame_len=512, sample_rate=fs, step_s=0.01)
    stft = compute_stft(
        y,
        fs,
        SpectrumParams(window="hann", nfft=512, stft_step_s=0.01),
        delta_f_hz=fs / 512,
    )
    assert stft.hop_samples == hop
    assert stft.n_frames >= 2


def test_angle_from_pulse_increments_per_edge() -> None:
    from nagilize.core.spectrum import angle_from_pulse

    fs = 1000.0
    # Square pulse train: 4 rising edges → 4 * 360° with ppr=1
    y = np.zeros(400)
    for i in (50, 150, 250, 350):
        y[i : i + 20] = 1.0
    theta = angle_from_pulse(y, fs, pulses_per_rev=1)
    assert theta.shape == y.shape
    # At each rising edge index, angle should be k * 360
    edges = [50, 150, 250, 350]
    for k, idx in enumerate(edges):
        assert abs(theta[idx] - k * 360.0) < 1e-9
    theta2 = angle_from_pulse(y, fs, pulses_per_rev=2)
    for k, idx in enumerate(edges):
        assert abs(theta2[idx] - k * 180.0) < 1e-9


def test_angle_from_rpm_linear() -> None:
    from nagilize.core.spectrum import angle_from_rpm

    fs = 100.0
    n = 200
    rpm = np.full(n, 60.0)  # 60 RPM → 360 deg/s → 6 deg per sample at 100 Hz? 
    # ω_deg/s = rpm * 6 = 360 deg/s; dt = 0.01 → 3.6 deg/sample
    theta = angle_from_rpm(rpm, fs)
    assert theta.shape == (n,)
    assert abs(theta[0]) < 1e-12
    expected_end = 360.0 * (n - 1) / fs  # trapezoid ≈ n-1 steps of 3.6
    assert abs(theta[-1] - expected_end) < 1e-6
    # Nearly linear
    diffs = np.diff(theta)
    assert np.max(np.abs(diffs - diffs[0])) < 1e-9


def test_equal_angle_frame_count() -> None:
    from nagilize.core.spectrum import equal_angle_frame_starts

    fs = 1000.0
    n = 2000
    # θ increases 1 deg per sample
    theta = np.arange(n, dtype=np.float64)
    frame_len = 100
    dth = 10.0
    starts, times, angles = equal_angle_frame_starts(
        theta, frame_len=frame_len, delta_theta_deg=dth, sample_rate=fs
    )
    half = frame_len // 2
    i0 = half
    i1 = n - (frame_len - half)
    th_lo = float(theta[i0])
    th_hi = float(theta[i1 - 1])
    expected_centers = np.arange(th_lo, th_hi + 0.5 * dth, dth)
    # Unique starts after snapping may be ≤ expected centers
    assert len(starts) >= 1
    assert len(starts) == times.size == angles.size
    assert abs(angles[1] - angles[0] - dth) < 1.5  # ~Δθ spacing
    assert len(starts) == len(expected_centers) or len(starts) == len(expected_centers) - 1


def test_compute_stft_equal_angle() -> None:
    from nagilize.core.spectrum import angle_from_rpm, compute_stft

    fs = 1000.0
    n = 4096
    t = np.arange(n) / fs
    y = np.sin(2.0 * np.pi * 50.0 * t)
    rpm = np.full(n, 600.0)  # 10 rev/s → 3600 deg/s
    theta = angle_from_rpm(rpm, fs)
    params = SpectrumParams(
        window="hann",
        nfft=256,
        stft_step_mode="equal_angle",
        delta_theta_deg=10.0,
        tacho_kind="rpm",
    )
    stft = compute_stft(
        y, fs, params, delta_f_hz=fs / 256, theta_deg=theta, delta_theta_deg=10.0
    )
    assert stft.n_frames > 1
    assert stft.angle_deg.shape == (stft.n_frames,)
    assert stft.time_s.shape == (stft.n_frames,)
    # Angle spacing ≈ 10 deg (allow snap jitter)
    dth = np.diff(stft.angle_deg)
    assert np.median(dth) > 5.0
    assert np.median(dth) < 15.0


def test_rpm_from_pulse_constant_speed() -> None:
    from nagilize.core.spectrum import rpm_from_pulse

    fs = 1000.0
    n = 2000
    # 1200 RPM, 1 ppr → 20 pulses/s → period = 50 samples
    y = np.zeros(n)
    for i in range(50, n, 50):
        y[i : i + 5] = 1.0
    rpm = rpm_from_pulse(y, fs, pulses_per_rev=1)
    mid = rpm[200:1800]
    assert abs(float(np.median(mid)) - 1200.0) < 5.0


def test_equal_rpm_frame_count() -> None:
    from nagilize.core.spectrum import equal_rpm_frame_starts

    fs = 1000.0
    n = 5000
    # Linear ramp 600 → 2400 RPM
    rpm = np.linspace(600.0, 2400.0, n)
    frame_len = 256
    drpm = 10.0
    starts, times, rpms, angles = equal_rpm_frame_starts(
        rpm, frame_len=frame_len, delta_rpm=drpm, sample_rate=fs
    )
    assert len(starts) >= 1
    assert times.size == rpms.size == len(starts)
    assert angles.size == 0
    assert abs(float(np.median(np.diff(rpms))) - drpm) < 1.5
    # Rough expected count from usable RPM span
    half = frame_len // 2
    i0 = half
    i1 = n - (frame_len - half)
    span = float(rpm[i1 - 1] - rpm[i0])
    expected = int(span / drpm) + 1
    assert abs(len(starts) - expected) <= 2


def test_compute_stft_equal_rpm() -> None:
    from nagilize.core.spectrum import angle_from_rpm, compute_stft

    fs = 1000.0
    n = 4096
    t = np.arange(n) / fs
    y = np.sin(2.0 * np.pi * 50.0 * t)
    rpm = np.linspace(600.0, 1800.0, n)
    theta = angle_from_rpm(rpm, fs)
    params = SpectrumParams(
        window="hann",
        nfft=256,
        stft_step_mode="equal_rpm",
        delta_rpm=10.0,
        tacho_kind="rpm",
    )
    stft = compute_stft(
        y,
        fs,
        params,
        delta_f_hz=fs / 256,
        rpm=rpm,
        delta_rpm=10.0,
        theta_deg=theta,
    )
    assert stft.n_frames > 1
    assert stft.rpm.shape == (stft.n_frames,)
    assert stft.angle_deg.shape == (stft.n_frames,)
    assert abs(float(np.median(np.diff(stft.rpm))) - 10.0) < 1.5


def test_spectrogram_frame_rpm_roundtrip(tmp_path) -> None:
    from nagilize.core.dummy import make_sine_with_noise
    from nagilize.core.project import Project
    from nagilize.core.spectrum import angle_from_rpm, compute_stft

    project = Project()
    project.add_recording(make_sine_with_noise())
    series = project.get(project.series_order[0])
    assert series is not None
    rpm = np.linspace(600.0, 1800.0, series.data.shape[0])
    theta = angle_from_rpm(rpm, series.sample_rate)
    stft = compute_stft(
        series.data,
        series.sample_rate,
        SpectrumParams(
            window="hann",
            nfft=256,
            stft_step_mode="equal_rpm",
            delta_rpm=20.0,
        ),
        delta_f_hz=2.0,
        rpm=rpm,
        delta_rpm=20.0,
        theta_deg=theta,
    )
    assert stft.rpm.size == stft.n_frames
    dataset = project.create_analysis_dataset(
        label="STFT rpm test", provenance="analysis:stft"
    )
    sid = project.add_spectrogram_result(
        name="stft-rpm",
        frequency_hz=stft.frequency_hz,
        time_s=stft.time_s,
        magnitude=stft.magnitude,
        phase_rad=stft.phase_rad,
        sample_rate=series.sample_rate,
        parent=series,
        source=dataset,
        angle_deg=stft.angle_deg,
        rpm=stft.rpm,
        params_dict=stft.params.to_dict(),
    )
    path = tmp_path / "rpm.nagproj"
    project.save(path)
    loaded = Project.open(path)
    result = loaded.get(sid)
    assert result is not None
    result.ensure_loaded()
    assert result.frame_rpm is not None
    np.testing.assert_allclose(result.frame_rpm, stft.rpm)
    np.testing.assert_allclose(result.frame_time_s, stft.time_s)


def test_spectrogram_frame_angle_roundtrip(tmp_path) -> None:
    from nagilize.core.dummy import make_sine_with_noise
    from nagilize.core.project import Project
    from nagilize.core.spectrum import angle_from_rpm, compute_stft

    project = Project()
    project.add_recording(make_sine_with_noise())
    series = project.get(project.series_order[0])
    assert series is not None
    rpm = np.full(series.data.shape[0], 300.0)
    theta = angle_from_rpm(rpm, series.sample_rate)
    stft = compute_stft(
        series.data,
        series.sample_rate,
        SpectrumParams(
            window="hann",
            nfft=256,
            stft_step_mode="equal_angle",
            delta_theta_deg=15.0,
        ),
        delta_f_hz=2.0,
        theta_deg=theta,
        delta_theta_deg=15.0,
    )
    assert stft.angle_deg.size == stft.n_frames
    dataset = project.create_analysis_dataset(
        label="STFT angle test", provenance="analysis:stft"
    )
    sid = project.add_spectrogram_result(
        name="stft-angle",
        frequency_hz=stft.frequency_hz,
        time_s=stft.time_s,
        magnitude=stft.magnitude,
        phase_rad=stft.phase_rad,
        sample_rate=series.sample_rate,
        parent=series,
        source=dataset,
        angle_deg=stft.angle_deg,
        params_dict=stft.params.to_dict(),
    )
    path = tmp_path / "angle.nagproj"
    project.save(path)
    loaded = Project.open(path)
    result = loaded.get(sid)
    assert result is not None
    result.ensure_loaded()
    assert result.frame_angle_deg is not None
    np.testing.assert_allclose(result.frame_angle_deg, stft.angle_deg)
    np.testing.assert_allclose(result.frame_time_s, stft.time_s)


def test_spectrogram_project_roundtrip(tmp_path) -> None:
    from nagilize.core.dummy import make_sine_with_noise
    from nagilize.core.project import Project
    from nagilize.core.spectrum import compute_stft

    project = Project()
    project.add_recording(make_sine_with_noise())
    series = project.get(project.series_order[0])
    assert series is not None
    stft = compute_stft(
        series.data,
        series.sample_rate,
        SpectrumParams(window="hann", nfft=512, overlap=0.5),
        delta_f_hz=2.0,
    )
    dataset = project.create_analysis_dataset(
        label="STFT result test", provenance="analysis:stft"
    )
    sid = project.add_spectrogram_result(
        name="stft",
        frequency_hz=stft.frequency_hz,
        time_s=stft.time_s,
        magnitude=stft.magnitude,
        phase_rad=stft.phase_rad,
        sample_rate=series.sample_rate,
        parent=series,
        source=dataset,
        params_dict=stft.params.to_dict(),
    )
    path = tmp_path / "t.nagproj"
    project.save(path)
    loaded = Project.open(path)
    result = loaded.get(sid)
    assert result is not None
    result.ensure_loaded()
    assert result.is_spectrogram()
    assert result.frame_time_s is not None
    assert result.data.ndim == 2
    assert result.phase_rad is not None
    assert result.phase_rad.shape == result.data.shape
    assert result.time is not None
    assert result.time.shape[0] == result.data.shape[0]


def test_nfft_from_delta_f_roundtrip() -> None:
    fs = 1000.0
    for df in (1.0, 0.5, 3.90625):
        nfft = nfft_from_delta_f(fs, df)
        assert abs(frequency_resolution_hz(fs, nfft) - df) < 1e-9


def test_resolve_nfft_modes() -> None:
    fs = 1000.0
    n = 4096
    assert resolve_nfft(n, fs, cap_to_signal_length=True) == 1024
    assert resolve_nfft(n, fs) == n
    assert resolve_nfft(n, fs, delta_f_hz=2.0) == 500
    assert resolve_nfft(n, fs, delta_f_hz=0.1, cap_to_signal_length=True) == n


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
            nfft=512,
            amplitude_scale="lin",
        ),
    )
    assert spec.n_segments >= 2
    assert spec.frequency_hz.size == spec.magnitude.size
    assert abs(spec.delta_f_hz - frequency_resolution_hz(fs, 512)) < 1e-12


def test_stft_hop_and_frame_count() -> None:
    from nagilize.core.spectrum import (
        max_averages,
        stft_frame_count,
        stft_hop_samples,
        stft_overlap_equiv,
    )

    fs = 1000.0
    nfft = 512
    hop_ov = stft_hop_samples(frame_len=nfft, sample_rate=fs, overlap=0.5)
    assert hop_ov == 256
    hop_dt = stft_hop_samples(frame_len=nfft, sample_rate=fs, step_s=0.01)
    assert hop_dt == 10
    assert stft_frame_count(4096, nfft, hop_ov) == max_averages(4096, nfft, 0.5)
    assert stft_overlap_equiv(nfft, hop_dt) == 0.95


def test_averaging_count_clamped_to_max() -> None:
    from nagilize.core.spectrum import max_averages

    fs = 1000.0
    y = np.random.default_rng(0).normal(size=4096)
    nfft = 512
    overlap = 0.5
    available = max_averages(y.size, nfft, overlap)
    assert available >= 2
    limited = compute_spectrum(
        y,
        fs,
        SpectrumParams(
            averaging=True,
            overlap=overlap,
            nfft=nfft,
            n_averages=3,
            amplitude_scale="lin",
        ),
    )
    assert limited.n_segments == 3
    clamped = compute_spectrum(
        y,
        fs,
        SpectrumParams(
            averaging=True,
            overlap=overlap,
            nfft=nfft,
            n_averages=available + 100,
            amplitude_scale="lin",
        ),
    )
    assert clamped.n_segments == available


def test_peak_to_peak_scale() -> None:
    fs = 1000.0
    n = 1000
    t = np.arange(n) / fs
    y = np.sin(2.0 * np.pi * 50.0 * t)
    spec = compute_spectrum(
        y, fs, SpectrumParams(window="rectangular", amplitude_scale="ptp")
    )
    idx = int(np.argmax(spec.magnitude))
    assert abs(spec.magnitude[idx] - 2.0) < 0.1


def test_default_amplitude_scale_is_peak() -> None:
    fs = 1000.0
    n = 1000
    t = np.arange(n) / fs
    y = np.sin(2.0 * np.pi * 50.0 * t)
    spec = compute_spectrum(y, fs, SpectrumParams(window="rectangular"))
    idx = int(np.argmax(spec.magnitude))
    assert abs(spec.magnitude[idx] - 1.0) < 0.05


def test_explicit_nfft_below_length_changes_resolution() -> None:
    fs = 1000.0
    y = np.sin(2.0 * np.pi * 50.0 * np.arange(1000) / fs)
    full = compute_spectrum(y, fs, SpectrumParams(window="rectangular", nfft=None))
    short = compute_spectrum(
        y, fs, SpectrumParams(window="rectangular", nfft=256, amplitude_scale="peak")
    )
    assert abs(full.delta_f_hz - 1.0) < 1e-9
    assert abs(short.delta_f_hz - frequency_resolution_hz(fs, 256)) < 1e-9
    assert short.frequency_hz.size < full.frequency_hz.size


def test_project_spectrum_roundtrip(tmp_path) -> None:
    from nagilize.core.dummy import make_sine_with_noise
    from nagilize.core.project import Project

    project = Project()
    project.add_recording(make_sine_with_noise())
    series = project.get(project.series_order[0])
    assert series is not None
    spec = compute_spectrum(series.data, series.sample_rate, SpectrumParams(window="hann"))
    dataset = project.create_analysis_dataset(label="FFT result test")
    sid = project.add_spectrum_result(
        name="fft",
        frequency_hz=spec.frequency_hz,
        magnitude=spec.magnitude,
        phase_rad=spec.phase_rad,
        sample_rate=series.sample_rate,
        parent=series,
        source=dataset,
        params_dict=spec.params.to_dict(),
    )
    path = tmp_path / "t.nagproj"
    project.save(path)
    loaded = Project.open(path)
    result = loaded.get(sid)
    assert result is not None
    result.ensure_loaded()
    assert result.is_spectrum()
    assert result.source_id == dataset.id
    assert result.source_label == "FFT result test"
    assert result.phase_rad is not None
    assert result.phase_rad.shape == result.data.shape


def test_each_fft_run_makes_new_dataset() -> None:
    from nagilize.core.dummy import make_sine_with_noise
    from nagilize.core.project import Project

    project = Project()
    project.add_recording(make_sine_with_noise())
    series = project.get(project.series_order[0])
    assert series is not None
    spec = compute_spectrum(series.data, series.sample_rate, SpectrumParams())
    d1 = project.create_analysis_dataset(label="FFT A")
    d2 = project.create_analysis_dataset(label="FFT B")
    project.add_spectrum_result(
        name="a",
        frequency_hz=spec.frequency_hz,
        magnitude=spec.magnitude,
        phase_rad=spec.phase_rad,
        sample_rate=series.sample_rate,
        parent=series,
        source=d1,
    )
    project.add_spectrum_result(
        name="b",
        frequency_hz=spec.frequency_hz,
        magnitude=spec.magnitude,
        phase_rad=spec.phase_rad,
        sample_rate=series.sample_rate,
        parent=series,
        source=d2,
    )
    assert d1.id != d2.id
    labels = {s.label for s in project.sources}
    assert "FFT A" in labels and "FFT B" in labels
    by_src = {}
    for s in project.all_series():
        if s.is_spectrum():
            by_src.setdefault(s.source_id, []).append(s)
    assert len(by_src[d1.id]) == 1
    assert len(by_src[d2.id]) == 1
