"""Dummy signal generator for M0 (no hardware / no files)."""

from __future__ import annotations

import numpy as np

from nagilize.core import Channel, Recording


def make_sine_with_noise(
    *,
    duration_s: float = 1.0,
    sample_rate: float = 10_000.0,
    frequency_hz: float = 50.0,
    amplitude: float = 1.0,
    noise_std: float = 0.05,
    seed: int = 0,
) -> Recording:
    """Generate a single-channel sine wave plus Gaussian noise."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * sample_rate)
    t = np.arange(n, dtype=float) / sample_rate
    signal = amplitude * np.sin(2.0 * np.pi * frequency_hz * t)
    signal = signal + noise_std * rng.standard_normal(n)

    return Recording(
        sample_rate=sample_rate,
        source="dummy:sine+noise",
        channels=[
            Channel(name="Ch1", data=signal.astype(np.float64), unit="a.u."),
        ],
    )
