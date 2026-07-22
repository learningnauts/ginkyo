"""Write demo CSVs with vibration + tacho (pulse or RPM) for equal-RPM STFT.

Both demos use the same run-up (600 → 2400 RPM) so Equal RPM (ΔRPM=10)
places frames on a rising speed axis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nagilize.core.model import Channel, Recording
from nagilize.export.csv_export import export_csv


def _vibration_from_angle(
    theta_deg: np.ndarray,
    *,
    amplitudes: tuple[float, ...] = (1.0, 0.45, 0.2),
    noise_std: float = 0.04,
    seed: int = 0,
) -> np.ndarray:
    """Synthetic vibration with shaft orders 1X / 2X / 3X + noise."""
    rng = np.random.default_rng(seed)
    th = np.deg2rad(np.asarray(theta_deg, dtype=np.float64))
    y = np.zeros_like(th)
    for k, amp in enumerate(amplitudes, start=1):
        y += amp * np.sin(k * th)
    y += noise_std * rng.standard_normal(th.size)
    return y.astype(np.float64)


def _rpm_ramp(
    t: np.ndarray,
    *,
    duration_s: float,
    rpm_start: float,
    rpm_end: float,
) -> np.ndarray:
    return rpm_start + (rpm_end - rpm_start) * (t / max(duration_s, 1e-12))


def _theta_from_rpm(rpm: np.ndarray, sample_rate: float) -> np.ndarray:
    dt = 1.0 / float(sample_rate)
    omega = np.asarray(rpm, dtype=np.float64) * 6.0
    theta = np.empty(omega.size, dtype=np.float64)
    theta[0] = 0.0
    if omega.size > 1:
        theta[1:] = np.cumsum(0.5 * (omega[:-1] + omega[1:]) * dt)
    return theta


def _pulse_train_from_rpm(
    rpm: np.ndarray,
    sample_rate: float,
    *,
    pulses_per_rev: int = 1,
    duty: float = 0.08,
) -> np.ndarray:
    """TTL-like pulses whose period tracks instantaneous RPM (run-up)."""
    n = int(rpm.size)
    y = np.zeros(n, dtype=np.float64)
    ppr = max(1, int(pulses_per_rev))
    # Place edges whenever cumulative revolutions advance by 1/ppr.
    theta = _theta_from_rpm(rpm, sample_rate)
    # Edge every 360/ppr degrees.
    step = 360.0 / float(ppr)
    targets = np.arange(step, float(theta[-1]) + 0.5 * step, step)
    if targets.size == 0:
        return y
    idx = np.searchsorted(theta, targets, side="left")
    idx = np.clip(idx, 0, n - 1)
    # Pulse width ~ duty of local period (samples).
    for i in idx.astype(int):
        local_rpm = max(float(rpm[i]), 1.0)
        period = sample_rate * 60.0 / (local_rpm * ppr)
        width = max(1, int(round(period * duty)))
        y[i : min(n, i + width)] = 1.0
    return y


def make_pulse_recording(
    *,
    duration_s: float = 4.0,
    sample_rate: float = 5_000.0,
    rpm_start: float = 600.0,
    rpm_end: float = 2_400.0,
    pulses_per_rev: int = 1,
) -> Recording:
    n = int(duration_s * sample_rate)
    t = np.arange(n, dtype=np.float64) / sample_rate
    rpm = _rpm_ramp(t, duration_s=duration_s, rpm_start=rpm_start, rpm_end=rpm_end)
    theta = _theta_from_rpm(rpm, sample_rate)
    vib = _vibration_from_angle(theta, seed=1)
    pulse = _pulse_train_from_rpm(
        rpm, sample_rate, pulses_per_rev=pulses_per_rev
    )
    return Recording(
        sample_rate=sample_rate,
        source="demo:tacho_pulse",
        time=t,
        channels=[
            Channel(name="Vibration", data=vib, unit="a.u."),
            Channel(name="Tacho_pulse", data=pulse, unit=""),
        ],
    )


def make_rpm_recording(
    *,
    duration_s: float = 4.0,
    sample_rate: float = 5_000.0,
    rpm_start: float = 600.0,
    rpm_end: float = 2_400.0,
) -> Recording:
    n = int(duration_s * sample_rate)
    t = np.arange(n, dtype=np.float64) / sample_rate
    rpm = _rpm_ramp(t, duration_s=duration_s, rpm_start=rpm_start, rpm_end=rpm_end)
    theta = _theta_from_rpm(rpm, sample_rate)
    vib = _vibration_from_angle(theta, seed=2)
    return Recording(
        sample_rate=sample_rate,
        source="demo:tacho_rpm",
        time=t,
        channels=[
            Channel(name="Vibration", data=vib, unit="a.u."),
            Channel(name="RPM", data=rpm.astype(np.float64), unit="rpm"),
        ],
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "samples"
    root.mkdir(parents=True, exist_ok=True)

    pulse_path = root / "demo_tacho_pulse.csv"
    rpm_path = root / "demo_tacho_rpm.csv"

    export_csv(make_pulse_recording(), pulse_path)
    export_csv(make_rpm_recording(), rpm_path)

    print(f"wrote {pulse_path}")
    print(f"wrote {rpm_path}")
    print(
        "Usage: File → Add file… → Analysis STFT → Equal RPM\n"
        "  · pulse CSV: Tacho = Tacho_pulse, Kind = Pulse, Pulses/rev = 1, ΔRPM = 10\n"
        "  · rpm CSV:   Tacho = RPM, Kind = RPM, ΔRPM = 10\n"
        "Views: right-click spectrogram → Y axis → RPM"
    )


if __name__ == "__main__":
    main()
