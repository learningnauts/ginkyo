"""Write a small demo WAV for M1 manual testing."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def write_demo_wav(path: Path, *, duration_s: float = 0.5, fs: float = 8000.0) -> None:
    t = np.arange(int(duration_s * fs), dtype=np.float64) / fs
    # stereo: 440 Hz + 880 Hz
    left = 0.4 * np.sin(2 * np.pi * 440 * t)
    right = 0.3 * np.sin(2 * np.pi * 880 * t)
    stereo = np.column_stack([left, right])
    pcm = np.clip(stereo * 32767.0, -32768, 32767).astype("<i2")

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(int(fs))
        wf.writeframes(pcm.tobytes())


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "samples" / "demo_sine_stereo.wav"
    write_demo_wav(out)
    print(f"wrote {out}")
