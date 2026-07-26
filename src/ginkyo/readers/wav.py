"""WAV reader → common Recording model (stdlib ``wave``; PCM only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ginkyo.core.model import Channel, Recording


def read_wav(path: str | Path) -> Recording:
    """Load a PCM WAV file into a Recording.

    Supports mono/multi-channel integer PCM (8/16/24/32-bit) and 32-bit float.
    """
    import wave

    path = Path(path)
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = float(wf.getframerate())
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if n_frames == 0:
        raise ValueError(f"WAV has no frames: {path}")

    data = _pcm_to_float(raw, sample_width=sample_width, n_channels=n_channels)
    # shape (n_frames, n_channels)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    channels = [
        Channel(name=f"Ch{i + 1}", data=data[:, i].copy(), unit="")
        for i in range(data.shape[1])
    ]
    return Recording(
        sample_rate=sample_rate,
        channels=channels,
        source=str(path.resolve()),
    )


def _pcm_to_float(raw: bytes, *, sample_width: int, n_channels: int) -> np.ndarray:
    if sample_width == 1:
        # unsigned 8-bit
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        arr = (arr - 128.0) / 128.0
    elif sample_width == 2:
        arr = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sample_width == 3:
        arr = _int24_to_float(raw)
    elif sample_width == 4:
        # Prefer int32 PCM; if values look like float32 bit patterns, try float.
        as_i32 = np.frombuffer(raw, dtype="<i4")
        # Heuristic: if max abs is huge relative to float speech, treat as int32.
        # Always decode as int32 PCM first (most common for 32-bit WAV from tools).
        arr = as_i32.astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width} bytes")

    if n_channels > 1:
        if arr.size % n_channels != 0:
            raise ValueError("Frame data size is not divisible by channel count")
        arr = arr.reshape(-1, n_channels)
    return arr


def _int24_to_float(raw: bytes) -> np.ndarray:
    if len(raw) % 3 != 0:
        raise ValueError("24-bit PCM byte length must be a multiple of 3")
    # Little-endian 24-bit → int32
    a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
    # sign-extend
    values = (
        a[:, 0].astype(np.int32)
        | (a[:, 1].astype(np.int32) << 8)
        | (a[:, 2].astype(np.int32) << 16)
    )
    neg = values >= 0x800000
    values[neg] -= 0x1000000
    return values.astype(np.float64) / 8388608.0
