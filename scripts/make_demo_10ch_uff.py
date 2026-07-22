"""Create a dataset-58 time-response UFF with ~10 channels for UI/analysis demos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyuff


def _fill_nones(data: dict) -> dict:
    """pyuff prepare_58 leaves some numeric fields as None; writers need numbers."""
    for key, value in list(data.items()):
        if value is not None:
            continue
        if "lab" in key or "name" in key or key.startswith("id"):
            data[key] = ""
        else:
            data[key] = 0
    return data


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "samples" / "demo_10ch_time.uff"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    fs = 2048.0
    duration_s = 2.0
    n = int(fs * duration_s)
    t = np.arange(n, dtype=np.float64) / fs
    rng = np.random.default_rng(42)

    # Ten distinct tones (+ light noise) so Mag+Phase / multi-select demos are obvious.
    specs = [
        (1, 12.5, 1.00, "Point1 · Z"),
        (2, 25.0, 0.80, "Point2 · Z"),
        (3, 37.5, 0.65, "Point3 · Z"),
        (4, 50.0, 0.55, "Point4 · Z"),
        (5, 62.5, 0.45, "Point5 · Z"),
        (6, 75.0, 0.40, "Point6 · X"),
        (7, 87.5, 0.35, "Point7 · Y"),
        (8, 100.0, 0.30, "Point8 · Z"),
        (9, 125.0, 0.25, "Point9 · Z"),
        (10, 150.0, 0.20, "Point10 · Z"),
    ]

    uff = pyuff.UFF(str(out))
    for node, freq_hz, amp, label in specs:
        y = amp * np.sin(2.0 * np.pi * freq_hz * t)
        y = y + 0.02 * amp * rng.normal(size=n)
        data = pyuff.prepare_58(
            binary=0,
            func_type=1,  # time response
            ver_num=1,
            load_case_id=0,
            rsp_ent_name="NONE",
            rsp_node=int(node),
            rsp_dir=3,
            ref_ent_name="NONE",
            ref_node=0,
            ref_dir=0,
            id1=label,
            id2=f"f0={freq_hz:g} Hz",
            id3="nagilize demo 10ch",
            id4="",
            id5="",
            data=y.astype(np.float64),
            x=t,
            abscissa_spacing=1,
            abscissa_min=0.0,
            abscissa_inc=1.0 / fs,
            num_pts=n,
            ord_data_type=2,
            abscissa_spec_data_type=17,  # time
            ordinate_spec_data_type=12,  # acceleration (label only)
            orddenom_spec_data_type=0,
            ordinate_axis_units_lab="m/s2",
            return_full_dict=True,
        )
        uff._write_set(_fill_nones(data), "add")

    print(f"wrote {out} ({len(specs)} channels, fs={fs:g} Hz, n={n})")


if __name__ == "__main__":
    main()
