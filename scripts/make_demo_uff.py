"""Create a small dataset-58 time-response UFF for testing."""

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
    out = Path(__file__).resolve().parents[1] / "samples" / "demo_time.uff"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    fs = 1000.0
    n = 500
    t = np.arange(n) / fs
    y1 = 0.5 * np.sin(2 * np.pi * 20 * t)
    y2 = 0.3 * np.sin(2 * np.pi * 45 * t)

    uff = pyuff.UFF(str(out))
    for i, (y, node) in enumerate([(y1, 1), (y2, 2)], start=1):
        data = pyuff.prepare_58(
            binary=0,
            func_type=1,  # time response
            ver_num=1,
            load_case_id=0,
            rsp_ent_name="NONE",
            rsp_node=node,
            rsp_dir=3,
            ref_ent_name="NONE",
            ref_node=0,
            ref_dir=0,
            id1=f"Demo time CH{i}",
            id2="",
            id3="",
            id4="",
            id5="",
            data=y,
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

    print(f"wrote {out}")


if __name__ == "__main__":
    main()
