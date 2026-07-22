# nagilize（ナギライズ）

振動・音データの解析ワークベンチ（開発中）。

## Status

**Project file + series meta + pages** — save/open `.nagproj`; series carry point/DOF/FRF reference fields; multi-page views; layout presets; DnD assign.

**Spectrum results (M3)** — Analysis page FFT → one Mag+Phase result in the project; drop onto a plot cell for Mag (top) / Phase (bottom).

## Requirements

- Python 3.9+
- macOS / Windows / Linux (desktop GUI)

## Quick start

```bash
cd /path/to/nagilize
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --no-compile -e .
python -m nagilize
```

> Tip: if full `PySide6` install fails on compile of template files, use `PySide6-Essentials` (already pinned) and/or `pip install --no-compile`.

### In the app

- **File → Open project… / Save project…** — `.nagproj` (single zip file; waveforms embedded; views restored)
- **File → Add file…** — add `.wav` / `.csv` / `.uff` / `.unv` into the project
- **File → Export selected source CSV…** — export the source selected in **Project data**
- **Analyze → Spectrum (FFT)…** / top ribbon **Analysis** — select series in Project data → Add; optional **Dataset name** (auto `FFT N · …` if blank); Run creates one new dataset per run
- **Views** (ribbon) — Page 1 / Page 2 / … tabs at the bottom for plot layouts; drag spectrum results onto a cell for Mag+Phase
- **Project data** — tree by source; row shows `name | point · dof | ← ref · ref_dof`; filter / sort; right-click series → Edit metadata; drag onto plots. Drag the splitter to resize the left pane
- **View → New / Close / Rename page** — pages live under **Views** (ribbon), not mixed with Analysis
- **View → Spectrum** — optional whole-page Time / Mag+Phase / Real+Imag preview (legacy on-the-fly FFT)
- **View → Cursor values** — optional floating table: mouse cursor **and** vertical marker positions / series values (hidden by default)
- **View → Layout** — presets; drag splitters to resize
- **View → Reset zoom** (`Ctrl+0`)
- **View → Link X axes / Link Y axes** — toggle pan/zoom linking across panels (X on by default, Y off)

### From Python (read-only)

```python
from nagilize import Project

proj = Project.open("Run.nagproj")
for sid in proj.series_order:
    s = proj.get(sid)
    print(s.name, s.meta.point_id, s.meta.dof)
    # Waveform loads on first access (cached):
    y = s.data
```

### Samples

| File | Notes |
|------|--------|
| `samples/demo_sine_stereo.wav` | stereo sine (M1) |
| `samples/demo_sine_stereo.csv` | exported CSV of the same |
| `samples/demo_time.uff` | 2-ch dataset-58 time (`scripts/make_demo_uff.py`) |
| `samples/demo_10ch_time.uff` | 10-ch dataset-58 time (`scripts/make_demo_10ch_uff.py`) |
| `samples/demo_tacho_pulse.csv` | Vibration + pulse tacho (run-up 600→2400 RPM, 1 ppr; `scripts/make_demo_tacho.py`) |
| `samples/demo_tacho_rpm.csv` | Vibration + RPM run-up 600→2400 (`scripts/make_demo_tacho.py`) |

Equal-RPM STFT: add either CSV → Analysis → STFT → **Equal RPM** (ΔRPM=10) → pick `Tacho_pulse` (Kind **Pulse**, Pulses/rev **1**) or `RPM` (Kind **RPM**). Views right-click → **Y axis → RPM**.

## License

MIT — see [LICENSE](LICENSE).

## Notes

Product vision and milestones live in the personal workspace memo  
(`Mybrain/03_SideHustle/app/lanxi/lanxi_notes.md`), not in this repo.
