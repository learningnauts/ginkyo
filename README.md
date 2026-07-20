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
- **Analyze → Spectrum (FFT)…** / **Analysis** tab — window, NFFT, Welch averaging, amplitude scale; Run adds one spectrum result. `fs` / `Δf` are derived from the measurement and settings
- **Project data** — tree by source; row shows `name | point · dof | ← ref · ref_dof`; filter / sort; right-click series → Edit metadata; drag onto plots (spectrum drop converts that cell to Mag+Phase). Drag the splitter to resize the left pane
- **View → New / Close / Rename page** — multiple view pages (Analysis tab stays open)
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

## License

MIT — see [LICENSE](LICENSE).

## Notes

Product vision and milestones live in the personal workspace memo  
(`Mybrain/03_SideHustle/app/lanxi/lanxi_notes.md`), not in this repo.
