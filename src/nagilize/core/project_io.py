"""Load / save ``.nagproj`` (zip) projects with on-demand waveform cache (Y1b)."""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from nagilize.core.model import SeriesMeta
from nagilize.core.project import Project, Series, SourceRef

FORMAT_VERSION = 1
PROJECT_JSON = "project.json"


def _safe_entry_name(series_id: str) -> str:
    return series_id.replace(":", "_").replace("/", "_")


def save_project(
    project: Project,
    path: str | Path,
    *,
    views: list[dict[str, Any]] | None = None,
) -> None:
    """Write project as a single ``.nagproj`` zip (embedded arrays + catalog)."""
    path = Path(path)
    if path.suffix.lower() != ".nagproj":
        path = path.with_suffix(".nagproj")

    # Materialize every series so the archive is self-contained.
    for sid in project.series_order:
        series = project.series.get(sid)
        if series is not None:
            series.ensure_loaded()

    catalog: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "sources": [
            {
                "id": src.id,
                "label": src.label,
                "provenance": src.provenance
                or (src.recording.source if src.recording is not None else ""),
            }
            for src in project.sources
        ],
        "series_order": list(project.series_order),
        "series": [],
        "views": list(views or []),
    }

    tmp_dir = Path(tempfile.mkdtemp(prefix="nagilize-save-"))
    try:
        data_dir = tmp_dir / "data"
        data_dir.mkdir()
        for sid in project.series_order:
            series = project.series.get(sid)
            if series is None:
                continue
            safe = _safe_entry_name(sid)
            data_name = f"data/{safe}.npy"
            time_name = None
            phase_name = None
            np.save(data_dir / f"{safe}.npy", series.data)
            if series.time is not None:
                time_name = f"data/{safe}.time.npy"
                np.save(data_dir / f"{safe}.time.npy", series.time)
            if series.phase_rad is not None:
                phase_name = f"data/{safe}.phase.npy"
                np.save(data_dir / f"{safe}.phase.npy", series.phase_rad)
            catalog["series"].append(
                {
                    "id": series.id,
                    "name": series.name,
                    "sample_rate": float(series.sample_rate),
                    "unit": series.unit,
                    "source_id": series.source_id,
                    "source_label": series.source_label,
                    "channel_index": int(series.channel_index),
                    "n_samples": int(series.n_samples),
                    "data_entry": data_name,
                    "time_entry": time_name,
                    "phase_entry": phase_name,
                    "meta": series.meta.to_dict(),
                }
            )

        (tmp_dir / PROJECT_JSON).write_text(
            json.dumps(catalog, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        staging = path.with_suffix(path.suffix + ".tmp")
        if staging.exists():
            staging.unlink()
        with zipfile.ZipFile(staging, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.write(tmp_dir / PROJECT_JSON, PROJECT_JSON)
            for p in sorted(data_dir.glob("*.npy")):
                zf.write(p, f"data/{p.name}")
        staging.replace(path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def load_project(path: str | Path) -> Project:
    """Open a ``.nagproj``; waveforms load into a temp cache on first use (Y1b)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    project = Project(path=path)
    cache = Path(tempfile.mkdtemp(prefix="nagilize-cache-"))
    project._cache_dir = cache

    with zipfile.ZipFile(path, "r") as zf:
        with zf.open(PROJECT_JSON) as fh:
            catalog = json.loads(fh.read().decode("utf-8"))

    if int(catalog.get("format_version") or 1) > FORMAT_VERSION:
        raise ValueError(
            f"Unsupported .nagproj format_version={catalog.get('format_version')}"
        )

    for src in catalog.get("sources") or []:
        project.sources.append(
            SourceRef(
                id=str(src.get("id") or ""),
                label=str(src.get("label") or "(unnamed)"),
                recording=None,
                provenance=str(src.get("provenance") or ""),
            )
        )

    zip_path = path

    def make_loader(
        data_entry: str, time_entry: str | None, phase_entry: str | None
    ):
        def _load() -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
            data_cache = cache / Path(data_entry).name
            if not data_cache.exists():
                with zipfile.ZipFile(zip_path, "r") as zf:
                    data_cache.write_bytes(zf.read(data_entry))
                    if time_entry:
                        time_cache = cache / Path(time_entry).name
                        time_cache.write_bytes(zf.read(time_entry))
                    if phase_entry:
                        phase_cache = cache / Path(phase_entry).name
                        if phase_entry in zf.namelist():
                            phase_cache.write_bytes(zf.read(phase_entry))
            data = np.load(data_cache)
            time = None
            if time_entry:
                time_cache = cache / Path(time_entry).name
                if not time_cache.exists():
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        time_cache.write_bytes(zf.read(time_entry))
                time = np.load(time_cache)
            phase = None
            if phase_entry:
                phase_cache = cache / Path(phase_entry).name
                if not phase_cache.exists():
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        if phase_entry in zf.namelist():
                            phase_cache.write_bytes(zf.read(phase_entry))
                if phase_cache.exists():
                    phase = np.load(phase_cache)
            return (
                np.asarray(data, dtype=np.float64),
                None if time is None else np.asarray(time, dtype=float),
                None if phase is None else np.asarray(phase, dtype=np.float64),
            )

        return _load

    series_rows = {str(row["id"]): row for row in (catalog.get("series") or [])}
    order = [str(x) for x in (catalog.get("series_order") or list(series_rows.keys()))]
    for sid in order:
        row = series_rows.get(sid)
        if row is None:
            continue
        data_entry = str(row.get("data_entry") or "")
        time_entry = row.get("time_entry")
        time_entry_s = str(time_entry) if time_entry else None
        phase_entry = row.get("phase_entry")
        phase_entry_s = str(phase_entry) if phase_entry else None
        series = Series.lazy(
            id=sid,
            name=str(row.get("name") or sid),
            sample_rate=float(row.get("sample_rate") or 0.0),
            n_samples=int(row.get("n_samples") or 0),
            loader=make_loader(data_entry, time_entry_s, phase_entry_s),
            unit=str(row.get("unit") or ""),
            source_id=str(row.get("source_id") or ""),
            source_label=str(row.get("source_label") or ""),
            channel_index=int(row.get("channel_index") or 0),
            meta=SeriesMeta.from_dict(row.get("meta")),
        )
        project.series[sid] = series
        project.series_order.append(sid)

    project._pending_views = list(catalog.get("views") or [])  # type: ignore[attr-defined]
    return project


def take_pending_views(project: Project) -> list[dict[str, Any]]:
    views = getattr(project, "_pending_views", None)
    if views is None:
        return []
    delattr(project, "_pending_views")
    return list(views)
