"""Project: multiple recordings expanded into a shared series pool."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from uuid import uuid4

import numpy as np

from nagilize.core.model import Recording, SeriesMeta


@dataclass
class Series:
    """One plottable series in the project pool (waveform may load on demand)."""

    id: str
    name: str
    sample_rate: float
    unit: str = ""
    source_id: str = ""
    source_label: str = ""
    channel_index: int = 0
    meta: SeriesMeta = field(default_factory=SeriesMeta)
    n_samples_hint: int = 0
    _data: np.ndarray | None = field(default=None, repr=False, compare=False)
    _time: np.ndarray | None = field(default=None, repr=False, compare=False)
    _phase: np.ndarray | None = field(default=None, repr=False, compare=False)
    _loader: Callable[
        [], tuple[np.ndarray, np.ndarray | None, np.ndarray | None]
    ] | None = field(default=None, repr=False, compare=False)

    @classmethod
    def create(
        cls,
        *,
        id: str,
        name: str,
        sample_rate: float,
        data: np.ndarray,
        unit: str = "",
        source_id: str = "",
        source_label: str = "",
        channel_index: int = 0,
        time: np.ndarray | None = None,
        phase: np.ndarray | None = None,
        meta: SeriesMeta | None = None,
    ) -> Series:
        arr = np.asarray(data, dtype=np.float64)
        return cls(
            id=id,
            name=name,
            sample_rate=float(sample_rate),
            unit=unit,
            source_id=source_id,
            source_label=source_label,
            channel_index=channel_index,
            meta=meta or SeriesMeta(),
            n_samples_hint=int(arr.shape[0]),
            _data=arr,
            _time=None if time is None else np.asarray(time, dtype=float),
            _phase=None if phase is None else np.asarray(phase, dtype=np.float64),
            _loader=None,
        )

    @classmethod
    def lazy(
        cls,
        *,
        id: str,
        name: str,
        sample_rate: float,
        n_samples: int,
        loader: Callable[[], tuple[np.ndarray, np.ndarray | None, np.ndarray | None]],
        unit: str = "",
        source_id: str = "",
        source_label: str = "",
        channel_index: int = 0,
        meta: SeriesMeta | None = None,
    ) -> Series:
        return cls(
            id=id,
            name=name,
            sample_rate=float(sample_rate),
            unit=unit,
            source_id=source_id,
            source_label=source_label,
            channel_index=channel_index,
            meta=meta or SeriesMeta(),
            n_samples_hint=int(n_samples),
            _data=None,
            _time=None,
            _phase=None,
            _loader=loader,
        )

    @property
    def is_loaded(self) -> bool:
        return self._data is not None

    def ensure_loaded(self) -> None:
        if self._data is not None:
            return
        if self._loader is None:
            self._data = np.zeros(0, dtype=np.float64)
            self._time = None
            self._phase = None
            return
        loaded = self._loader()
        if len(loaded) == 2:
            data, time = loaded
            phase = None
        else:
            data, time, phase = loaded
        self._data = np.asarray(data, dtype=np.float64)
        self._time = None if time is None else np.asarray(time, dtype=float)
        self._phase = None if phase is None else np.asarray(phase, dtype=np.float64)
        self.n_samples_hint = int(self._data.shape[0])
        self._loader = None

    @property
    def data(self) -> np.ndarray:
        self.ensure_loaded()
        assert self._data is not None
        return self._data

    @data.setter
    def data(self, value: np.ndarray) -> None:
        self._data = np.asarray(value, dtype=np.float64)
        self.n_samples_hint = int(self._data.shape[0])
        self._loader = None

    @property
    def time(self) -> np.ndarray | None:
        self.ensure_loaded()
        return self._time

    @time.setter
    def time(self, value: np.ndarray | None) -> None:
        self.ensure_loaded()
        self._time = None if value is None else np.asarray(value, dtype=float)

    def time_axis(self) -> np.ndarray:
        self.ensure_loaded()
        if self._time is not None:
            return np.asarray(self._time, dtype=float)
        n = int(self._data.shape[0]) if self._data is not None else 0
        if self.sample_rate <= 0:
            return np.arange(n, dtype=float)
        return np.arange(n, dtype=float) / self.sample_rate

    @property
    def phase_rad(self) -> np.ndarray | None:
        """Phase radians for spectrum results; None for time waveforms."""
        self.ensure_loaded()
        return self._phase

    @phase_rad.setter
    def phase_rad(self, value: np.ndarray | None) -> None:
        self.ensure_loaded()
        self._phase = None if value is None else np.asarray(value, dtype=np.float64)

    def is_spectrum(self) -> bool:
        return (self.meta.quantity or "").strip().lower() == "spectrum"

    @property
    def n_samples(self) -> int:
        if self._data is not None:
            return int(self._data.shape[0])
        return int(self.n_samples_hint)

    @property
    def duration_s(self) -> float:
        if not self.is_loaded and self.sample_rate > 0 and self.n_samples_hint > 0:
            return float(self.n_samples_hint - 1) / float(self.sample_rate)
        t = self.time_axis()
        if t.size == 0:
            return 0.0
        return float(t[-1] - t[0]) if t.size > 1 else 0.0

    @property
    def display_name(self) -> str:
        return f"{self.source_label} / {self.name}"

    def tree_label(self) -> str:
        return self.meta.tree_row_text(self.name)


@dataclass
class SourceRef:
    """A recording added to the project (kept for export / remove)."""

    id: str
    label: str
    recording: Recording | None = None
    provenance: str = ""


@dataclass
class Project:
    """Pool of series from one or more sources."""

    sources: list[SourceRef] = field(default_factory=list)
    series: dict[str, Series] = field(default_factory=dict)
    series_order: list[str] = field(default_factory=list)
    path: Path | None = None
    _cache_dir: Path | None = field(default=None, repr=False, compare=False)

    def clear(self) -> None:
        self.sources.clear()
        self.series.clear()
        self.series_order.clear()
        self.path = None
        self._cleanup_cache()

    def _cleanup_cache(self) -> None:
        if self._cache_dir is None:
            return
        import shutil

        try:
            shutil.rmtree(self._cache_dir, ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass
        self._cache_dir = None

    def get(self, series_id: str) -> Series | None:
        return self.series.get(series_id)

    def all_series(self) -> list[Series]:
        return [self.series[i] for i in self.series_order if i in self.series]

    def source_by_id(self, source_id: str) -> SourceRef | None:
        for src in self.sources:
            if src.id == source_id:
                return src
        return None

    def add_recording(self, recording: Recording) -> list[str]:
        """Expand channels into series. Returns new series ids (in order)."""
        source_id = uuid4().hex[:12]
        label = _source_label(recording.source)
        provenance = recording.source or label
        self.sources.append(
            SourceRef(
                id=source_id,
                label=label,
                recording=recording,
                provenance=provenance,
            )
        )
        new_ids: list[str] = []
        shared_time = recording.time
        for ci, ch in enumerate(recording.channels):
            sid = f"{source_id}:ch{ci}"
            series = Series.create(
                id=sid,
                name=ch.name,
                data=np.asarray(ch.data, dtype=np.float64),
                sample_rate=float(recording.sample_rate),
                unit=ch.unit,
                source_id=source_id,
                source_label=label,
                channel_index=ci,
                time=None if shared_time is None else np.asarray(shared_time, dtype=float),
                meta=SeriesMeta(quantity="time", provenance=provenance),
            )
            self.series[sid] = series
            self.series_order.append(sid)
            new_ids.append(sid)
        return new_ids

    def remove_source(self, source_id: str) -> list[str]:
        """Remove a source and its series. Returns removed series ids."""
        removed = [
            sid
            for sid in list(self.series_order)
            if self.series.get(sid) is not None
            and self.series[sid].source_id == source_id
        ]
        for sid in removed:
            self.series.pop(sid, None)
            if sid in self.series_order:
                self.series_order.remove(sid)
        self.sources = [s for s in self.sources if s.id != source_id]
        return removed

    def prune_ids(self, series_ids: list[str]) -> list[str]:
        """Keep only ids that still exist in the pool."""
        return [sid for sid in series_ids if sid in self.series]

    def ensure_analysis_source(self) -> SourceRef:
        """Return (creating if needed) the synthetic Analysis source."""
        existing = self.source_by_id("analysis")
        if existing is not None:
            return existing
        src = SourceRef(id="analysis", label="Analysis", recording=None, provenance="analysis")
        self.sources.append(src)
        return src

    def add_spectrum_result(
        self,
        *,
        name: str,
        frequency_hz: np.ndarray,
        magnitude: np.ndarray,
        phase_rad: np.ndarray,
        sample_rate: float,
        unit: str = "",
        parent: Series | None = None,
        params_dict: dict | None = None,
        attrs: dict | None = None,
    ) -> str:
        """Add one spectrum result (Mag+Phase) to the series pool. Returns series id."""
        self.ensure_analysis_source()
        sid = f"analysis:{uuid4().hex[:10]}"
        freq = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
        mag = np.asarray(magnitude, dtype=np.float64).reshape(-1)
        phase = np.asarray(phase_rad, dtype=np.float64).reshape(-1)
        if freq.size != mag.size or mag.size != phase.size:
            raise ValueError("frequency, magnitude, and phase must have the same length")
        meta_attrs = dict(attrs or {})
        if params_dict:
            meta_attrs["spectrum_params"] = dict(params_dict)
        if parent is not None:
            meta_attrs["parent_series_id"] = parent.id
        meta = SeriesMeta(
            quantity="spectrum",
            point_id=parent.meta.point_id if parent else "",
            point_name=parent.meta.point_name if parent else "",
            dof=parent.meta.dof if parent else "",
            ref_point_id=parent.meta.ref_point_id if parent else "",
            ref_point_name=parent.meta.ref_point_name if parent else "",
            ref_dof=parent.meta.ref_dof if parent else "",
            provenance=(
                f"FFT of {parent.display_name}" if parent is not None else "FFT"
            ),
            attrs=meta_attrs,
        )
        series = Series.create(
            id=sid,
            name=name,
            sample_rate=float(sample_rate),
            data=mag,
            unit=unit,
            source_id="analysis",
            source_label="Analysis",
            channel_index=0,
            time=freq,
            phase=phase,
            meta=meta,
        )
        self.series[sid] = series
        self.series_order.append(sid)
        return sid

    def ensure_series_loaded(self, series_id: str) -> Series | None:
        series = self.get(series_id)
        if series is not None:
            series.ensure_loaded()
        return series

    @classmethod
    def open(cls, path: str | Path) -> Project:
        from nagilize.core.project_io import load_project

        return load_project(path)

    def save(self, path: str | Path | None = None, *, views: list | None = None) -> Path:
        from nagilize.core.project_io import save_project

        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("No path given and project has no path")
        save_project(self, target, views=views)
        self.path = target
        return target


def _source_label(source: str) -> str:
    if not source:
        return "(unnamed)"
    if source.startswith("dummy:"):
        return source
    return Path(source).name
