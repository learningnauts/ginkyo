"""Project: multiple recordings expanded into a shared series pool."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from uuid import uuid4

import numpy as np

from ginkyo.core.model import Recording, SeriesMeta


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
    _frame_time: np.ndarray | None = field(default=None, repr=False, compare=False)
    _frame_angle: np.ndarray | None = field(default=None, repr=False, compare=False)
    _frame_rpm: np.ndarray | None = field(default=None, repr=False, compare=False)
    _loader: Callable[
        [],
        tuple[
            np.ndarray,
            np.ndarray | None,
            np.ndarray | None,
            np.ndarray | None,
            np.ndarray | None,
            np.ndarray | None,
        ],
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
        frame_time: np.ndarray | None = None,
        frame_angle: np.ndarray | None = None,
        frame_rpm: np.ndarray | None = None,
        meta: SeriesMeta | None = None,
    ) -> Series:
        arr = np.asarray(data, dtype=np.float64)
        n_hint = int(arr.shape[-1]) if arr.ndim == 2 else int(arr.shape[0])
        return cls(
            id=id,
            name=name,
            sample_rate=float(sample_rate),
            unit=unit,
            source_id=source_id,
            source_label=source_label,
            channel_index=channel_index,
            meta=meta or SeriesMeta(),
            n_samples_hint=n_hint,
            _data=arr,
            _time=None if time is None else np.asarray(time, dtype=float),
            _phase=None if phase is None else np.asarray(phase, dtype=np.float64),
            _frame_time=None
            if frame_time is None
            else np.asarray(frame_time, dtype=float),
            _frame_angle=None
            if frame_angle is None
            else np.asarray(frame_angle, dtype=float),
            _frame_rpm=None
            if frame_rpm is None
            else np.asarray(frame_rpm, dtype=float),
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
        loader: Callable[
            [],
            tuple[
                np.ndarray,
                np.ndarray | None,
                np.ndarray | None,
                np.ndarray | None,
                np.ndarray | None,
                np.ndarray | None,
            ],
        ],
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
            _frame_time=None,
            _frame_angle=None,
            _frame_rpm=None,
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
            self._frame_time = None
            self._frame_angle = None
            self._frame_rpm = None
            return
        loaded = self._loader()
        if len(loaded) == 2:
            data, time = loaded
            phase = None
            frame_time = None
            frame_angle = None
            frame_rpm = None
        elif len(loaded) == 3:
            data, time, phase = loaded
            frame_time = None
            frame_angle = None
            frame_rpm = None
        elif len(loaded) == 4:
            data, time, phase, frame_time = loaded
            frame_angle = None
            frame_rpm = None
        elif len(loaded) == 5:
            data, time, phase, frame_time, frame_angle = loaded
            frame_rpm = None
        else:
            data, time, phase, frame_time, frame_angle, frame_rpm = loaded
        self._data = np.asarray(data, dtype=np.float64)
        self._time = None if time is None else np.asarray(time, dtype=float)
        self._phase = None if phase is None else np.asarray(phase, dtype=np.float64)
        self._frame_time = (
            None if frame_time is None else np.asarray(frame_time, dtype=float)
        )
        self._frame_angle = (
            None if frame_angle is None else np.asarray(frame_angle, dtype=float)
        )
        self._frame_rpm = (
            None if frame_rpm is None else np.asarray(frame_rpm, dtype=float)
        )
        if self._data.ndim == 2:
            self.n_samples_hint = int(self._data.shape[-1])
        else:
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

    @property
    def frame_time_s(self) -> np.ndarray | None:
        """Frame center times for spectrogram results; None for 1D series."""
        self.ensure_loaded()
        return self._frame_time

    @frame_time_s.setter
    def frame_time_s(self, value: np.ndarray | None) -> None:
        self.ensure_loaded()
        self._frame_time = None if value is None else np.asarray(value, dtype=float)

    @property
    def frame_angle_deg(self) -> np.ndarray | None:
        """Frame center angles (deg) for equal-angle spectrograms; else None."""
        self.ensure_loaded()
        return self._frame_angle

    @frame_angle_deg.setter
    def frame_angle_deg(self, value: np.ndarray | None) -> None:
        self.ensure_loaded()
        self._frame_angle = None if value is None else np.asarray(value, dtype=float)

    @property
    def frame_rpm(self) -> np.ndarray | None:
        """Frame center RPM for equal-RPM spectrograms; else None."""
        self.ensure_loaded()
        return self._frame_rpm

    @frame_rpm.setter
    def frame_rpm(self, value: np.ndarray | None) -> None:
        self.ensure_loaded()
        self._frame_rpm = None if value is None else np.asarray(value, dtype=float)

    def is_spectrum(self) -> bool:
        return (self.meta.quantity or "").strip().lower() == "spectrum"

    def is_spectrogram(self) -> bool:
        return (self.meta.quantity or "").strip().lower() in ("spectrogram", "stft")

    def is_fft_result(self) -> bool:
        return self.is_spectrum() or self.is_spectrogram()

    @property
    def n_samples(self) -> int:
        if self._data is not None:
            if self._data.ndim == 2:
                return int(self._data.shape[-1])
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

    def create_analysis_dataset(
        self, *, label: str | None = None, provenance: str = "analysis:fft"
    ) -> SourceRef:
        """Create a new Analysis dataset (one per FFT run)."""
        source_id = f"analysis:{uuid4().hex[:8]}"
        n = 1 + sum(1 for s in self.sources if str(s.id).startswith("analysis"))
        src = SourceRef(
            id=source_id,
            label=(label or f"FFT result {n}").strip() or f"FFT result {n}",
            recording=None,
            provenance=provenance,
        )
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
        source: SourceRef | None = None,
    ) -> str:
        """Add one spectrum result into an Analysis dataset. Returns series id.

        If ``source`` is omitted, a new dataset is created (single-result run).
        For batch FFT, create one dataset with ``create_analysis_dataset`` and
        pass it to every call in that run.
        """
        dataset = source if source is not None else self.create_analysis_dataset()
        sid = f"{dataset.id}:ch{uuid4().hex[:8]}"
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
        # Channel index = order within this dataset so far.
        ch_index = sum(1 for s in self.series.values() if s.source_id == dataset.id)
        series = Series.create(
            id=sid,
            name=name,
            sample_rate=float(sample_rate),
            data=mag,
            unit=unit,
            source_id=dataset.id,
            source_label=dataset.label,
            channel_index=ch_index,
            time=freq,
            phase=phase,
            meta=meta,
        )
        self.series[sid] = series
        self.series_order.append(sid)
        return sid

    def add_spectrogram_result(
        self,
        *,
        name: str,
        frequency_hz: np.ndarray,
        time_s: np.ndarray,
        magnitude: np.ndarray,
        phase_rad: np.ndarray,
        sample_rate: float,
        unit: str = "",
        parent: Series | None = None,
        params_dict: dict | None = None,
        attrs: dict | None = None,
        source: SourceRef | None = None,
        angle_deg: np.ndarray | None = None,
        rpm: np.ndarray | None = None,
    ) -> str:
        """Add one STFT spectrogram result into an Analysis dataset."""
        dataset = source if source is not None else self.create_analysis_dataset(
            provenance="analysis:stft"
        )
        sid = f"{dataset.id}:ch{uuid4().hex[:8]}"
        freq = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
        times = np.asarray(time_s, dtype=np.float64).reshape(-1)
        mag = np.asarray(magnitude, dtype=np.float64)
        phase = np.asarray(phase_rad, dtype=np.float64)
        angles = (
            None
            if angle_deg is None
            else np.asarray(angle_deg, dtype=np.float64).reshape(-1)
        )
        rpms = (
            None if rpm is None else np.asarray(rpm, dtype=np.float64).reshape(-1)
        )
        if mag.ndim != 2 or phase.ndim != 2:
            raise ValueError("magnitude and phase must be 2-D arrays")
        if mag.shape != phase.shape:
            raise ValueError("magnitude and phase must have the same shape")
        if mag.shape[0] != freq.size:
            raise ValueError("frequency axis must match magnitude rows")
        if mag.shape[1] != times.size:
            raise ValueError("time axis must match magnitude columns")
        if angles is not None and angles.size != times.size:
            raise ValueError("angle axis must match time axis length")
        if rpms is not None and rpms.size != times.size:
            raise ValueError("rpm axis must match time axis length")
        meta_attrs = dict(attrs or {})
        if params_dict:
            meta_attrs["stft_params"] = dict(params_dict)
        if parent is not None:
            meta_attrs["parent_series_id"] = parent.id
        meta = SeriesMeta(
            quantity="spectrogram",
            point_id=parent.meta.point_id if parent else "",
            point_name=parent.meta.point_name if parent else "",
            dof=parent.meta.dof if parent else "",
            ref_point_id=parent.meta.ref_point_id if parent else "",
            ref_point_name=parent.meta.ref_point_name if parent else "",
            ref_dof=parent.meta.ref_dof if parent else "",
            provenance=(
                f"STFT of {parent.display_name}" if parent is not None else "STFT"
            ),
            attrs=meta_attrs,
        )
        ch_index = sum(1 for s in self.series.values() if s.source_id == dataset.id)
        series = Series.create(
            id=sid,
            name=name,
            sample_rate=float(sample_rate),
            data=mag,
            unit=unit,
            source_id=dataset.id,
            source_label=dataset.label,
            channel_index=ch_index,
            time=freq,
            phase=phase,
            frame_time=times,
            frame_angle=angles,
            frame_rpm=rpms,
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
        from ginkyo.core.project_io import load_project

        return load_project(path)

    def save(self, path: str | Path | None = None, *, views: list | None = None) -> Path:
        from ginkyo.core.project_io import save_project

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
