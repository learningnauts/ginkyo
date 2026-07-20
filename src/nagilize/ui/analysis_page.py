"""Analysis page: FFT spectrum settings form (M3)."""

from __future__ import annotations

from typing import Callable

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from nagilize.core.project import Project, Series
from nagilize.core.spectrum import (
    AMPLITUDE_SCALES,
    WINDOW_NAMES,
    SpectrumParams,
    compute_spectrum,
    frequency_resolution_hz,
)


class AnalysisPage(QWidget):
    """Document-style form to compute a spectrum result into the project."""

    def __init__(
        self,
        *,
        get_project: Callable[[], Project],
        on_result: Callable[[str], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._get_project = get_project
        self._on_result = on_result

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        title = QLabel("Analysis — Spectrum (FFT)")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        root.addWidget(title)
        hint = QLabel(
            "Choose a time series, set FFT options, then Run. "
            "The result is added to Project data — drag it onto a plot cell "
            "to show Magnitude (top) and Phase (bottom)."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #444;")
        root.addWidget(hint)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 8, 0)

        # --- Input ---
        input_box = QGroupBox("Input")
        input_form = QFormLayout(input_box)
        self._series_combo = QComboBox()
        self._series_combo.currentIndexChanged.connect(self._update_derived)
        input_form.addRow("Time series", self._series_combo)
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Auto from series name")
        input_form.addRow("Result name", self._name_edit)
        body_layout.addWidget(input_box)

        # --- Window / length ---
        win_box = QGroupBox("Window & length")
        win_form = QFormLayout(win_box)
        self._window_combo = QComboBox()
        for w in WINDOW_NAMES:
            self._window_combo.addItem(w, w)
        self._window_combo.setCurrentIndex(max(0, WINDOW_NAMES.index("hann")))
        self._window_combo.currentIndexChanged.connect(self._update_derived)
        win_form.addRow("Window", self._window_combo)
        self._nfft_spin = QSpinBox()
        self._nfft_spin.setRange(0, 16_777_216)
        self._nfft_spin.setSpecialValueText("Auto (signal / segment)")
        self._nfft_spin.setValue(0)
        self._nfft_spin.setSingleStep(256)
        self._nfft_spin.valueChanged.connect(self._update_derived)
        win_form.addRow("FFT length (0 = auto)", self._nfft_spin)
        body_layout.addWidget(win_box)

        # --- Averaging ---
        avg_box = QGroupBox("Averaging (Welch)")
        avg_form = QFormLayout(avg_box)
        self._avg_check = QCheckBox("Enable averaging")
        self._avg_check.toggled.connect(self._on_avg_toggled)
        avg_form.addRow(self._avg_check)
        self._overlap_spin = QDoubleSpinBox()
        self._overlap_spin.setRange(0.0, 0.95)
        self._overlap_spin.setSingleStep(0.05)
        self._overlap_spin.setValue(0.5)
        self._overlap_spin.setDecimals(2)
        self._overlap_spin.valueChanged.connect(self._update_derived)
        avg_form.addRow("Overlap", self._overlap_spin)
        self._seg_spin = QSpinBox()
        self._seg_spin.setRange(0, 16_777_216)
        self._seg_spin.setSpecialValueText("Auto (min(n, 1024))")
        self._seg_spin.setValue(0)
        self._seg_spin.setSingleStep(256)
        self._seg_spin.valueChanged.connect(self._update_derived)
        avg_form.addRow("Segment length (0 = auto)", self._seg_spin)
        body_layout.addWidget(avg_box)

        # --- Amplitude ---
        amp_box = QGroupBox("Amplitude scale")
        amp_form = QFormLayout(amp_box)
        self._scale_combo = QComboBox()
        labels = {
            "lin": "Linear (|F| / Σw)",
            "peak": "Peak (single-sided)",
            "rms": "RMS (peak / √2)",
        }
        for key in AMPLITUDE_SCALES:
            self._scale_combo.addItem(labels.get(key, key), key)
        self._scale_combo.currentIndexChanged.connect(self._update_derived)
        amp_form.addRow("Definition", self._scale_combo)
        body_layout.addWidget(amp_box)

        # --- Derived ---
        info_box = QGroupBox("Derived (from measurement + settings)")
        info_form = QFormLayout(info_box)
        self._fs_label = QLabel("—")
        self._df_label = QLabel("—")
        self._nyquist_label = QLabel("—")
        self._n_label = QLabel("—")
        info_form.addRow("Sample rate fs", self._fs_label)
        info_form.addRow("Frequency resolution Δf", self._df_label)
        info_form.addRow("Nyquist", self._nyquist_label)
        info_form.addRow("Samples (source)", self._n_label)
        body_layout.addWidget(info_box)

        body_layout.addStretch(1)
        scroll.setWidget(body)
        root.addWidget(scroll, stretch=1)

        run_row = QHBoxLayout()
        run_row.addStretch(1)
        self._run_btn = QPushButton("Run FFT")
        self._run_btn.setDefault(True)
        self._run_btn.clicked.connect(self.run_fft)
        run_row.addWidget(self._run_btn)
        root.addLayout(run_row)

        self._on_avg_toggled(False)
        self.refresh_series()

    def _on_avg_toggled(self, enabled: bool) -> None:
        self._overlap_spin.setEnabled(enabled)
        self._seg_spin.setEnabled(enabled)
        self._update_derived()

    def _selected_series(self) -> Series | None:
        sid = self._series_combo.currentData()
        if not sid:
            return None
        return self._get_project().get(str(sid))

    def refresh_series(self) -> None:
        current = self._series_combo.currentData()
        self._series_combo.blockSignals(True)
        self._series_combo.clear()
        project = self._get_project()
        for series in project.all_series():
            if series.is_spectrum():
                continue
            self._series_combo.addItem(series.display_name, series.id)
        self._series_combo.blockSignals(False)
        if current is not None:
            idx = self._series_combo.findData(current)
            if idx >= 0:
                self._series_combo.setCurrentIndex(idx)
        self._update_derived()

    def _current_params(self, series: Series | None) -> SpectrumParams:
        nfft = int(self._nfft_spin.value())
        seg = int(self._seg_spin.value())
        averaging = self._avg_check.isChecked()
        return SpectrumParams(
            window=str(self._window_combo.currentData() or "hann"),
            nfft=None if nfft <= 0 else nfft,
            averaging=averaging,
            overlap=float(self._overlap_spin.value()),
            segment_length=None if (not averaging or seg <= 0) else seg,
            amplitude_scale=str(self._scale_combo.currentData() or "lin"),
        )

    def _update_derived(self) -> None:
        series = self._selected_series()
        if series is None:
            self._fs_label.setText("—")
            self._df_label.setText("—")
            self._nyquist_label.setText("—")
            self._n_label.setText("—")
            return
        fs = float(series.sample_rate)
        n = int(series.n_samples)
        self._fs_label.setText(f"{fs:.6g} Hz")
        self._nyquist_label.setText(f"{fs / 2.0:.6g} Hz")
        self._n_label.setText(str(n))
        params = self._current_params(series)
        if params.averaging:
            seg = params.segment_length or min(n, 1024)
            seg = max(2, min(seg, n if n > 0 else seg))
            nfft = params.nfft or seg
            nfft = max(seg, nfft)
        else:
            nfft = params.nfft or max(n, 2)
            nfft = max(n if n > 0 else 2, nfft)
        df = frequency_resolution_hz(fs, nfft)
        self._df_label.setText(f"{df:.6g} Hz  (fs / NFFT, NFFT={nfft})")
        if not self._name_edit.text().strip():
            pass  # keep placeholder; fill on run if empty

    def run_fft(self) -> None:
        series = self._selected_series()
        if series is None:
            QMessageBox.warning(self, "FFT", "Select a time series first.")
            return
        params = self._current_params(series)
        try:
            spec = compute_spectrum(series.data, float(series.sample_rate), params)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "FFT failed", str(exc))
            return
        if spec.frequency_hz.size == 0:
            QMessageBox.warning(self, "FFT", "Spectrum is empty (signal too short).")
            return
        name = self._name_edit.text().strip()
        if not name:
            name = f"{series.name} · Spectrum"
        project = self._get_project()
        sid = project.add_spectrum_result(
            name=name,
            frequency_hz=spec.frequency_hz,
            magnitude=spec.magnitude,
            phase_rad=spec.phase_rad,
            sample_rate=float(series.sample_rate),
            unit=series.unit,
            parent=series,
            params_dict={
                **params.to_dict(),
                "delta_f_hz": spec.delta_f_hz,
                "n_segments": spec.n_segments,
            },
        )
        self._on_result(sid)
        self.status_message = f"Added spectrum: {name}"
