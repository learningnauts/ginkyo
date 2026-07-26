"""Analysis page: FFT spectrum settings form (M3)."""

from __future__ import annotations

import json
from typing import Callable

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ginkyo.core.project import Project, Series
from ginkyo.core.spectrum import (
    AMPLITUDE_SCALES,
    WINDOW_NAMES,
    SpectrumParams,
    angle_from_pulse,
    angle_from_rpm,
    compute_spectrum,
    compute_stft,
    equal_angle_frame_starts,
    equal_rpm_frame_starts,
    frequency_resolution_hz,
    max_averages,
    resolve_nfft,
    rpm_from_pulse,
    stft_frame_count,
    stft_hop_samples,
    stft_overlap_equiv,
)
from ginkyo.ui.panel_shell import SERIES_MIME

DEFAULT_RESULT_SUFFIX = "· Spectrum"
DEFAULT_STFT_RESULT_SUFFIX = "· Spectrogram"
DEFAULT_DELTA_F_HZ = 1.0
WELCH_SPIN_WIDTH = 88
MODE_STATIC = "static"
MODE_STFT = "stft"
STFT_STEP_OVERLAP = "overlap"
STFT_STEP_DT = "fixed_dt"
STFT_STEP_ANGLE = "equal_angle"
STFT_STEP_RPM = "equal_rpm"
TACHO_KIND_PULSE = "pulse"
TACHO_KIND_RPM = "rpm"

# Units that strongly suggest an RPM (speed) channel rather than a pulse train.
_RPM_UNIT_TOKENS = frozenset(
    {
        "rpm",
        "r/min",
        "rev/min",
        "revs/min",
        "1/min",
        "min^-1",
        "min-1",
    }
)


class AnalysisPage(QWidget):
    """Document-style form: build a selection, then run FFT on it."""

    def __init__(
        self,
        *,
        get_project: Callable[[], Project],
        get_project_selection: Callable[[], list[str]],
        on_result: Callable[[str], None] | Callable[[list[str]], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._get_project = get_project
        self._get_project_selection = get_project_selection
        self._on_result = on_result
        self._picked_ids: list[str] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        title = QLabel("Analysis — Spectrum (FFT)")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        root.addWidget(title)
        hint = QLabel(
            "In Project data (left), select series → Add. "
            "Remove mistakes from this list, then Run. "
            "You can also drag series from the left tree onto the list."
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

        # --- Selection set ---
        input_box = QGroupBox("Data to analyze")
        input_layout = QVBoxLayout(input_box)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add from Project selection")
        add_btn.setToolTip("Add series currently selected in Project data (left)")
        add_btn.clicked.connect(self.add_from_project_selection)
        remove_btn = QPushButton("Remove selected")
        remove_btn.clicked.connect(self.remove_selected_from_pick)
        clear_btn = QPushButton("Clear list")
        clear_btn.clicked.connect(self.clear_pick)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(remove_btn)
        btn_row.addWidget(clear_btn)
        btn_row.addStretch(1)
        input_layout.addLayout(btn_row)

        self._series_list = QListWidget()
        self._series_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._series_list.setMinimumHeight(140)
        self._series_list.setAcceptDrops(True)
        self._series_list.viewport().setAcceptDrops(True)
        self._series_list.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        self._series_list.setDefaultDropAction(Qt.DropAction.CopyAction)
        # Hook drop handling on the list widget.
        self._series_list.dragEnterEvent = self._list_drag_enter  # type: ignore[method-assign]
        self._series_list.dragMoveEvent = self._list_drag_move  # type: ignore[method-assign]
        self._series_list.dropEvent = self._list_drop  # type: ignore[method-assign]
        input_layout.addWidget(self._series_list)

        self._selection_label = QLabel("0 series in analysis set")
        self._selection_label.setStyleSheet("color: #444;")
        input_layout.addWidget(self._selection_label)
        self._selection_error_label = QLabel()
        self._selection_error_label.setWordWrap(True)
        self._selection_error_label.setStyleSheet("color: #a40000;")
        self._selection_error_label.hide()
        input_layout.addWidget(self._selection_error_label)
        body_layout.addWidget(input_box)

        mode_box = QGroupBox("Analysis mode")
        mode_layout = QHBoxLayout(mode_box)
        self._mode_group = QButtonGroup(self)
        self._mode_static_btn = QRadioButton("Static FFT")
        self._mode_static_btn.setToolTip(
            "One spectrum per channel (optional Welch averaging over time)"
        )
        self._mode_stft_btn = QRadioButton("Short-time FFT (STFT)")
        self._mode_stft_btn.setToolTip(
            "Time–frequency spectrogram per channel"
        )
        self._mode_static_btn.setChecked(True)
        self._mode_group.addButton(self._mode_static_btn, 0)
        self._mode_group.addButton(self._mode_stft_btn, 1)
        self._mode_group.idClicked.connect(self._on_mode_changed)
        mode_layout.addWidget(self._mode_static_btn)
        mode_layout.addWidget(self._mode_stft_btn)
        mode_layout.addStretch(1)
        body_layout.addWidget(mode_box)

        # --- Window / length ---
        win_box = QGroupBox("Window & length")
        win_form = QFormLayout(win_box)
        self._configure_form(win_form)
        self._window_combo = QComboBox()
        default_window = SpectrumParams().window
        for w in WINDOW_NAMES:
            label = f"{w} (default)" if w == default_window else w
            self._window_combo.addItem(label, w)
        self._window_combo.setCurrentIndex(max(0, WINDOW_NAMES.index(default_window)))
        self._window_combo.currentIndexChanged.connect(self._update_derived)
        win_form.addRow("Window", self._window_combo)
        self._df_spin = QDoubleSpinBox()
        self._df_spin.setRange(0.0, 1.0e9)
        self._df_spin.setSpecialValueText("Auto")
        self._df_spin.setToolTip("0 = auto")
        self._df_spin.setValue(DEFAULT_DELTA_F_HZ)
        self._df_spin.setDecimals(6)
        self._df_spin.setSuffix(" Hz")
        self._df_spin.setMaximumWidth(220)
        self._df_spin.valueChanged.connect(self._update_derived)
        self._df_spin.editingFinished.connect(self._update_derived)
        df_cell = QWidget()
        df_cell.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        df_layout = QVBoxLayout(df_cell)
        df_layout.setContentsMargins(0, 0, 0, 0)
        df_layout.setSpacing(4)
        df_row = QHBoxLayout()
        df_row.setContentsMargins(0, 0, 0, 0)
        df_row.addWidget(self._df_spin)
        self._df_min_label = QLabel()
        self._df_min_label.setStyleSheet("color: #666;")
        df_row.addWidget(self._df_min_label)
        df_row.addStretch(1)
        df_layout.addLayout(df_row)
        self._nfft_hint_label = self._hint_label()
        df_layout.addWidget(self._nfft_hint_label)
        df_label = QLabel("Δf (frequency resolution)")
        df_label.setToolTip("Target bin spacing: Δf = fs / NFFT")
        win_form.addRow(df_label, df_cell)
        body_layout.addWidget(win_box)

        # --- Static: Welch averaging ---
        self._static_avg_box = QGroupBox("Averaging (Welch)")
        avg_form = QFormLayout(self._static_avg_box)
        self._configure_form(avg_form)
        self._avg_check = QCheckBox("Enable averaging")
        self._avg_check.toggled.connect(self._on_avg_toggled)
        avg_form.addRow(self._avg_check)
        self._welch_overlap_spin = QSpinBox()
        self._welch_overlap_spin.setRange(0, 95)
        self._welch_overlap_spin.setSingleStep(5)
        self._welch_overlap_spin.setValue(50)
        self._welch_overlap_spin.setSuffix(" %")
        self._welch_overlap_spin.setToolTip("Overlap between consecutive Welch segments")
        self._configure_welch_spin(self._welch_overlap_spin)
        self._welch_overlap_spin.valueChanged.connect(self._update_derived)
        welch_overlap_cell = QWidget()
        welch_overlap_layout = QHBoxLayout(welch_overlap_cell)
        welch_overlap_layout.setContentsMargins(0, 0, 0, 0)
        welch_overlap_layout.addWidget(self._welch_overlap_spin)
        welch_overlap_layout.addStretch(1)
        avg_form.addRow("Overlap", welch_overlap_cell)
        self._avg_count_spin = QSpinBox()
        self._avg_count_spin.setRange(0, 1_000_000)
        self._avg_count_spin.setSpecialValueText("Auto")
        self._avg_count_spin.setValue(0)
        self._avg_count_spin.setToolTip(
            "0 = Auto (use all segments that fit). "
            "Values above the maximum are clamped to max."
        )
        self._configure_welch_spin(self._avg_count_spin)
        self._avg_count_spin.valueChanged.connect(self._update_derived)
        avg_count_cell = QWidget()
        avg_count_cell.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        avg_count_layout = QHBoxLayout(avg_count_cell)
        avg_count_layout.setContentsMargins(0, 0, 0, 0)
        avg_count_layout.addWidget(self._avg_count_spin)
        self._avg_max_label = QLabel()
        self._avg_max_label.setStyleSheet("color: #666;")
        avg_count_layout.addWidget(self._avg_max_label)
        avg_count_layout.addStretch(1)
        avg_form.addRow("Averages", avg_count_cell)
        body_layout.addWidget(self._static_avg_box)

        # --- STFT: time stepping ---
        self._stft_box = QGroupBox("Time stepping")
        self._stft_form = QFormLayout(self._stft_box)
        self._configure_form(self._stft_form)
        step_row = QWidget()
        step_layout = QHBoxLayout(step_row)
        step_layout.setContentsMargins(0, 0, 0, 0)
        self._stft_step_group = QButtonGroup(self)
        self._stft_step_overlap_btn = QRadioButton("Overlap %")
        self._stft_step_dt_btn = QRadioButton("Fixed Δt")
        self._stft_step_angle_btn = QRadioButton("Equal angle")
        self._stft_step_rpm_btn = QRadioButton("Equal RPM")
        self._stft_step_overlap_btn.setChecked(True)
        self._stft_step_group.addButton(self._stft_step_overlap_btn, 0)
        self._stft_step_group.addButton(self._stft_step_dt_btn, 1)
        self._stft_step_group.addButton(self._stft_step_angle_btn, 2)
        self._stft_step_group.addButton(self._stft_step_rpm_btn, 3)
        self._stft_step_group.idClicked.connect(self._on_stft_step_changed)
        step_layout.addWidget(self._stft_step_overlap_btn)
        step_layout.addWidget(self._stft_step_dt_btn)
        step_layout.addWidget(self._stft_step_angle_btn)
        step_layout.addWidget(self._stft_step_rpm_btn)
        step_layout.addStretch(1)
        self._stft_form.addRow("Stepping", step_row)
        self._stft_overlap_row = QWidget()
        stft_overlap_form = QHBoxLayout(self._stft_overlap_row)
        stft_overlap_form.setContentsMargins(0, 0, 0, 0)
        self._stft_overlap_spin = QSpinBox()
        self._stft_overlap_spin.setRange(0, 95)
        self._stft_overlap_spin.setSingleStep(5)
        self._stft_overlap_spin.setValue(50)
        self._stft_overlap_spin.setSuffix(" %")
        self._stft_overlap_spin.setToolTip("Overlap between consecutive STFT frames")
        self._configure_welch_spin(self._stft_overlap_spin)
        self._stft_overlap_spin.valueChanged.connect(self._update_derived)
        stft_overlap_form.addWidget(self._stft_overlap_spin)
        stft_overlap_form.addStretch(1)
        self._stft_form.addRow("Overlap", self._stft_overlap_row)
        self._stft_dt_row = QWidget()
        stft_dt_form = QHBoxLayout(self._stft_dt_row)
        stft_dt_form.setContentsMargins(0, 0, 0, 0)
        self._stft_step_dt_spin = QDoubleSpinBox()
        self._stft_step_dt_spin.setRange(0.0, 1_000_000.0)
        self._stft_step_dt_spin.setDecimals(6)
        self._stft_step_dt_spin.setSingleStep(0.001)
        self._stft_step_dt_spin.setValue(0.01)
        self._stft_step_dt_spin.setSuffix(" s")
        self._stft_step_dt_spin.setToolTip("Time shift between consecutive STFT frames")
        self._configure_welch_spin(self._stft_step_dt_spin)
        self._stft_step_dt_spin.valueChanged.connect(self._update_derived)
        stft_dt_form.addWidget(self._stft_step_dt_spin)
        stft_dt_form.addStretch(1)
        self._stft_form.addRow("Step", self._stft_dt_row)

        self._stft_tacho_kind_combo = QComboBox()
        self._stft_tacho_kind_combo.addItem("Pulse", TACHO_KIND_PULSE)
        self._stft_tacho_kind_combo.addItem("RPM", TACHO_KIND_RPM)
        self._stft_tacho_kind_combo.setToolTip(
            "Equal RPM only: Pulse or RPM. Choosing a kind picks the "
            "best-matching series from Data to analyze (by unit / name). "
            "Equal angle always uses a pulse tacho."
        )
        self._stft_tacho_kind_combo.currentIndexChanged.connect(self._on_tacho_kind_changed)
        self._stft_tacho_kind_row = QWidget()
        kind_row_layout = QHBoxLayout(self._stft_tacho_kind_row)
        kind_row_layout.setContentsMargins(0, 0, 0, 0)
        kind_row_layout.addWidget(self._stft_tacho_kind_combo)
        kind_row_layout.addStretch(1)
        self._stft_form.addRow("Tacho kind", self._stft_tacho_kind_row)

        self._stft_tacho_combo = QComboBox()
        self._stft_tacho_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._stft_tacho_combo.setToolTip(
            "Tacho channel from Data to analyze. "
            "Equal angle: pulse train. Equal RPM: filled from Tacho kind."
        )
        self._stft_tacho_combo.currentIndexChanged.connect(self._on_tacho_series_changed)
        self._stft_tacho_row = QWidget()
        tacho_row_layout = QHBoxLayout(self._stft_tacho_row)
        tacho_row_layout.setContentsMargins(0, 0, 0, 0)
        tacho_row_layout.addWidget(self._stft_tacho_combo)
        self._stft_form.addRow("Tacho series", self._stft_tacho_row)

        self._stft_dtheta_spin = QDoubleSpinBox()
        self._stft_dtheta_spin.setRange(0.1, 3600.0)
        self._stft_dtheta_spin.setDecimals(2)
        self._stft_dtheta_spin.setSingleStep(1.0)
        self._stft_dtheta_spin.setValue(10.0)
        self._stft_dtheta_spin.setSuffix(" deg")
        self._stft_dtheta_spin.setToolTip("Angular step between consecutive STFT frames")
        self._configure_welch_spin(self._stft_dtheta_spin)
        self._stft_dtheta_spin.valueChanged.connect(self._update_derived)
        self._stft_dtheta_row = QWidget()
        dtheta_layout = QHBoxLayout(self._stft_dtheta_row)
        dtheta_layout.setContentsMargins(0, 0, 0, 0)
        dtheta_layout.addWidget(self._stft_dtheta_spin)
        dtheta_layout.addStretch(1)
        self._stft_form.addRow("Δθ", self._stft_dtheta_row)

        self._stft_drpm_spin = QDoubleSpinBox()
        self._stft_drpm_spin.setRange(0.1, 10_000.0)
        self._stft_drpm_spin.setDecimals(2)
        self._stft_drpm_spin.setSingleStep(1.0)
        self._stft_drpm_spin.setValue(10.0)
        self._stft_drpm_spin.setSuffix(" RPM")
        self._stft_drpm_spin.setToolTip("RPM step between consecutive STFT frames")
        self._configure_welch_spin(self._stft_drpm_spin)
        self._stft_drpm_spin.valueChanged.connect(self._update_derived)
        self._stft_drpm_row = QWidget()
        drpm_layout = QHBoxLayout(self._stft_drpm_row)
        drpm_layout.setContentsMargins(0, 0, 0, 0)
        drpm_layout.addWidget(self._stft_drpm_spin)
        drpm_layout.addStretch(1)
        self._stft_form.addRow("ΔRPM", self._stft_drpm_row)

        self._stft_ppr_spin = QSpinBox()
        self._stft_ppr_spin.setRange(1, 1024)
        self._stft_ppr_spin.setValue(1)
        self._stft_ppr_spin.setToolTip("Pulses per shaft revolution")
        self._configure_welch_spin(self._stft_ppr_spin)
        self._stft_ppr_spin.valueChanged.connect(self._update_derived)
        self._stft_ppr_row = QWidget()
        ppr_layout = QHBoxLayout(self._stft_ppr_row)
        ppr_layout.setContentsMargins(0, 0, 0, 0)
        ppr_layout.addWidget(self._stft_ppr_spin)
        ppr_layout.addStretch(1)
        self._stft_form.addRow("Pulses / rev", self._stft_ppr_row)

        self._stft_frames_max_label = QLabel("(max: —)")
        self._stft_frames_max_label.setStyleSheet("color: #666;")
        stft_frames_cell = QWidget()
        stft_frames_layout = QHBoxLayout(stft_frames_cell)
        stft_frames_layout.setContentsMargins(0, 0, 0, 0)
        stft_frames_layout.addWidget(self._stft_frames_max_label)
        stft_frames_layout.addStretch(1)
        self._stft_form.addRow("Frames", stft_frames_cell)
        self._stft_step_hint_label = self._hint_label()
        self._stft_form.addRow("", self._stft_step_hint_label)
        body_layout.addWidget(self._stft_box)

        # --- Amplitude ---
        amp_box = QGroupBox("Amplitude scale")
        amp_form = QFormLayout(amp_box)
        self._configure_form(amp_form)
        self._scale_combo = QComboBox()
        scale_labels = {
            "lin": "Linear",
            "peak": "Peak",
            "rms": "RMS",
            "ptp": "Peak-to-peak",
        }
        default_scale = SpectrumParams().amplitude_scale
        for key in AMPLITUDE_SCALES:
            label = scale_labels.get(key, key)
            if key == default_scale:
                label = f"{label} (default)"
            self._scale_combo.addItem(label, key)
        self._scale_combo.setCurrentIndex(AMPLITUDE_SCALES.index(default_scale))
        self._scale_combo.currentIndexChanged.connect(self._update_derived)
        amp_form.addRow("Definition", self._scale_combo)
        body_layout.addWidget(amp_box)

        name_box = QGroupBox("Output names")
        name_layout = QVBoxLayout(name_box)
        name_layout.setSpacing(8)
        dataset_label = QLabel("Dataset name")
        name_layout.addWidget(dataset_label)
        self._dataset_name_edit = QLineEdit()
        self._configure_name_field(self._dataset_name_edit)
        name_layout.addWidget(self._dataset_name_edit)
        suffix_label = QLabel("Result name suffix")
        name_layout.addWidget(suffix_label)
        self._name_edit = QLineEdit()
        self._configure_name_field(self._name_edit)
        self._name_edit.setPlaceholderText(DEFAULT_RESULT_SUFFIX)
        self._name_edit.textChanged.connect(self._update_suffix_preview)
        name_layout.addWidget(self._name_edit)
        self._suffix_preview = self._hint_label()
        name_layout.addWidget(self._suffix_preview)
        body_layout.addWidget(name_box)

        body_layout.addStretch(1)
        scroll.setWidget(body)
        root.addWidget(scroll, stretch=1)

        run_row = QHBoxLayout()
        self._run_btn = QPushButton("Run FFT")
        self._run_btn.setDefault(True)
        self._run_btn.clicked.connect(self.run_fft)
        run_row.addWidget(self._run_btn)
        run_row.addStretch(1)
        root.addLayout(run_row)

        self._on_avg_toggled(False)
        self._on_mode_changed(0)
        self._on_stft_step_changed(0)
        self._rebuild_list()
        self._update_suffix_preview()

    def _analysis_mode(self) -> str:
        return MODE_STFT if self._mode_stft_btn.isChecked() else MODE_STATIC

    def _is_stft_mode(self) -> bool:
        return self._analysis_mode() == MODE_STFT

    def _stft_step_mode(self) -> str:
        if self._stft_step_rpm_btn.isChecked():
            return STFT_STEP_RPM
        if self._stft_step_angle_btn.isChecked():
            return STFT_STEP_ANGLE
        if self._stft_step_dt_btn.isChecked():
            return STFT_STEP_DT
        return STFT_STEP_OVERLAP

    def _tacho_kind(self) -> str:
        return str(self._stft_tacho_kind_combo.currentData() or TACHO_KIND_PULSE)

    def _uses_tacho_stepping(self) -> bool:
        return self._stft_step_mode() in (STFT_STEP_ANGLE, STFT_STEP_RPM)

    def _set_form_row_visible(self, field: QWidget, visible: bool) -> None:
        """Show/hide a QFormLayout row including its left-hand label."""
        field.setVisible(visible)
        form = getattr(self, "_stft_form", None)
        if form is None:
            return
        label = form.labelForField(field)
        if label is not None:
            label.setVisible(visible)

    def _on_tacho_kind_changed(self, _index: int) -> None:
        # Kind is only meaningful for Equal RPM; Equal angle is always pulse.
        pulse = self._tacho_kind() == TACHO_KIND_PULSE
        self._set_form_row_visible(
            self._stft_ppr_row,
            self._uses_tacho_stepping() and pulse,
        )
        self._apply_tacho_series_suggestion()
        self._update_derived()

    def _force_tacho_kind(self, kind: str) -> None:
        idx = self._stft_tacho_kind_combo.findData(kind)
        if idx < 0 or self._stft_tacho_kind_combo.currentIndex() == idx:
            return
        self._stft_tacho_kind_combo.blockSignals(True)
        self._stft_tacho_kind_combo.setCurrentIndex(idx)
        self._stft_tacho_kind_combo.blockSignals(False)

    def _on_stft_step_changed(self, _index: int) -> None:
        mode = self._stft_step_mode()
        uses_tacho = mode in (STFT_STEP_ANGLE, STFT_STEP_RPM)
        equal_angle = mode == STFT_STEP_ANGLE
        equal_rpm = mode == STFT_STEP_RPM
        self._set_form_row_visible(self._stft_overlap_row, mode == STFT_STEP_OVERLAP)
        self._set_form_row_visible(self._stft_dt_row, mode == STFT_STEP_DT)
        # Equal angle: pulse only → no Kind picker. Equal RPM: Pulse/RPM Kind.
        self._set_form_row_visible(self._stft_tacho_kind_row, equal_rpm)
        self._set_form_row_visible(self._stft_tacho_row, uses_tacho)
        self._set_form_row_visible(self._stft_dtheta_row, equal_angle)
        self._set_form_row_visible(self._stft_drpm_row, equal_rpm)
        if equal_angle:
            self._force_tacho_kind(TACHO_KIND_PULSE)
        self._set_form_row_visible(
            self._stft_ppr_row,
            uses_tacho and self._tacho_kind() == TACHO_KIND_PULSE,
        )
        if uses_tacho:
            self._apply_tacho_series_suggestion()
        self._update_derived()

    def _selected_tacho_series(self) -> Series | None:
        sid = self._stft_tacho_combo.currentData()
        if not sid:
            return None
        project = self._get_project()
        return project.get(str(sid))

    @staticmethod
    def _normalize_unit(unit: str) -> str:
        s = (unit or "").strip().lower().replace(" ", "")
        s = s.replace("／", "/").replace("⁻", "-")
        return s

    @classmethod
    def _tacho_kind_from_unit(cls, unit: str) -> str | None:
        """Return suggested kind from unit, or None if inconclusive."""
        token = cls._normalize_unit(unit)
        if not token:
            return None
        if token in _RPM_UNIT_TOKENS or "rpm" in token:
            return TACHO_KIND_RPM
        return None

    @classmethod
    def _tacho_kind_from_name(cls, name: str) -> str | None:
        """Return suggested kind from channel name, or None if inconclusive."""
        n = (name or "").strip().lower()
        if not n:
            return None
        if "rpm" in n or "speed" in n:
            return TACHO_KIND_RPM
        if "pulse" in n or "tacho" in n or "tach" in n or "keyphasor" in n:
            return TACHO_KIND_PULSE
        return None

    @classmethod
    def _tacho_match_score(cls, series: Series, kind: str) -> int:
        """Higher = better match for the requested tacho kind."""
        unit_kind = cls._tacho_kind_from_unit(series.unit)
        name_kind = cls._tacho_kind_from_name(series.name)
        if kind == TACHO_KIND_RPM:
            score = 0
            if unit_kind == TACHO_KIND_RPM:
                score += 100
            if name_kind == TACHO_KIND_RPM:
                score += 80
            if name_kind == TACHO_KIND_PULSE:
                score -= 50
            return score
        score = 0
        if name_kind == TACHO_KIND_PULSE:
            score += 100
        if unit_kind == TACHO_KIND_RPM or name_kind == TACHO_KIND_RPM:
            score -= 50
        return score

    def _best_tacho_series_for_kind(self, kind: str) -> Series | None:
        project = self._get_project()
        best: Series | None = None
        best_score = 0
        for sid in self._picked_ids:
            series = project.get(sid)
            if series is None or series.is_fft_result():
                continue
            score = self._tacho_match_score(series, kind)
            if score > best_score:
                best_score = score
                best = series
        return best if best_score > 0 else None

    def _apply_tacho_series_suggestion(self) -> None:
        """Pick the likeliest tacho series for the current Tacho kind."""
        kind = self._tacho_kind()
        current = self._selected_tacho_series()
        if current is not None and self._tacho_match_score(current, kind) > 0:
            return
        best = self._best_tacho_series_for_kind(kind)
        if best is None:
            return
        idx = self._stft_tacho_combo.findData(best.id)
        if idx < 0 or self._stft_tacho_combo.currentIndex() == idx:
            return
        self._stft_tacho_combo.blockSignals(True)
        self._stft_tacho_combo.setCurrentIndex(idx)
        self._stft_tacho_combo.blockSignals(False)

    def _on_tacho_series_changed(self, _index: int) -> None:
        self._update_derived()

    def _prepare_tacho_arrays(
        self, tacho: Series, *, n_signal: int, fs: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(theta_deg, rpm)`` aligned to the signal length."""
        y = np.asarray(tacho.data, dtype=np.float64).ravel()
        tacho_fs = float(tacho.sample_rate)
        if abs(tacho_fs - fs) > 1e-9 * max(fs, 1.0):
            raise ValueError(
                f"tacho sample rate {tacho_fs:g} Hz ≠ signal {fs:g} Hz"
            )
        if y.size < n_signal:
            raise ValueError(
                f"tacho length {y.size} < signal length {n_signal}"
            )
        y = y[:n_signal]
        ppr = int(self._stft_ppr_spin.value())
        if self._tacho_kind() == TACHO_KIND_RPM:
            rpm = y.astype(np.float64, copy=False)
            theta = angle_from_rpm(rpm, fs)
            return theta, rpm
        theta = angle_from_pulse(y, fs, pulses_per_rev=ppr)
        rpm = rpm_from_pulse(y, fs, pulses_per_rev=ppr)
        return theta, rpm

    def _theta_for_tacho(
        self, tacho: Series, *, n_signal: int, fs: float
    ) -> np.ndarray:
        theta, _rpm = self._prepare_tacho_arrays(
            tacho, n_signal=n_signal, fs=fs
        )
        return theta


    def _on_mode_changed(self, _index: int) -> None:
        stft = self._is_stft_mode()
        self._static_avg_box.setVisible(not stft)
        self._stft_box.setVisible(stft)
        if stft:
            self._df_spin.setToolTip(
                "0 = auto (min(n, 1024) samples); sets frame length via Δf = fs / NFFT"
            )
        elif self._avg_check.isChecked():
            self._df_spin.setToolTip(
                "0 = auto (min(n, 1024) samples); sets Welch segment length via Δf = fs / NFFT"
            )
        else:
            self._df_spin.setToolTip(
                "0 = auto (full signal length); NFFT = n gives Δf = fs / n"
            )
        self._update_derived()

    @staticmethod
    def _configure_form(form: QFormLayout) -> None:
        form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        form.setLabelAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        form.setFormAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )

    @staticmethod
    def _hint_label(text: str = "—") -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("color: #666; font-size: 11px;")
        label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        return label

    @staticmethod
    def _configure_welch_spin(spin: QSpinBox | QDoubleSpinBox) -> None:
        spin.setFixedWidth(WELCH_SPIN_WIDTH)
        spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def _stft_hop_for_preview(self, *, nfft: int, fs: float) -> int:
        if self._stft_step_mode() == STFT_STEP_DT:
            return stft_hop_samples(
                frame_len=nfft,
                sample_rate=fs,
                step_s=float(self._stft_step_dt_spin.value()),
            )
        return stft_hop_samples(
            frame_len=nfft,
            sample_rate=fs,
            overlap=float(self._stft_overlap_spin.value()) / 100.0,
        )

    def _update_stft_angle_preview(
        self, series_list: list[Series], *, nfft: int, fs: float
    ) -> None:
        dth = float(self._stft_dtheta_spin.value())
        tacho = self._selected_tacho_series()
        if tacho is None:
            self._stft_frames_max_label.setText("(max: —)")
            self._stft_step_hint_label.setText(
                f"Δθ = {dth:.4g} deg  ·  select a tacho series"
            )
            self._run_btn.setEnabled(False)
            return
        frame_counts: list[int] = []
        rev_lo: float | None = None
        rev_hi: float | None = None
        err: str | None = None
        for s in series_list:
            try:
                th = self._theta_for_tacho(
                    tacho, n_signal=int(s.n_samples), fs=float(s.sample_rate)
                )
            except Exception as exc:  # noqa: BLE001
                err = str(exc)
                break
            fl = min(nfft, int(s.n_samples))
            starts, _times, angles = equal_angle_frame_starts(
                th, frame_len=fl, delta_theta_deg=dth, sample_rate=float(s.sample_rate)
            )
            frame_counts.append(len(starts))
            if th.size:
                span = float(th[-1] - th[0])
                revs = span / 360.0
                rev_lo = revs if rev_lo is None else min(rev_lo, revs)
                rev_hi = revs if rev_hi is None else max(rev_hi, revs)
        if err is not None:
            self._stft_frames_max_label.setText("(max: —)")
            self._stft_step_hint_label.setText(f"Tacho error: {err}")
            self._run_btn.setEnabled(False)
            return
        if not frame_counts or max(frame_counts) == 0:
            self._stft_frames_max_label.setText("(max: 0)")
            kind = self._tacho_kind()
            self._stft_step_hint_label.setText(
                f"Δθ = {dth:.4g} deg  ·  no frames "
                f"(tacho too short or insufficient {kind} edges)"
            )
            self._run_btn.setEnabled(False)
            return
        lo, hi = min(frame_counts), max(frame_counts)
        if lo == hi:
            self._stft_frames_max_label.setText(f"(max: {hi})")
        else:
            self._stft_frames_max_label.setText(f"(max: {lo}–{hi})")
        hint = f"Δθ = {dth:.4g} deg  ·  tacho: {tacho.display_name}"
        if rev_lo is not None and rev_hi is not None:
            if abs(rev_lo - rev_hi) < 1e-9:
                hint += f"  ·  ≈ {rev_lo:.4g} rev"
            else:
                hint += f"  ·  ≈ {rev_lo:.4g}–{rev_hi:.4g} rev"
        self._stft_step_hint_label.setText(hint)

    def _update_stft_rpm_preview(
        self, series_list: list[Series], *, nfft: int, fs: float
    ) -> None:
        drpm = float(self._stft_drpm_spin.value())
        tacho = self._selected_tacho_series()
        if tacho is None:
            self._stft_frames_max_label.setText("(max: —)")
            self._stft_step_hint_label.setText(
                f"ΔRPM = {drpm:.4g}  ·  select a tacho series"
            )
            self._run_btn.setEnabled(False)
            return
        frame_counts: list[int] = []
        rpm_lo: float | None = None
        rpm_hi: float | None = None
        err: str | None = None
        for s in series_list:
            try:
                _th, rpm = self._prepare_tacho_arrays(
                    tacho, n_signal=int(s.n_samples), fs=float(s.sample_rate)
                )
            except Exception as exc:  # noqa: BLE001
                err = str(exc)
                break
            fl = min(nfft, int(s.n_samples))
            starts, _times, rpms, _angles = equal_rpm_frame_starts(
                rpm,
                frame_len=fl,
                delta_rpm=drpm,
                sample_rate=float(s.sample_rate),
            )
            frame_counts.append(len(starts))
            if rpm.size:
                lo = float(np.min(rpm))
                hi = float(np.max(rpm))
                rpm_lo = lo if rpm_lo is None else min(rpm_lo, lo)
                rpm_hi = hi if rpm_hi is None else max(rpm_hi, hi)
        if err is not None:
            self._stft_frames_max_label.setText("(max: —)")
            self._stft_step_hint_label.setText(f"Tacho error: {err}")
            self._run_btn.setEnabled(False)
            return
        if not frame_counts or max(frame_counts) == 0:
            self._stft_frames_max_label.setText("(max: 0)")
            kind = self._tacho_kind()
            self._stft_step_hint_label.setText(
                f"ΔRPM = {drpm:.4g}  ·  no frames "
                f"(need a rising RPM ramp or more {kind} edges)"
            )
            self._run_btn.setEnabled(False)
            return
        lo, hi = min(frame_counts), max(frame_counts)
        if lo == hi:
            self._stft_frames_max_label.setText(f"(max: {hi})")
        else:
            self._stft_frames_max_label.setText(f"(max: {lo}–{hi})")
        hint = f"ΔRPM = {drpm:.4g}  ·  tacho: {tacho.display_name}"
        if rpm_lo is not None and rpm_hi is not None:
            hint += f"  ·  ≈ {rpm_lo:.4g}–{rpm_hi:.4g} RPM"
        self._stft_step_hint_label.setText(hint)

    @staticmethod
    def _configure_name_field(edit: QLineEdit) -> None:
        edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def _requested_delta_f_hz(self) -> float | None:
        df = float(self._df_spin.value())
        return None if df <= 0 else df

    def _cap_nfft_to_signal(self) -> bool:
        return self._is_stft_mode() or self._avg_check.isChecked()

    def _effective_nfft(self, *, n: int, fs: float) -> int:
        return resolve_nfft(
            n,
            fs,
            delta_f_hz=self._requested_delta_f_hz(),
            cap_to_signal_length=self._cap_nfft_to_signal(),
        )

    def _on_avg_toggled(self, enabled: bool) -> None:
        if self._is_stft_mode():
            return
        self._welch_overlap_spin.setEnabled(enabled)
        self._avg_count_spin.setEnabled(enabled)
        if enabled:
            self._df_spin.setToolTip(
                "0 = auto (min(n, 1024) samples); sets Welch segment length via Δf = fs / NFFT"
            )
        else:
            self._df_spin.setToolTip(
                "0 = auto (full signal length); NFFT = n gives Δf = fs / n"
            )
        self._update_derived()

    def _list_drag_enter(self, event) -> None:  # noqa: ANN001
        if event.mimeData().hasFormat(SERIES_MIME):
            event.acceptProposedAction()
        else:
            event.ignore()

    def _list_drag_move(self, event) -> None:  # noqa: ANN001
        if event.mimeData().hasFormat(SERIES_MIME):
            event.acceptProposedAction()
        else:
            event.ignore()

    def _list_drop(self, event) -> None:  # noqa: ANN001
        raw = bytes(event.mimeData().data(SERIES_MIME)).decode("utf-8")
        try:
            ids = json.loads(raw)
        except json.JSONDecodeError:
            event.ignore()
            return
        if not isinstance(ids, list) or not ids:
            event.ignore()
            return
        added = self.add_series_ids([str(x) for x in ids])
        event.acceptProposedAction()
        if added:
            self._flash_status(f"Added {added} series to analysis set")

    def _flash_status(self, message: str) -> None:
        self._selection_label.setText(message)

    def add_series_ids(self, series_ids: list[str]) -> int:
        """Add time series ids to the analysis set. Returns how many were newly added."""
        project = self._get_project()
        added = 0
        for sid in series_ids:
            series = project.get(sid)
            if series is None or series.is_fft_result():
                continue
            if sid in self._picked_ids:
                continue
            self._picked_ids.append(sid)
            added += 1
        if added:
            self._rebuild_list()
        else:
            self._update_derived()
        return added

    def add_from_project_selection(self) -> None:
        ids = self._get_project_selection()
        if not ids:
            QMessageBox.information(
                self,
                "Add series",
                "Select one or more time series in Project data (left), then click Add.",
            )
            return
        added = self.add_series_ids(ids)
        if added == 0:
            QMessageBox.information(
                self,
                "Add series",
                "Nothing new to add (already in the list, or only spectrum results selected).",
            )

    def remove_selected_from_pick(self) -> None:
        rows = sorted(
            {i.row() for i in self._series_list.selectedIndexes()},
            reverse=True,
        )
        if not rows:
            QMessageBox.information(
                self, "Remove", "Select items in this list to remove."
            )
            return
        for row in rows:
            if 0 <= row < len(self._picked_ids):
                del self._picked_ids[row]
        self._rebuild_list()

    def clear_pick(self) -> None:
        self._picked_ids.clear()
        self._rebuild_list()

    def _selected_series_list(self) -> list[Series]:
        project = self._get_project()
        out: list[Series] = []
        for sid in self._picked_ids:
            series = project.get(sid)
            if series is not None and not series.is_fft_result():
                out.append(series)
        return out

    def refresh_series(self) -> None:
        """Prune deleted series and refresh labels (called from MainWindow)."""
        project = self._get_project()
        self._picked_ids = [
            sid
            for sid in self._picked_ids
            if (s := project.get(sid)) is not None and not s.is_fft_result()
        ]
        self._rebuild_list()

    def _rebuild_list(self) -> None:
        project = self._get_project()
        self._series_list.clear()
        for sid in self._picked_ids:
            series = project.get(sid)
            if series is None:
                continue
            item = QListWidgetItem(series.display_name)
            item.setData(Qt.ItemDataRole.UserRole, sid)
            self._series_list.addItem(item)
        self._rebuild_tacho_combo()
        self._update_derived()

    def _rebuild_tacho_combo(self) -> None:
        """Refresh tacho picker from the Data to analyze list only."""
        project = self._get_project()
        prev = self._stft_tacho_combo.currentData()
        self._stft_tacho_combo.blockSignals(True)
        self._stft_tacho_combo.clear()
        self._stft_tacho_combo.addItem("(none)", None)
        for sid in self._picked_ids:
            series = project.get(sid)
            if series is None or series.is_fft_result():
                continue
            self._stft_tacho_combo.addItem(series.display_name, sid)
        if prev is not None:
            idx = self._stft_tacho_combo.findData(prev)
            if idx >= 0:
                self._stft_tacho_combo.setCurrentIndex(idx)
        self._stft_tacho_combo.blockSignals(False)
        if self._uses_tacho_stepping():
            self._apply_tacho_series_suggestion()

    def _current_params(self) -> SpectrumParams:
        series_list = self._selected_series_list()
        if series_list:
            ref = series_list[0]
            nfft = self._effective_nfft(n=int(ref.n_samples), fs=float(ref.sample_rate))
            nfft_arg = None if self._requested_delta_f_hz() is None else nfft
        else:
            nfft_arg = None
        if self._is_stft_mode():
            mode = self._stft_step_mode()
            step_s = (
                float(self._stft_step_dt_spin.value())
                if mode == STFT_STEP_DT
                else None
            )
            tacho = self._selected_tacho_series()
            equal_angle = mode == STFT_STEP_ANGLE
            equal_rpm = mode == STFT_STEP_RPM
            uses_tacho = equal_angle or equal_rpm
            return SpectrumParams(
                window=str(self._window_combo.currentData() or "hann"),
                nfft=nfft_arg,
                averaging=False,
                overlap=float(self._stft_overlap_spin.value()) / 100.0,
                amplitude_scale=str(self._scale_combo.currentData() or "peak"),
                stft_step_s=step_s,
                stft_step_mode=mode,
                delta_theta_deg=(
                    float(self._stft_dtheta_spin.value()) if equal_angle else None
                ),
                delta_rpm=(
                    float(self._stft_drpm_spin.value()) if equal_rpm else None
                ),
                tacho_kind=(
                    TACHO_KIND_PULSE
                    if equal_angle
                    else (self._tacho_kind() if equal_rpm else None)
                ),
                pulses_per_rev=(
                    int(self._stft_ppr_spin.value())
                    if uses_tacho
                    and (
                        equal_angle
                        or self._tacho_kind() == TACHO_KIND_PULSE
                    )
                    else None
                ),
                tacho_series_id=tacho.id if (uses_tacho and tacho is not None) else None,
            )
        averaging = self._avg_check.isChecked()
        n_avg = int(self._avg_count_spin.value())
        return SpectrumParams(
            window=str(self._window_combo.currentData() or "hann"),
            nfft=nfft_arg,
            averaging=averaging,
            overlap=float(self._welch_overlap_spin.value()) / 100.0,
            n_averages=None if (not averaging or n_avg <= 0) else n_avg,
            amplitude_scale=str(self._scale_combo.currentData() or "peak"),
        )

    def _effective_result_suffix(self) -> str:
        text = self._name_edit.text().strip()
        if text:
            return text
        return (
            DEFAULT_STFT_RESULT_SUFFIX
            if self._is_stft_mode()
            else DEFAULT_RESULT_SUFFIX
        )

    def _update_dataset_name_hint(self) -> None:
        project = self._get_project()
        params = self._current_params()
        n_series = len(self._selected_series_list())
        self._dataset_name_edit.setPlaceholderText(
            self._auto_dataset_label(project, params, n_series)
        )

    def _update_suffix_preview(self) -> None:
        series_list = self._selected_series_list()
        if not series_list:
            self._suffix_preview.clear()
            return
        suffix = self._effective_result_suffix()
        example = f"{series_list[0].name} {suffix}".strip()
        self._suffix_preview.setText(f"Preview: {example}")

    @staticmethod
    def _selection_validation_errors(series_list: list[Series]) -> list[str]:
        if len(series_list) < 2:
            return []
        rates = sorted({round(float(s.sample_rate), 9) for s in series_list})
        if len(rates) > 1:
            rate_text = ", ".join(f"{fs:.6g} Hz" for fs in rates)
            return [
                "Cannot run FFT: selected series have different sample rates "
                f"({rate_text}). Remove or fix mismatched series."
            ]
        return []

    def _update_derived(self) -> None:
        series_list = self._selected_series_list()
        self._selection_label.setText(f"{len(series_list)} series in analysis set")
        issues = self._selection_validation_errors(series_list)
        if issues:
            self._selection_error_label.setText(issues[0])
            self._selection_error_label.show()
            self._run_btn.setEnabled(False)
        else:
            self._selection_error_label.clear()
            self._selection_error_label.hide()
            self._run_btn.setEnabled(bool(series_list))

        if not series_list:
            self._df_min_label.setText("(finest: —)")
            self._nfft_hint_label.setText("NFFT = —")
            self._avg_max_label.setText("(max: —)")
            self._stft_frames_max_label.setText("(max: —)")
            self._stft_step_hint_label.setText("—")
            self._update_dataset_name_hint()
            self._update_suffix_preview()
            return

        lengths = {int(s.n_samples) for s in series_list}
        ref = series_list[0]
        fs = float(ref.sample_rate)
        n = int(ref.n_samples)
        n_min = min(lengths)
        n_max = max(lengths)

        params = self._current_params()
        cap = self._cap_nfft_to_signal()
        finest_n = n_min if cap else n_max
        finest_df = frequency_resolution_hz(fs, max(2, finest_n))
        coarsest_df = frequency_resolution_hz(fs, 2)
        self._df_spin.blockSignals(True)
        self._df_spin.setMaximum(max(coarsest_df, finest_df))
        self._df_spin.setSingleStep(max(finest_df / 10.0, 1e-9))
        self._df_spin.blockSignals(False)

        nfft = self._effective_nfft(n=n, fs=fs)
        self._df_min_label.setText(f"(finest: {finest_df:.6g} Hz)")
        eff_df = frequency_resolution_hz(fs, nfft)
        req_df = self._requested_delta_f_hz()
        if req_df is not None and abs(req_df - eff_df) > max(1e-12, eff_df * 1e-6):
            self._nfft_hint_label.setText(
                f"NFFT = {nfft}  ·  effective Δf = {eff_df:.6g} Hz"
            )
        else:
            self._nfft_hint_label.setText(f"NFFT = {nfft}")

        if self._is_stft_mode():
            mode = self._stft_step_mode()
            if mode == STFT_STEP_ANGLE:
                self._update_stft_angle_preview(series_list, nfft=nfft, fs=fs)
            elif mode == STFT_STEP_RPM:
                self._update_stft_rpm_preview(series_list, nfft=nfft, fs=fs)
            else:
                hop = self._stft_hop_for_preview(nfft=nfft, fs=fs)
                max_frames = [
                    stft_frame_count(
                        int(s.n_samples), min(nfft, int(s.n_samples)), hop
                    )
                    for s in series_list
                ]
                lo, hi = min(max_frames), max(max_frames)
                if lo == hi:
                    self._stft_frames_max_label.setText(f"(max: {hi})")
                else:
                    self._stft_frames_max_label.setText(f"(max: {lo}–{hi})")
                if self._stft_step_mode() == STFT_STEP_DT:
                    dt = float(self._stft_step_dt_spin.value())
                    ov = 100.0 * stft_overlap_equiv(nfft, hop)
                    self._stft_step_hint_label.setText(
                        f"Δt = {dt:.6g} s  ·  Overlap ≈ {ov:.4g} %"
                    )
                else:
                    overlap = float(self._stft_overlap_spin.value()) / 100.0
                    self._stft_step_hint_label.setText(
                        f"Δt = {hop / fs:.6g} s  ·  Overlap = {overlap * 100:.4g} %"
                    )
            self._avg_max_label.setText("(max: —)")
        elif params.averaging:
            max_avgs = [
                max_averages(
                    int(s.n_samples),
                    min(nfft, int(s.n_samples)),
                    params.overlap,
                )
                for s in series_list
            ]
            max_lo, max_hi = min(max_avgs), max(max_avgs)
            if max_lo == max_hi:
                self._avg_max_label.setText(f"(max: {max_hi})")
            else:
                self._avg_max_label.setText(f"(max: {max_lo}–{max_hi})")
            self._stft_frames_max_label.setText("(max: —)")
            self._stft_step_hint_label.setText("—")
        else:
            self._avg_max_label.setText("(max: —)")
            self._stft_frames_max_label.setText("(max: —)")
            self._stft_step_hint_label.setText("—")

        self._update_dataset_name_hint()
        self._update_suffix_preview()

    def _auto_dataset_label(self, project: Project, params: SpectrumParams, n_series: int) -> str:
        run_n = 1 + sum(1 for s in project.sources if str(s.id).startswith("analysis"))
        kind = "STFT" if self._is_stft_mode() else "FFT"
        return f"{kind} {run_n} · {params.window} · {n_series} ch"

    def _unique_dataset_label(self, project: Project, base: str) -> str:
        base = base.strip()
        if not base:
            return base
        existing = {s.label for s in project.sources}
        if base not in existing:
            return base
        n = 2
        while f"{base} ({n})" in existing:
            n += 1
        return f"{base} ({n})"

    def _dataset_label_for_run(
        self, project: Project, params: SpectrumParams, n_series: int
    ) -> str:
        custom = self._dataset_name_edit.text().strip()
        if custom:
            return self._unique_dataset_label(project, custom)
        return self._auto_dataset_label(project, params, n_series)

    def run_fft(self) -> None:
        series_list = self._selected_series_list()
        if not series_list:
            QMessageBox.warning(
                self,
                "FFT",
                "Add at least one time series from Project data first.",
            )
            return

        issues = self._selection_validation_errors(series_list)
        if issues:
            QMessageBox.warning(self, "FFT", issues[0])
            return

        if self._is_stft_mode():
            self._run_stft(series_list)
            return

        params = self._current_params()
        suffix = self._effective_result_suffix()
        project = self._get_project()
        dataset_label = self._dataset_label_for_run(
            project, params, len(series_list)
        )
        dataset = project.create_analysis_dataset(label=dataset_label)
        new_ids: list[str] = []
        errors: list[str] = []

        for series in series_list:
            try:
                spec = compute_spectrum(
                    series.data, float(series.sample_rate), params
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{series.name}: {exc}")
                continue
            if spec.frequency_hz.size == 0:
                errors.append(f"{series.name}: empty spectrum (signal too short)")
                continue
            name = f"{series.name} {suffix}".strip()
            sid = project.add_spectrum_result(
                name=name,
                frequency_hz=spec.frequency_hz,
                magnitude=spec.magnitude,
                phase_rad=spec.phase_rad,
                sample_rate=float(series.sample_rate),
                unit=series.unit,
                parent=series,
                source=dataset,
                params_dict={
                    **params.to_dict(),
                    "delta_f_hz": spec.delta_f_hz,
                    "n_segments": spec.n_segments,
                },
            )
            new_ids.append(sid)

        if not new_ids:
            # Drop empty dataset if every channel failed.
            project.sources = [s for s in project.sources if s.id != dataset.id]
            detail = "\n".join(errors) if errors else "No series produced a spectrum."
            QMessageBox.critical(self, "FFT failed", detail)
            return

        try:
            self._on_result(new_ids)  # type: ignore[arg-type]
        except TypeError:
            for sid in new_ids:
                self._on_result(sid)  # type: ignore[arg-type]

        msg = (
            f"Created dataset “{dataset.label}” "
            f"with {len(new_ids)} spectrum result(s)"
        )
        if errors:
            msg += f" ({len(errors)} skipped)"
            QMessageBox.warning(
                self,
                "FFT finished with warnings",
                msg + "\n\n" + "\n".join(errors),
            )
        self.status_message = msg

    def _run_stft(self, series_list: list[Series]) -> None:
        params = self._current_params()
        delta_f = self._requested_delta_f_hz()
        suffix = self._effective_result_suffix()
        project = self._get_project()
        mode = self._stft_step_mode()
        equal_angle = mode == STFT_STEP_ANGLE
        equal_rpm = mode == STFT_STEP_RPM
        uses_tacho = equal_angle or equal_rpm
        tacho: Series | None = None
        if uses_tacho:
            tacho = self._selected_tacho_series()
            if tacho is None:
                label = "Equal RPM" if equal_rpm else "Equal angle"
                QMessageBox.critical(
                    self,
                    "STFT failed",
                    f"{label} stepping requires a tacho series.",
                )
                return

        dataset_label = self._dataset_label_for_run(
            project, params, len(series_list)
        )
        dataset = project.create_analysis_dataset(
            label=dataset_label, provenance="analysis:stft"
        )
        new_ids: list[str] = []
        errors: list[str] = []

        for series in series_list:
            if uses_tacho and tacho is not None and series.id == tacho.id:
                # Skip using tacho as an FFT channel when it was also picked.
                continue
            try:
                theta = None
                rpm = None
                if uses_tacho and tacho is not None:
                    theta, rpm = self._prepare_tacho_arrays(
                        tacho,
                        n_signal=int(series.n_samples),
                        fs=float(series.sample_rate),
                    )
                stft = compute_stft(
                    series.data,
                    float(series.sample_rate),
                    params,
                    delta_f_hz=delta_f,
                    theta_deg=theta,
                    delta_theta_deg=(
                        float(self._stft_dtheta_spin.value()) if equal_angle else None
                    ),
                    rpm=rpm if equal_rpm else None,
                    delta_rpm=(
                        float(self._stft_drpm_spin.value()) if equal_rpm else None
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{series.name}: {exc}")
                continue
            if stft.n_frames == 0:
                errors.append(f"{series.name}: empty spectrogram (signal too short)")
                continue
            name = f"{series.name} {suffix}".strip()
            sid = project.add_spectrogram_result(
                name=name,
                frequency_hz=stft.frequency_hz,
                time_s=stft.time_s,
                magnitude=stft.magnitude,
                phase_rad=stft.phase_rad,
                sample_rate=float(series.sample_rate),
                unit=series.unit,
                parent=series,
                source=dataset,
                angle_deg=stft.angle_deg if stft.angle_deg.size else None,
                rpm=stft.rpm if stft.rpm.size else None,
                params_dict={
                    **params.to_dict(),
                    "delta_f_hz": stft.delta_f_hz,
                    "n_frames": stft.n_frames,
                    "hop_samples": stft.hop_samples,
                    "frame_len": stft.frame_len,
                },
            )
            new_ids.append(sid)

        if not new_ids:
            project.sources = [s for s in project.sources if s.id != dataset.id]
            detail = (
                "\n".join(errors)
                if errors
                else "No series produced a spectrogram."
            )
            QMessageBox.critical(self, "STFT failed", detail)
            return

        try:
            self._on_result(new_ids)  # type: ignore[arg-type]
        except TypeError:
            for sid in new_ids:
                self._on_result(sid)  # type: ignore[arg-type]

        msg = (
            f"Created dataset “{dataset.label}” "
            f"with {len(new_ids)} spectrogram result(s)"
        )
        if errors:
            msg += f" ({len(errors)} skipped)"
            QMessageBox.warning(
                self,
                "STFT finished with warnings",
                msg + "\n\n" + "\n".join(errors),
            )
        self.status_message = msg
