"""Main window: multi-panel viewer + spectrum pairs (M2/M3 minimal)."""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup, QKeyEvent, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSplitter,
    QSpinBox,
    QStackedWidget,
    QStatusBar,
    QStyleFactory,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from pyqtgraph.graphicsItems.InfiniteLine import InfLineLabel
from pyqtgraph.graphicsItems.ViewBox.ViewBoxMenu import ViewBoxMenu

from ginkyo.core.dummy import make_sine_with_noise
from ginkyo.core.measure import (
    band_rms,
    find_spectrum_peaks,
    level_db,
    series_amplitude_scale,
    to_db,
)
from ginkyo.core.model import Channel, Recording
from ginkyo.core.project import Project, Series
from ginkyo.core.project_io import take_pending_views
from ginkyo.core.spectrum import compute_spectrum
from ginkyo.export.csv_export import export_csv
from ginkyo.readers.csv_reader import read_csv
from ginkyo.readers.uff import read_uff
from ginkyo.readers.wav import read_wav
from ginkyo.ui.analysis_page import AnalysisPage
from ginkyo.ui.layout_state import (
    LayoutNode,
    WorkspaceLayout,
    build_preset,
    layouts_dir,
    load_layout,
    preset_ids,
    preset_label,
    save_layout,
)
from ginkyo.ui.panel_shell import ROLE_SERIES, ROLE_SOURCE, PanelShell, SeriesTree

_PENS = (
    "#1f4e79",  # navy
    "#c45c26",  # rust
    "#2a9d8f",  # teal
    "#6d597a",  # mauve
    "#e9c46a",  # gold
    "#264653",  # dark teal
    "#e76f51",  # coral
    "#457b9d",  # steel blue
    "#2a9d4f",  # green
    "#9b2226",  # wine
    "#0077b6",  # azure
    "#bc6c25",  # brown
)


def _spectrogram_db(magnitude: np.ndarray) -> np.ndarray:
    return np.asarray(to_db(magnitude), dtype=np.float64)


def _axis_image_extent(axis: np.ndarray) -> tuple[float, float]:
    """Return (edge0, width) that covers bin centers on a uniform-ish axis."""
    x = np.asarray(axis, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return 0.0, 1.0
    if x.size == 1:
        return float(x[0]) - 0.5, 1.0
    step = float(np.median(np.diff(x)))
    if not np.isfinite(step) or step <= 0:
        step = float(x[-1] - x[0]) / max(x.size - 1, 1) or 1.0
    edge0 = float(x[0]) - step / 2.0
    width = step * float(x.size)
    return edge0, width


def _nice_db_levels(lo: float, hi: float, *, step: float = 5.0) -> tuple[float, float]:
    """Round a dB range outward onto a clean step grid."""
    if not np.isfinite(lo) or not np.isfinite(hi):
        return -80.0, 0.0
    if hi < lo:
        lo, hi = hi, lo
    step = max(float(step), 1e-6)
    lo_r = float(math.floor(lo / step) * step)
    hi_r = float(math.ceil(hi / step) * step)
    if lo_r >= hi_r:
        hi_r = lo_r + step
    return lo_r, hi_r


def _colorbar_tick_values(lo: float, hi: float, *, count: int = 9) -> list[float]:
    """Dense-enough, rounded tick values for a color-bar axis."""
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return [float(lo), float(hi)] if hi != lo else [float(lo)]
    span = float(hi - lo)
    raw_step = span / max(count - 1, 1)
    # Prefer 1 / 2 / 5 / 10 … dB steps.
    exp = math.floor(math.log10(max(raw_step, 1e-9)))
    base = 10.0 ** exp
    for mult in (1.0, 2.0, 5.0, 10.0):
        step = base * mult
        if step >= raw_step * 0.8:
            break
    start = math.ceil(lo / step) * step
    ticks = [float(lo)]
    v = start
    while v < hi - 0.25 * step:
        if abs(v - lo) > 0.25 * step and abs(v - hi) > 0.25 * step:
            ticks.append(float(v))
        v += step
    ticks.append(float(hi))
    # Deduplicate while preserving order.
    out: list[float] = []
    for t in ticks:
        if not out or abs(t - out[-1]) > 1e-9:
            out.append(t)
    return out


_MODE_TIME = "time"
_MODE_MAG_PHASE = "mag_phase"
_MODE_REAL_IMAG = "real_imag"
_VIEW_SPECTROGRAM = "spectrogram"


class _NagilizeViewBoxMenu(ViewBoxMenu):
    """Manual min/max show data-domain values even when the axis is in log mode."""

    def __init__(self, view):  # noqa: ANN001
        super().__init__(view)
        self._host: weakref.ref | None = None
        self._marker_actions_added = False
        self._series_menu: QMenu | None = None

    def attach_host(self, host: MainWindow) -> None:
        self._host = weakref.ref(host)
        if self._marker_actions_added:
            return
        self._marker_actions_added = True
        self.addSeparator()
        self._series_menu = self.addMenu("Panel series")
        self.aboutToShow.connect(self._rebuild_series_menu)
        self._spec_scale_act = self.addAction("Spectrogram color scale…")
        self._spec_scale_act.triggered.connect(self._open_spectrogram_color_scale)
        self.aboutToShow.connect(self._update_spectrogram_menu_actions)
        self._spec_y_menu = self.addMenu("Y axis")
        self._spec_y_time_act = self._spec_y_menu.addAction("Time")
        self._spec_y_time_act.setCheckable(True)
        self._spec_y_time_act.triggered.connect(
            lambda: self._set_spectrogram_y_axis("time")
        )
        self._spec_y_angle_act = self._spec_y_menu.addAction("Angle (deg)")
        self._spec_y_angle_act.setCheckable(True)
        self._spec_y_angle_act.triggered.connect(
            lambda: self._set_spectrogram_y_axis("angle")
        )
        self._spec_y_rpm_act = self._spec_y_menu.addAction("RPM")
        self._spec_y_rpm_act.setCheckable(True)
        self._spec_y_rpm_act.triggered.connect(
            lambda: self._set_spectrogram_y_axis("rpm")
        )
        self.addSeparator()
        add_v = self.addAction("Add vertical line here")
        add_v.triggered.connect(lambda: self._call_host("add_marker", vertical=True))
        add_h = self.addAction("Add horizontal line here")
        add_h.triggered.connect(lambda: self._call_host("add_marker", vertical=False))
        clear_m = self.addAction("Clear marker lines")
        clear_m.triggered.connect(lambda: self._call_host("clear_markers"))
        remove_near = self.addAction("Remove nearest marker")
        remove_near.triggered.connect(lambda: self._call_host("remove_nearest_marker"))

    def _rebuild_series_menu(self) -> None:
        if self._series_menu is None:
            return
        self._series_menu.clear()
        host = self._host() if self._host is not None else None
        if host is None:
            return
        panel = host._panel_for_viewbox(self.view())
        if panel is None or not panel.series_ids:
            empty = self._series_menu.addAction("(no series on this panel)")
            empty.setEnabled(False)
            return
        for sid in list(panel.series_ids):
            series = host._project.get(sid)
            label = series.display_name if series is not None else sid
            sub = self._series_menu.addMenu(label)
            color_act = sub.addAction("Set color…")
            color_act.triggered.connect(
                lambda checked=False, p=panel, s=sid: host.set_panel_series_color(p, s)
            )
            reset_act = sub.addAction("Reset color (use panel order)")
            reset_act.triggered.connect(
                lambda checked=False, p=panel, s=sid: host.reset_panel_series_color(p, s)
            )
            sub.addSeparator()
            rem = sub.addAction("Remove from panel")
            rem.triggered.connect(
                lambda checked=False, p=panel, s=sid: host.remove_series_from_panel(p, s)
            )
        self._series_menu.addSeparator()
        clear_act = self._series_menu.addAction("Clear all series on panel")
        clear_act.triggered.connect(
            lambda checked=False, p=panel: host.clear_panel_series(p)
        )

    def _update_spectrogram_menu_actions(self) -> None:
        host = self._host() if self._host is not None else None
        is_spec = False
        has_angle = False
        has_rpm = False
        y_axis = "time"
        if host is not None:
            panel = host._panel_for_viewbox(self.view())
            is_spec = panel is not None and panel.is_spectrogram()
            if is_spec and panel is not None:
                y_axis = panel.spectrogram_y_axis
                has_angle = host._panel_has_spectrogram_angle(panel)
                has_rpm = host._panel_has_spectrogram_rpm(panel)
        act = getattr(self, "_spec_scale_act", None)
        if act is not None:
            act.setVisible(is_spec)
            act.setEnabled(is_spec)
        y_menu = getattr(self, "_spec_y_menu", None)
        if y_menu is not None:
            y_menu.menuAction().setVisible(is_spec)
            y_menu.setEnabled(is_spec)
        time_act = getattr(self, "_spec_y_time_act", None)
        angle_act = getattr(self, "_spec_y_angle_act", None)
        rpm_act = getattr(self, "_spec_y_rpm_act", None)
        if time_act is not None:
            time_act.setChecked(y_axis == "time")
            time_act.setEnabled(is_spec)
        if angle_act is not None:
            angle_act.setChecked(y_axis == "angle")
            angle_act.setEnabled(is_spec and has_angle)
        if rpm_act is not None:
            rpm_act.setChecked(y_axis == "rpm")
            rpm_act.setEnabled(is_spec and has_rpm)

    def _set_spectrogram_y_axis(self, axis: str) -> None:
        host = self._host() if self._host is not None else None
        if host is None:
            return
        panel = host._panel_for_viewbox(self.view())
        if panel is not None:
            try:
                host._active_panel = host._panels.index(panel)
            except ValueError:
                pass
        host.set_spectrogram_y_axis(axis)

    def _open_spectrogram_color_scale(self) -> None:
        host = self._host() if self._host is not None else None
        if host is None:
            return
        panel = host._panel_for_viewbox(self.view())
        if panel is not None:
            try:
                host._active_panel = host._panels.index(panel)
            except ValueError:
                pass
        host.show_spectrogram_color_scale()

    def _call_host(self, method: str, **kwargs) -> None:
        host = self._host() if self._host is not None else None
        if host is None:
            return
        getattr(host, method)(**kwargs)

    def updateState(self) -> None:
        super().updateState()
        view = self.view()
        if view is None:
            return
        state = view.getState(copy=False)
        for i in (0, 1):
            if not state["logMode"][i]:
                continue
            lo, hi = state["targetRange"][i]
            self.ctrl[i].minText.setText(f"{10 ** float(lo):0.5g}")
            self.ctrl[i].maxText.setText(f"{10 ** float(hi):0.5g}")

    def _validateRangeText(self, axis: int) -> list[float]:
        inputs = (self.ctrl[axis].minText.text(), self.ctrl[axis].maxText.text())
        vals = list(self.view().viewRange()[axis])
        log_mode = bool(self.view().state["logMode"][axis])
        for i, text in enumerate(inputs):
            try:
                v = float(text)
            except ValueError:
                continue
            if log_mode:
                if v <= 0:
                    continue
                vals[i] = float(np.log10(v))
            else:
                vals[i] = v
        return vals


class _NagilizeViewBox(pg.ViewBox):
    """3-button: right-drag scales. 1-button: no right-drag scale."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        super().__init__(*args, **kwargs)
        self._host_ref: weakref.ref | None = None
        if self.menu is not None:
            old = self.menu
            self.menu = _NagilizeViewBoxMenu(self)
            old.setParent(None)
            old.deleteLater()

    def attach_host(self, host: MainWindow) -> None:
        self._host_ref = weakref.ref(host)
        if isinstance(self.menu, _NagilizeViewBoxMenu):
            self.menu.attach_host(host)

    def mouseDragEvent(self, ev, axis=None):  # noqa: ANN001
        if (
            ev.button() == Qt.MouseButton.RightButton
            and self.state["mouseMode"] == pg.ViewBox.RectMode
        ):
            ev.ignore()
            return
        super().mouseDragEvent(ev, axis)

    def mouseClickEvent(self, ev):  # noqa: ANN001
        host = self._host_ref() if self._host_ref is not None else None
        if host is not None and ev.button() == Qt.MouseButton.LeftButton:
            host._set_active_panel_from_viewbox(self)
        super().mouseClickEvent(ev)

    def fit_axis(self, axis: int) -> None:
        prev_visible = list(self.state["autoVisibleOnly"])
        try:
            if axis == pg.ViewBox.YAxis:
                self.state["autoVisibleOnly"][0] = False
                self.state["autoVisibleOnly"][1] = True
            else:
                self.state["autoVisibleOnly"][0] = False
                self.state["autoVisibleOnly"][1] = False
            self.enableAutoRange(axis=axis, enable=True)
            self.disableAutoRange(axis=axis)
        finally:
            self.state["autoVisibleOnly"][0] = prev_visible[0]
            self.state["autoVisibleOnly"][1] = prev_visible[1]


class _MarkerLine(pg.InfiniteLine):
    """Draggable marker; right-click to set/remove; arrows nudge when selected."""

    def __init__(self, host: MainWindow, *, vertical: bool, **kwargs) -> None:
        hover = kwargs.pop("hoverPen", None)
        self._base_color = "#c0392b" if vertical else "#2980b9"
        if hover is None:
            hover = pg.mkPen(self._base_color, width=4)
        super().__init__(hoverPen=hover, **kwargs)
        self._host_ref = weakref.ref(host)
        self.vertical = vertical
        self.setZValue(200)
        self.setAcceptedMouseButtons(
            Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton
        )
        self.set_selected_style(False)

    def set_selected_style(self, selected: bool) -> None:
        width = 2.5 if selected else 1.5
        style = Qt.PenStyle.SolidLine if selected else Qt.PenStyle.DashLine
        self.setPen(pg.mkPen(self._base_color, width=width, style=style))
        self.setHoverPen(pg.mkPen(self._base_color, width=max(width + 2, 4)))

    def mouseClickEvent(self, ev):  # noqa: ANN001
        host = self._host_ref()
        if host is None:
            ev.ignore()
            return

        if ev.button() == Qt.MouseButton.LeftButton:
            host.select_marker(self)
            super().mouseClickEvent(ev)
            ev.accept()
            return

        if ev.button() == Qt.MouseButton.RightButton:
            host.select_marker(self)
            menu = QMenu()
            edit_act = menu.addAction("Set position…")
            remove_act = menu.addAction("Remove this line")
            chosen = menu.exec(ev.screenPos().toPoint())
            if chosen == edit_act:
                host.edit_marker_value(self)
            elif chosen == remove_act:
                host.remove_marker(self)
            ev.accept()
            return

        super().mouseClickEvent(ev)


class _AutoFitAxis(pg.AxisItem):
    """Double-click an axis to auto-range that axis only."""

    def mouseDoubleClickEvent(self, event):  # noqa: ANN001
        if event.button() != Qt.MouseButton.LeftButton:
            event.ignore()
            return
        if not self.boundingRect().contains(self.mapFromScene(event.scenePos())):
            event.ignore()
            return
        vb = self.linkedView()
        if vb is None or not hasattr(vb, "fit_axis"):
            event.ignore()
            return
        axis = (
            pg.ViewBox.XAxis
            if self.orientation in ("bottom", "top")
            else pg.ViewBox.YAxis
        )
        vb.fit_axis(axis)
        event.accept()


@dataclass
class _PlotPanel:
    plot: pg.PlotItem
    plot_widget: pg.PlotWidget
    shell: PanelShell
    v_line: pg.InfiniteLine
    curves: list[pg.PlotDataItem] = field(default_factory=list)
    image_items: list[pg.ImageItem] = field(default_factory=list)
    color_bars: list[pg.ColorBarItem] = field(default_factory=list)
    spectrogram_levels: tuple[float, float] | None = None
    spectrogram_y_axis: str = "time"  # time | angle | rpm
    series_ids: list[str] = field(default_factory=list)
    series_colors: dict[str, str] = field(default_factory=dict)
    y_kind: str = "time"  # time | mag | phase | real | imag | spectrogram
    layout_leaf: LayoutNode | None = None

    @property
    def widget(self) -> PanelShell:
        return self.shell

    def is_spectrogram(self) -> bool:
        return self.y_kind == _VIEW_SPECTROGRAM

    def is_frequency(self) -> bool:
        return self.y_kind not in ("time", _VIEW_SPECTROGRAM)


@dataclass
class _SyncedMarker:
    """One logical marker mirrored on every panel."""

    vertical: bool
    lines: list[_MarkerLine] = field(default_factory=list)


@dataclass
class _PageState:
    """One view page: layout + live plot widgets (project data is shared)."""

    title: str
    workspace: WorkspaceLayout
    display_mode: str = _MODE_TIME
    mag_db: bool = False
    measure_use_band: bool = False
    measure_f_lo: float = 0.0
    measure_f_hi: float = 0.0
    built: bool = False
    panels: list[_PlotPanel] = field(default_factory=list)
    synced_markers: list[_SyncedMarker] = field(default_factory=list)
    selected_marker: _MarkerLine | None = None
    active_panel: int = 0
    root_plot_widget: QWidget | None = None
    node_widgets: list[tuple[LayoutNode, QWidget]] = field(default_factory=list)
    mouse_proxies: list = field(default_factory=list)
    pending_markers: list[tuple[bool, float]] = field(default_factory=list)
    pending_view_ranges: list[dict] = field(default_factory=list)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Ginkyo 吟響")
        self.resize(1200, 700)

        self._project = Project()
        self._synced_markers: list[_SyncedMarker] = []
        self._selected_marker: _MarkerLine | None = None
        self._last_view_pos: tuple[float, float] | None = None
        self._panels: list[_PlotPanel] = []
        self._active_panel = 0
        self._workspace = WorkspaceLayout()
        self._display_mode = _MODE_TIME
        self._pages: list[_PageState] = [
            _PageState(title="Page 1", workspace=self._workspace)
        ]
        self._page_index = 0
        self._suppress_tab_change = False
        self._spectrum_actions: dict[str, QAction] = {}
        self._layout_actions: dict[str, QAction] = {}
        self._updating_assign = False
        self._selected_source_id: str | None = None
        self._mouse_proxies: list = []
        self._root_plot_widget: QWidget | None = None
        self._node_widgets: list[tuple[LayoutNode, QWidget]] = []
        self._link_x = True
        self._link_y = False
        self._mag_db = False
        self._measure_use_band = False
        self._measure_f_lo = 0.0
        self._measure_f_hi = 0.0
        self._updating_measure_dock = False

        central = QWidget(self)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)

        main_split = QSplitter(Qt.Orientation.Horizontal)
        main_split.setChildrenCollapsible(False)
        # Wide transparent handle: easier to grab without a visible gray bar.
        main_split.setHandleWidth(14)
        main_split.setStyleSheet(
            "QSplitter::handle:horizontal {"
            "  width: 14px;"
            "  background: transparent;"
            "  border: none;"
            "  margin: 0;"
            "  padding: 0;"
            "}"
        )

        side = QVBoxLayout()
        side.setContentsMargins(6, 6, 6, 6)

        project_box = QGroupBox("Project data")
        project_layout = QVBoxLayout(project_box)
        filter_row = QHBoxLayout()
        self._project_filter = QLineEdit()
        self._project_filter.setPlaceholderText("Filter…")
        self._project_filter.textChanged.connect(self._refresh_project_tree)
        filter_row.addWidget(self._project_filter, stretch=1)
        self._project_sort = QComboBox()
        self._project_sort.addItem("Sort: name", "name")
        self._project_sort.addItem("Sort: point", "point")
        self._project_sort.addItem("Sort: dof", "dof")
        self._project_sort.currentIndexChanged.connect(self._refresh_project_tree)
        filter_row.addWidget(self._project_sort)
        project_layout.addLayout(filter_row)
        self._project_tree = SeriesTree()
        self._project_tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._project_tree.setToolTip(
            "Drag series onto a plot to add · Shift+drop replaces · "
            "Right-click → Add to Analysis / Panel series to remove from plots"
        )
        self._project_tree.itemSelectionChanged.connect(self._on_project_selection)
        self._project_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._project_tree.customContextMenuRequested.connect(self._project_tree_menu)
        project_layout.addWidget(self._project_tree, stretch=1)
        project_btns = QHBoxLayout()
        remove_src_btn = QPushButton("Remove source")
        remove_src_btn.clicked.connect(self.remove_selected_source)
        clear_proj_btn = QPushButton("Clear")
        clear_proj_btn.clicked.connect(self.clear_project)
        project_btns.addWidget(remove_src_btn)
        project_btns.addWidget(clear_proj_btn)
        project_layout.addLayout(project_btns)
        side.addWidget(project_box, stretch=1)

        side_wrap = QWidget()
        side_wrap.setLayout(side)
        side_wrap.setMinimumWidth(180)
        main_split.addWidget(side_wrap)

        # Workspace shell: Office-like ribbon (Analysis / Views) + stacked content.
        # View pages stay as a quieter bottom strip inside Views only.
        workspace = QWidget()
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(0)

        ribbon = QFrame()
        ribbon.setObjectName("RibbonBar")
        ribbon.setFixedHeight(42)
        ribbon.setStyleSheet(
            """
            QFrame#RibbonBar {
                background: #f3f2f1;
                border: none;
                border-bottom: 1px solid #c8c6c4;
            }
            QLabel#RibbonBrand {
                color: #1f4e79;
                font-size: 13px;
                font-weight: 700;
                padding: 0 14px 0 4px;
                letter-spacing: 0.2px;
            }
            QToolButton#RibbonTab {
                background: transparent;
                color: #323130;
                border: none;
                border-bottom: 3px solid transparent;
                padding: 8px 18px 6px 18px;
                margin: 0 1px;
                font-size: 12px;
                font-weight: 500;
                min-width: 78px;
            }
            QToolButton#RibbonTab:hover {
                background: #e1dfdd;
                color: #201f1e;
            }
            QToolButton#RibbonTab:checked {
                background: #ffffff;
                color: #1f4e79;
                border-bottom: 3px solid #1f4e79;
                font-weight: 600;
            }
            """
        )
        ribbon_row = QHBoxLayout(ribbon)
        ribbon_row.setContentsMargins(10, 0, 10, 0)
        ribbon_row.setSpacing(0)
        brand = QLabel("Ginkyo 吟響")
        brand.setObjectName("RibbonBrand")
        ribbon_row.addWidget(brand)

        self._ribbon_analysis_btn = QToolButton()
        self._ribbon_analysis_btn.setObjectName("RibbonTab")
        self._ribbon_analysis_btn.setText("Analysis")
        self._ribbon_analysis_btn.setCheckable(True)
        self._ribbon_analysis_btn.setAutoExclusive(True)
        self._ribbon_analysis_btn.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextOnly
        )
        self._ribbon_views_btn = QToolButton()
        self._ribbon_views_btn.setObjectName("RibbonTab")
        self._ribbon_views_btn.setText("Views")
        self._ribbon_views_btn.setCheckable(True)
        self._ribbon_views_btn.setAutoExclusive(True)
        self._ribbon_views_btn.setChecked(True)
        self._ribbon_views_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._ribbon_group = QButtonGroup(self)
        self._ribbon_group.setExclusive(True)
        self._ribbon_group.addButton(self._ribbon_analysis_btn, 0)
        self._ribbon_group.addButton(self._ribbon_views_btn, 1)
        self._ribbon_group.idClicked.connect(self._set_workspace_mode)
        ribbon_row.addWidget(self._ribbon_analysis_btn)
        ribbon_row.addWidget(self._ribbon_views_btn)
        ribbon_row.addStretch(1)
        workspace_layout.addWidget(ribbon)

        self._workspace_stack = QStackedWidget()
        self._analysis_page = AnalysisPage(
            get_project=lambda: self._project,
            get_project_selection=lambda: self._project_tree.selected_series_ids(),
            on_result=self._on_spectrum_result,
        )
        self._workspace_stack.addWidget(self._analysis_page)

        views_host = QWidget()
        views_layout = QVBoxLayout(views_host)
        views_layout.setContentsMargins(0, 0, 0, 0)
        views_layout.setSpacing(0)
        self._tabs = QTabWidget()
        self._tabs.setObjectName("ViewPageTabs")
        self._tabs.setDocumentMode(False)
        self._tabs.setTabsClosable(True)
        self._tabs.setMovable(True)
        self._tabs.setTabPosition(QTabWidget.TabPosition.South)
        # Excel-like sheet strip: left-aligned chips, never elide "Page 1".
        page_bar = self._tabs.tabBar()
        page_bar.setExpanding(False)
        page_bar.setElideMode(Qt.TextElideMode.ElideNone)
        page_bar.setUsesScrollButtons(True)
        # Native styles (esp. macOS) may center tabs; Fusion packs them left.
        fusion = QStyleFactory.create("Fusion")
        if fusion is not None:
            page_bar.setStyle(fusion)
        self._tabs.setStyleSheet(
            """
            QTabWidget#ViewPageTabs::pane {
                border: none;
                border-bottom: 1px solid #b1b1b1;
                background: #ffffff;
            }
            QTabWidget#ViewPageTabs > QTabBar {
                background: #e6e6e6;
                border-top: 1px solid #b1b1b1;
                min-height: 28px;
                alignment: left;
            }
            QTabWidget#ViewPageTabs > QTabBar::tab {
                background: #d4d4d4;
                color: #333333;
                border: 1px solid #a6a6a6;
                border-top: none;
                border-bottom-left-radius: 3px;
                border-bottom-right-radius: 3px;
                padding: 5px 16px 5px 14px;
                margin: 0 1px 0 0;
                min-width: 88px;
                font-size: 12px;
                font-weight: 400;
            }
            QTabWidget#ViewPageTabs > QTabBar::tab:selected {
                background: #ffffff;
                color: #1f4e79;
                border-color: #8a8a8a;
                border-top: 1px solid #ffffff;
                margin-top: -1px;
                padding-top: 6px;
                font-weight: 600;
            }
            QTabWidget#ViewPageTabs > QTabBar::tab:hover:!selected {
                background: #cfcfcf;
                color: #201f1e;
            }
            QTabWidget#ViewPageTabs > QTabBar::close-button {
                subcontrol-position: right;
                padding: 1px;
            }
            QTabWidget#ViewPageTabs QToolButton {
                background: #d4d4d4;
                border: 1px solid #a6a6a6;
                border-radius: 2px;
                color: #333333;
                padding: 3px 10px;
                margin: 2px 4px;
                font-size: 14px;
                font-weight: 600;
            }
            QTabWidget#ViewPageTabs QToolButton:hover {
                background: #ffffff;
                color: #1f4e79;
                border-color: #1f4e79;
            }
            """
        )
        self._tabs.currentChanged.connect(self._on_tab_changed)
        self._tabs.tabCloseRequested.connect(self._close_view_page)
        self._tabs.tabBarDoubleClicked.connect(self._rename_view_page)
        add_page_btn = QToolButton(self._tabs)
        add_page_btn.setText("+")
        add_page_btn.setToolTip("New view page")
        add_page_btn.clicked.connect(self._new_view_page)
        self._tabs.setCornerWidget(add_page_btn, Qt.Corner.BottomRightCorner)
        self._tabs.addTab(self._make_page_host(), self._pages[0].title)
        views_layout.addWidget(self._tabs)
        self._workspace_stack.addWidget(views_host)
        self._workspace_stack.setCurrentIndex(1)
        workspace_layout.addWidget(self._workspace_stack, stretch=1)
        main_split.addWidget(workspace)
        main_split.setStretchFactor(0, 0)
        main_split.setStretchFactor(1, 1)
        main_split.setSizes([320, 880])
        for i in range(1, main_split.count()):
            handle = main_split.handle(i)
            if handle is not None:
                handle.setCursor(Qt.CursorShape.SplitHCursor)
        self._main_split = main_split
        root.addWidget(main_split)

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar(self))
        self._build_cursor_dock()
        self._build_spectrogram_dock()
        self._build_measure_dock()
        self._build_menu()
        self._rebuild_panels()
        self.add_recording(make_sine_with_noise(), reset_layout=True)

    def _build_cursor_dock(self) -> None:
        dock = QDockWidget("Cursor values", self)
        dock.setObjectName("CursorValuesDock")
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea
            | Qt.DockWidgetArea.TopDockWidgetArea
        )
        body = QWidget()
        layout = QVBoxLayout(body)
        self._cursor_label = QLabel("Cursor: (move mouse on plot)")
        self._cursor_label.setWordWrap(True)
        layout.addWidget(self._cursor_label)
        self._cursor_table = QTableWidget(0, 3)
        self._cursor_table.setHorizontalHeaderLabels(["At", "Series", "Value"])
        self._cursor_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._cursor_table)
        copy_btn = QPushButton("Copy table")
        copy_btn.clicked.connect(self._copy_cursor_table)
        layout.addWidget(copy_btn)
        dock.setWidget(body)
        dock.setFloating(True)
        dock.hide()
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._cursor_dock = dock

    def _build_spectrogram_dock(self) -> None:
        dock = QDockWidget("Spectrogram color scale", self)
        dock.setObjectName("SpectrogramScaleDock")
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea
            | Qt.DockWidgetArea.TopDockWidgetArea
        )
        body = QWidget()
        layout = QVBoxLayout(body)
        hint = QLabel(
            "Z-axis (dB) for the active spectrogram panel. "
            "Set Min / Max, or drag the color-bar handles. "
            "Color map: right-click the color bar."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #444;")
        layout.addWidget(hint)
        self._spec_scale_status = QLabel("No spectrogram panel selected")
        self._spec_scale_status.setStyleSheet("color: #666;")
        layout.addWidget(self._spec_scale_status)

        min_row = QHBoxLayout()
        min_row.addWidget(QLabel("Min dB"))
        self._spec_db_min_spin = QDoubleSpinBox()
        self._spec_db_min_spin.setRange(-300.0, 300.0)
        self._spec_db_min_spin.setDecimals(1)
        self._spec_db_min_spin.setSingleStep(2.0)
        self._spec_db_min_spin.setValue(-80.0)
        self._spec_db_min_spin.setToolTip("Lower end of the color scale (Z min)")
        self._spec_db_min_spin.valueChanged.connect(self._on_spec_levels_changed)
        min_row.addWidget(self._spec_db_min_spin)
        layout.addLayout(min_row)

        max_row = QHBoxLayout()
        max_row.addWidget(QLabel("Max dB"))
        self._spec_db_max_spin = QDoubleSpinBox()
        self._spec_db_max_spin.setRange(-300.0, 300.0)
        self._spec_db_max_spin.setDecimals(1)
        self._spec_db_max_spin.setSingleStep(2.0)
        self._spec_db_max_spin.setValue(0.0)
        self._spec_db_max_spin.setToolTip("Upper end of the color scale (Z max)")
        self._spec_db_max_spin.valueChanged.connect(self._on_spec_levels_changed)
        max_row.addWidget(self._spec_db_max_spin)
        layout.addLayout(max_row)

        btn_row = QHBoxLayout()
        auto_btn = QPushButton("Auto")
        auto_btn.setToolTip("Set Min / Max from data")
        auto_btn.clicked.connect(self._auto_spectrogram_levels)
        btn_row.addWidget(auto_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)
        layout.addStretch(1)

        dock.setWidget(body)
        dock.setFloating(True)
        dock.hide()
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._spectrogram_dock = dock
        self._spec_scale_widgets = (
            self._spec_db_min_spin,
            self._spec_db_max_spin,
            auto_btn,
        )
        self._updating_spec_dock = False

    def _build_measure_dock(self) -> None:
        dock = QDockWidget("Spectrum measure", self)
        dock.setObjectName("SpectrumMeasureDock")
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea
            | Qt.DockWidgetArea.TopDockWidgetArea
        )
        body = QWidget()
        layout = QVBoxLayout(body)
        hint = QLabel(
            "Measure Mag spectra in Views: Linear/dB display, "
            "overall / band RMS (peak·rms amplitude scales), peak pick. "
            "Prefer peak or rms amplitude scale from Analysis."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #444;")
        layout.addWidget(hint)

        disp_row = QHBoxLayout()
        disp_row.addWidget(QLabel("Mag Y"))
        self._mag_linear_btn = QRadioButton("Linear")
        self._mag_db_btn = QRadioButton("dB")
        self._mag_linear_btn.setChecked(True)
        mag_group = QButtonGroup(self)
        mag_group.addButton(self._mag_linear_btn)
        mag_group.addButton(self._mag_db_btn)
        self._mag_linear_btn.toggled.connect(self._on_mag_db_toggled)
        self._mag_db_btn.toggled.connect(self._on_mag_db_toggled)
        disp_row.addWidget(self._mag_linear_btn)
        disp_row.addWidget(self._mag_db_btn)
        disp_row.addStretch(1)
        layout.addLayout(disp_row)

        self._measure_series_label = QLabel("Series: —")
        self._measure_series_label.setStyleSheet("color: #666;")
        layout.addWidget(self._measure_series_label)

        self._measure_overall_label = QLabel("Overall RMS: —")
        layout.addWidget(self._measure_overall_label)
        self._measure_band_label = QLabel("Band RMS: —")
        layout.addWidget(self._measure_band_label)

        self._measure_use_band_cb = QCheckBox("Use band (f min / f max)")
        self._measure_use_band_cb.setChecked(False)
        self._measure_use_band_cb.toggled.connect(self._on_measure_band_changed)
        layout.addWidget(self._measure_use_band_cb)

        band_row = QHBoxLayout()
        band_row.addWidget(QLabel("f min"))
        self._measure_f_lo_spin = QDoubleSpinBox()
        self._measure_f_lo_spin.setRange(0.0, 1e9)
        self._measure_f_lo_spin.setDecimals(3)
        self._measure_f_lo_spin.setSuffix(" Hz")
        self._measure_f_lo_spin.valueChanged.connect(self._on_measure_band_changed)
        band_row.addWidget(self._measure_f_lo_spin)
        band_row.addWidget(QLabel("f max"))
        self._measure_f_hi_spin = QDoubleSpinBox()
        self._measure_f_hi_spin.setRange(0.0, 1e9)
        self._measure_f_hi_spin.setDecimals(3)
        self._measure_f_hi_spin.setSuffix(" Hz")
        self._measure_f_hi_spin.valueChanged.connect(self._on_measure_band_changed)
        band_row.addWidget(self._measure_f_hi_spin)
        layout.addLayout(band_row)

        from_markers_btn = QPushButton("Band from V1 / V2")
        from_markers_btn.setToolTip(
            "Set f min / f max from the first two vertical markers"
        )
        from_markers_btn.clicked.connect(self._measure_band_from_markers)
        layout.addWidget(from_markers_btn)

        peak_row = QHBoxLayout()
        peak_row.addWidget(QLabel("Peaks"))
        self._measure_n_peaks_spin = QSpinBox()
        self._measure_n_peaks_spin.setRange(1, 50)
        self._measure_n_peaks_spin.setValue(5)
        peak_row.addWidget(self._measure_n_peaks_spin)
        pick_btn = QPushButton("Pick peaks")
        pick_btn.setToolTip("Find local maxima and place vertical markers")
        pick_btn.clicked.connect(self._pick_spectrum_peaks)
        peak_row.addWidget(pick_btn)
        peak_row.addStretch(1)
        layout.addLayout(peak_row)

        self._measure_peaks_table = QTableWidget(0, 3)
        self._measure_peaks_table.setHorizontalHeaderLabels(["f [Hz]", "Mag", "Unit"])
        self._measure_peaks_table.horizontalHeader().setStretchLastSection(True)
        self._measure_peaks_table.setMaximumHeight(160)
        layout.addWidget(self._measure_peaks_table)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_measure_dock)
        layout.addWidget(refresh_btn)
        layout.addStretch(1)

        dock.setWidget(body)
        dock.setFloating(True)
        dock.hide()
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._measure_dock = dock

    # --- workspace mode + view pages -------------------------------------

    def _is_views_mode(self) -> bool:
        return self._workspace_stack.currentIndex() == 1

    def _set_workspace_mode(self, index: int) -> None:
        index = int(index)
        if index == self._workspace_stack.currentIndex():
            return
        if index == 0:
            # Leaving Views: stash the live plot page.
            if 0 <= self._page_index < len(self._pages):
                self._stash_live_to_page(self._page_index)
            self._workspace_stack.setCurrentIndex(0)
            self._ribbon_analysis_btn.setChecked(True)
            self._analysis_page.refresh_series()
            return

        self._workspace_stack.setCurrentIndex(1)
        self._ribbon_views_btn.setChecked(True)
        # Entering Views: ensure current page is active.
        if 0 <= self._page_index < len(self._pages):
            page = self._pages[self._page_index]
            if not page.built or page.root_plot_widget is None:
                self._activate_page(self._page_index)
            else:
                self._panels = page.panels
                self._synced_markers = page.synced_markers
                self._selected_marker = page.selected_marker
                self._active_panel = page.active_panel
                self._root_plot_widget = page.root_plot_widget
                self._node_widgets = page.node_widgets
                self._mouse_proxies = page.mouse_proxies
                self._workspace = page.workspace
                self._display_mode = page.display_mode
                self._mag_db = page.mag_db
                self._measure_use_band = page.measure_use_band
                self._measure_f_lo = page.measure_f_lo
                self._measure_f_hi = page.measure_f_hi
                self._sync_measure_dock_controls()

    def _on_mode_changed(self, index: int) -> None:
        self._set_workspace_mode(index)

    def _plot_host_for_page(self, page_index: int) -> QWidget | None:
        if page_index < 0 or page_index >= self._tabs.count():
            return None
        return self._tabs.widget(page_index)

    def _make_page_host(self) -> QWidget:
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        return host

    def _current_host_layout(self) -> QVBoxLayout:
        host = self._plot_host_for_page(self._page_index)
        assert host is not None
        layout = host.layout()
        assert isinstance(layout, QVBoxLayout)
        return layout

    def _stash_live_to_page(self, index: int) -> None:
        """Keep this page's widgets alive; remember refs for later restore."""
        if index < 0 or index >= len(self._pages):
            return
        self._sync_workspace_from_panels()
        page = self._pages[index]
        page.display_mode = self._display_mode
        page.mag_db = self._mag_db
        page.measure_use_band = self._measure_use_band
        page.measure_f_lo = self._measure_f_lo
        page.measure_f_hi = self._measure_f_hi
        if 0 <= index < self._tabs.count():
            page.title = self._tabs.tabText(index)
        page.panels = self._panels
        page.synced_markers = self._synced_markers
        page.selected_marker = self._selected_marker
        page.active_panel = self._active_panel
        page.root_plot_widget = self._root_plot_widget
        page.node_widgets = self._node_widgets
        page.mouse_proxies = self._mouse_proxies
        page.built = self._root_plot_widget is not None

    def _clear_live_refs(self) -> None:
        self._panels = []
        self._synced_markers = []
        self._selected_marker = None
        self._active_panel = 0
        self._root_plot_widget = None
        self._node_widgets = []
        self._mouse_proxies = []

    def _destroy_page_runtime(self, page: _PageState) -> None:
        """Destroy widgets belonging to a page (e.g. on close)."""
        for group in page.synced_markers:
            for line in group.lines:
                for panel in page.panels:
                    try:
                        panel.plot.removeItem(line)
                    except Exception:  # noqa: BLE001
                        pass
        if page.root_plot_widget is not None:
            page.root_plot_widget.setParent(None)
            page.root_plot_widget.deleteLater()
        page.panels = []
        page.synced_markers = []
        page.selected_marker = None
        page.active_panel = 0
        page.root_plot_widget = None
        page.node_widgets = []
        page.mouse_proxies = []
        page.built = False

    def _apply_spectrum_menu(self, mode: str) -> None:
        for key, act in self._spectrum_actions.items():
            act.setChecked(key == mode)
        if hasattr(self, "_mag_db_action"):
            self._mag_db_action.blockSignals(True)
            self._mag_db_action.setChecked(bool(self._mag_db))
            self._mag_db_action.blockSignals(False)

    def _set_workspace(self, layout: WorkspaceLayout) -> None:
        self._workspace = layout
        if 0 <= self._page_index < len(self._pages):
            self._pages[self._page_index].workspace = layout

    def _activate_page(self, index: int) -> None:
        page = self._pages[index]
        self._page_index = index
        self._workspace = page.workspace
        self._display_mode = page.display_mode
        self._mag_db = bool(page.mag_db)
        self._measure_use_band = bool(page.measure_use_band)
        self._measure_f_lo = float(page.measure_f_lo)
        self._measure_f_hi = float(page.measure_f_hi)
        self._apply_spectrum_menu(page.display_mode)
        self._sync_measure_dock_controls()

        if page.built and page.root_plot_widget is not None:
            self._panels = page.panels
            self._synced_markers = page.synced_markers
            self._selected_marker = page.selected_marker
            self._active_panel = page.active_panel
            self._root_plot_widget = page.root_plot_widget
            self._node_widgets = page.node_widgets
            self._mouse_proxies = page.mouse_proxies
            self._prune_workspace_series()
            if not self._spectrum_mode():
                for panel in self._panels:
                    leaf = panel.layout_leaf
                    if leaf is None:
                        continue
                    panel.series_ids = list(leaf.series_ids)
                    panel.series_colors = {
                        k: v
                        for k, v in leaf.series_colors.items()
                        if k in panel.series_ids
                    }
            self._update_active_panel_label()
            self._sync_layout_menu_enabled()
            if self._workspace.preset_id:
                self._select_preset_in_menu(self._workspace.preset_id)
            # Refresh data if project changed; keep zoom/pan.
            self._redraw_curves(fit=False)
            self._refresh_marker_readout()
            self._update_meta()
            return

        self._clear_live_refs()
        self._rebuild_panels()
        self._redraw_curves(fit=not page.pending_view_ranges)
        for vertical, value in page.pending_markers:
            self.add_marker(vertical=vertical, value=value)
        if page.pending_view_ranges:
            for panel, rng in zip(self._panels, page.pending_view_ranges):
                x = rng.get("x")
                y = rng.get("y")
                if x is None or y is None:
                    continue
                vb = panel.plot.getViewBox()
                vb.setRange(xRange=x, yRange=y, padding=0)
                vb.disableAutoRange()
        page.pending_markers = []
        page.pending_view_ranges = []
        page.synced_markers = self._synced_markers
        self._stash_live_to_page(index)
        self._update_meta()

    def _on_tab_changed(self, index: int) -> None:
        if self._suppress_tab_change or index < 0:
            return
        old = self._page_index
        if old == index:
            return
        if 0 <= old < len(self._pages):
            self._stash_live_to_page(old)
        self._activate_page(index)

    def _show_analysis_page(self) -> None:
        self._set_workspace_mode(0)

    def _show_views_workspace(self) -> None:
        self._set_workspace_mode(1)

    def _close_current_view_page(self) -> None:
        self._show_views_workspace()
        self._close_view_page(self._tabs.currentIndex())

    def _rename_current_view_page(self) -> None:
        self._show_views_workspace()
        self._rename_view_page(self._tabs.currentIndex())

    def _on_spectrum_result(self, result: str | list[str]) -> None:
        self._refresh_project_tree()
        self._analysis_page.refresh_series()
        ids = result if isinstance(result, list) else [result]
        if not ids:
            return
        first = self._project.get(ids[0])
        dataset = first.source_label if first is not None else "FFT result"
        if len(ids) == 1:
            name = first.name if first is not None else ids[0]
            msg = f"Created “{dataset}” / {name} — switch to Views to plot"
        else:
            msg = (
                f"Created dataset “{dataset}” ({len(ids)} spectra)"
                " — switch to Views to plot"
            )
        saved = self._autosave_project_if_path_known()
        if saved:
            msg += f" · saved {self._project.path.name}"
        self.statusBar().showMessage(msg, 6000)

    def _autosave_project_if_path_known(self) -> bool:
        """Save in place when the project already has a .ginkyo path. New projects skip."""
        if self._project.path is None:
            return False
        try:
            self._project.save(views=self._serialize_views())
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(
                self,
                "Auto-save after analysis failed",
                f"Analysis results are in memory, but saving failed:\n{exc}",
            )
            return False
        return True

    def _new_view_page(self) -> None:
        self._show_views_workspace()
        if 0 <= self._page_index < len(self._pages):
            self._stash_live_to_page(self._page_index)
        n = len(self._pages) + 1
        page = _PageState(title=f"Page {n}", workspace=WorkspaceLayout())
        self._pages.append(page)
        host = self._make_page_host()
        self._suppress_tab_change = True
        tab = self._tabs.addTab(host, page.title)
        self._tabs.setCurrentIndex(tab)
        self._suppress_tab_change = False
        self._activate_page(len(self._pages) - 1)
        self.statusBar().showMessage(f"New page: {page.title}", 3000)

    def _close_view_page(self, index: int) -> None:
        if len(self._pages) <= 1:
            self.statusBar().showMessage("Cannot close the last page", 3000)
            return
        if index < 0 or index >= len(self._pages):
            return
        closing_current = index == self._page_index
        if closing_current:
            self._stash_live_to_page(index)
        page = self._pages[index]
        if closing_current:
            # MainWindow still holds the same lists; clear without double-free.
            self._clear_live_refs()
            page.panels = []
            page.synced_markers = []
            page.selected_marker = None
            page.node_widgets = []
            page.mouse_proxies = []
            if page.root_plot_widget is not None:
                page.root_plot_widget.setParent(None)
                page.root_plot_widget.deleteLater()
                page.root_plot_widget = None
            page.built = False
        else:
            self._destroy_page_runtime(page)

        self._suppress_tab_change = True
        self._tabs.removeTab(index)
        del self._pages[index]
        if self._page_index > index:
            self._page_index -= 1
        elif self._page_index == index:
            self._page_index = min(index, len(self._pages) - 1)
        new_index = self._tabs.currentIndex()
        self._suppress_tab_change = False
        if closing_current:
            self._activate_page(new_index)
        elif new_index != self._page_index:
            self._page_index = new_index
        self.statusBar().showMessage("Page closed", 3000)

    def _rename_view_page(self, index: int) -> None:
        if index < 0 or index >= len(self._pages):
            return
        current = self._tabs.tabText(index)
        text, ok = QInputDialog.getText(self, "Rename page", "Page name:", text=current)
        if not ok:
            return
        name = text.strip() or current
        self._pages[index].title = name
        self._tabs.setTabText(index, name)

    # --- panels / layout -------------------------------------------------

    def apply_layout_preset(self, preset_id: str) -> None:
        if self._spectrum_mode():
            return
        self._sync_workspace_from_panels()
        # Keep each panel's overlay as-is; extra panels stay empty (do not redistribute).
        old = [
            (list(leaf.series_ids), dict(leaf.series_colors), leaf.view_kind)
            for leaf in self._workspace.leaves()
        ]
        lay = build_preset(preset_id, [])
        for i, leaf in enumerate(lay.leaves()):
            if i < len(old):
                ids, colors, view_kind = old[i]
                leaf.series_ids = list(ids)
                leaf.series_colors = {k: v for k, v in colors.items() if k in ids}
                leaf.view_kind = view_kind
            else:
                leaf.series_ids = []
                leaf.series_colors = {}
                leaf.view_kind = "time"
        lay.display_mode = _MODE_TIME
        self._set_workspace(lay)
        self._display_mode = _MODE_TIME
        if 0 <= self._page_index < len(self._pages):
            self._pages[self._page_index].display_mode = _MODE_TIME
        for key, act in self._spectrum_actions.items():
            act.setChecked(key == _MODE_TIME)
        self._rebuild_panels()
        self._redraw_curves(fit=True)
        self._update_active_panel_label()
        self._select_preset_in_menu(preset_id)
        self.statusBar().showMessage(f"Layout: {preset_label(preset_id)}", 3000)

    def _select_preset_in_menu(self, preset_id: str) -> None:
        act = self._layout_actions.get(preset_id)
        if act is not None:
            act.setChecked(True)

    def remove_series_from_panel(self, panel: _PlotPanel, series_id: str) -> None:
        if series_id not in panel.series_ids:
            return
        panel.series_ids = [s for s in panel.series_ids if s != series_id]
        panel.series_colors.pop(series_id, None)
        leaf = panel.layout_leaf
        new_kind = self._view_kind_for_series_ids(panel.series_ids)
        kind_changed = (
            leaf is not None
            and not self._spectrum_mode()
            and leaf.view_kind != new_kind
        )
        self._write_panel_assignment(panel)
        if leaf is not None and not self._spectrum_mode():
            leaf.view_kind = new_kind
        if kind_changed:
            self._rebuild_panels()
        self._redraw_curves(fit=False)
        self.statusBar().showMessage("Removed series from panel", 3000)

    def clear_panel_series(self, panel: _PlotPanel) -> None:
        panel.series_ids = []
        panel.series_colors.clear()
        leaf = panel.layout_leaf
        kind_changed = (
            leaf is not None
            and not self._spectrum_mode()
            and leaf.view_kind != "time"
        )
        self._write_panel_assignment(panel)
        if leaf is not None and not self._spectrum_mode():
            leaf.view_kind = "time"
        if kind_changed:
            self._rebuild_panels()
        self._redraw_curves(fit=False)
        self.statusBar().showMessage("Cleared panel series", 3000)

    def set_panel_series_color(self, panel: _PlotPanel, series_id: str) -> None:
        from PySide6.QtGui import QColor

        current = self._color_for_panel_series(panel, series_id)
        color = QColorDialog.getColor(QColor(current), self, "Series color")
        if not color.isValid():
            return
        panel.series_colors[series_id] = color.name()
        self._write_panel_assignment(panel)
        self._redraw_curves(fit=False)

    def reset_panel_series_color(self, panel: _PlotPanel, series_id: str) -> None:
        panel.series_colors.pop(series_id, None)
        self._write_panel_assignment(panel)
        self._redraw_curves(fit=False)

    def _color_for_panel_series(self, panel: _PlotPanel, series_id: str) -> str:
        override = panel.series_colors.get(series_id)
        if override:
            return override
        try:
            idx = panel.series_ids.index(series_id)
        except ValueError:
            idx = 0
        return _PENS[idx % len(_PENS)]

    def _write_panel_assignment(self, panel: _PlotPanel) -> None:
        try:
            idx = self._panels.index(panel)
        except ValueError:
            return
        self._active_panel = idx
        if self._spectrum_mode():
            leaves = self._workspace.leaves()
            if leaves:
                leaves[0].series_ids = list(panel.series_ids)
                leaves[0].series_colors = {
                    k: v for k, v in panel.series_colors.items() if k in panel.series_ids
                }
            for p in self._panels:
                p.series_ids = list(panel.series_ids)
                p.series_colors = dict(panel.series_colors)
            return
        leaf = panel.layout_leaf
        if leaf is None:
            leaves = self._workspace.leaves()
            # Fallback: first panel per leaf (mag of a pair) by unique layout_leaf
            return
        leaf.series_ids = list(panel.series_ids)
        leaf.series_colors = {
            k: v for k, v in panel.series_colors.items() if k in panel.series_ids
        }
        for p in self._panels:
            if p.layout_leaf is leaf:
                p.series_ids = list(panel.series_ids)
                p.series_colors = dict(panel.series_colors)

    def _view_kind_for_series_ids(self, series_ids: list[str]) -> str:
        for sid in series_ids:
            series = self._project.get(sid)
            if series is not None and series.is_spectrogram():
                return _VIEW_SPECTROGRAM
        for sid in series_ids:
            series = self._project.get(sid)
            if series is not None and series.is_spectrum():
                return "mag_phase"
        return "time"

    def assign_series_to_panel(
        self, panel: _PlotPanel, series_ids: list[str], *, replace: bool = False
    ) -> None:
        valid = self._project.prune_ids(list(series_ids))
        if not valid:
            return
        try:
            idx = self._panels.index(panel)
        except ValueError:
            return
        self._active_panel = idx
        if replace:
            panel.series_ids = valid
        else:
            merged = list(panel.series_ids)
            for sid in valid:
                if sid not in merged:
                    merged.append(sid)
            panel.series_ids = merged
        leaf = panel.layout_leaf
        new_kind = self._view_kind_for_series_ids(panel.series_ids)
        kind_changed = (
            leaf is not None
            and not self._spectrum_mode()
            and leaf.view_kind != new_kind
        )
        self._write_panel_assignment(panel)
        if leaf is not None and not self._spectrum_mode():
            leaf.view_kind = new_kind
        self._update_active_panel_label()
        if kind_changed:
            self._rebuild_panels()
        self._redraw_curves(fit=False)
        self.statusBar().showMessage(
            ("Replaced" if replace else "Added")
            + f" {len(valid)} series on panel {idx + 1}"
            + (
                " → Spectrogram"
                if new_kind == _VIEW_SPECTROGRAM
                else (" → Mag+Phase" if new_kind == "mag_phase" else "")
            ),
            3000,
        )

    def _panel_for_viewbox(self, vb) -> _PlotPanel | None:  # noqa: ANN001
        for panel in self._panels:
            if panel.plot.getViewBox() is vb:
                return panel
        return None

    def _update_active_panel_label(self) -> None:
        self._refresh_spectrogram_dock()

    def _tear_down_plots(self, host: QWidget | None) -> None:
        self.clear_markers()
        self._mouse_proxies.clear()
        if self._root_plot_widget is not None:
            if host is not None:
                layout = host.layout()
                if layout is not None:
                    layout.removeWidget(self._root_plot_widget)
            self._root_plot_widget.setParent(None)
            self._root_plot_widget.deleteLater()
            self._root_plot_widget = None
        self._panels.clear()
        self._node_widgets.clear()
        if 0 <= self._page_index < len(self._pages):
            page = self._pages[self._page_index]
            page.panels = []
            page.synced_markers = []
            page.selected_marker = None
            page.root_plot_widget = None
            page.node_widgets = []
            page.mouse_proxies = []
            page.built = False

    def _rebuild_panels(self) -> None:
        host = self._plot_host_for_page(self._page_index)
        self._tear_down_plots(host)
        self._active_panel = 0
        self._prune_workspace_series()

        if self._display_mode == _MODE_MAG_PHASE:
            ids = list(self._workspace.leaves()[0].series_ids) if self._workspace.leaves() else []
            tree = LayoutNode.split(
                "vertical",
                LayoutNode.leaf(ids),
                LayoutNode.leaf(ids),
            )
            widget = self._build_widget_from_node(tree, y_kinds=["mag", "phase"])
        elif self._display_mode == _MODE_REAL_IMAG:
            ids = list(self._workspace.leaves()[0].series_ids) if self._workspace.leaves() else []
            tree = LayoutNode.split(
                "vertical",
                LayoutNode.leaf(ids),
                LayoutNode.leaf(ids),
            )
            widget = self._build_widget_from_node(tree, y_kinds=["real", "imag"])
        else:
            widget = self._build_widget_from_node(self._workspace.root, y_kinds=None)

        self._current_host_layout().addWidget(widget)
        self._root_plot_widget = widget

        self._apply_axis_links()

        self._wire_mouse_proxies()
        self._update_active_panel_label()
        self._sync_layout_menu_enabled()
        if self._workspace.preset_id:
            self._select_preset_in_menu(self._workspace.preset_id)
        self._stash_live_to_page(self._page_index)

    def _apply_axis_links(self) -> None:
        """Link / unlink X and Y axes across panels according to toggles."""
        if not self._panels:
            return
        first = self._panels[0].plot
        first.setXLink(None)
        first.setYLink(None)
        for panel in self._panels[1:]:
            panel.plot.setXLink(first if self._link_x else None)
            panel.plot.setYLink(first if self._link_y else None)

    def _set_link_x(self, enabled: bool) -> None:
        self._link_x = bool(enabled)
        self._apply_axis_links()
        state = "on" if self._link_x else "off"
        self.statusBar().showMessage(f"X-axis link: {state}", 3000)

    def _set_link_y(self, enabled: bool) -> None:
        self._link_y = bool(enabled)
        self._apply_axis_links()
        state = "on" if self._link_y else "off"
        self.statusBar().showMessage(f"Y-axis link: {state}", 3000)

    def _build_widget_from_node(
        self,
        node: LayoutNode,
        *,
        y_kinds: list[str] | None,
        leaf_counter: list[int] | None = None,
    ) -> QWidget:
        if leaf_counter is None:
            leaf_counter = [0]
        if node.is_leaf():
            if y_kinds is None and (node.view_kind or "time") == "mag_phase":
                splitter = QSplitter(Qt.Orientation.Vertical)
                splitter.setChildrenCollapsible(False)
                splitter.setHandleWidth(6)
                mag = self._make_leaf_panel(
                    series_ids=list(node.series_ids),
                    series_colors=dict(node.series_colors),
                    y_kind="mag",
                    layout_leaf=node,
                )
                phase = self._make_leaf_panel(
                    series_ids=list(node.series_ids),
                    series_colors=dict(node.series_colors),
                    y_kind="phase",
                    layout_leaf=node,
                )
                self._panels.append(mag)
                self._panels.append(phase)
                splitter.addWidget(mag.shell)
                splitter.addWidget(phase.shell)
                splitter.setSizes([500, 500])
                self._node_widgets.append((node, splitter))
                leaf_counter[0] += 1
                return splitter

            if y_kinds is None and (node.view_kind or "time") == _VIEW_SPECTROGRAM:
                panel = self._make_leaf_panel(
                    series_ids=list(node.series_ids),
                    series_colors=dict(node.series_colors),
                    y_kind=_VIEW_SPECTROGRAM,
                    layout_leaf=node,
                )
                self._panels.append(panel)
                self._node_widgets.append((node, panel.shell))
                leaf_counter[0] += 1
                return panel.shell

            idx = leaf_counter[0]
            leaf_counter[0] += 1
            if y_kinds is not None and idx < len(y_kinds):
                y_kind = y_kinds[idx]
            else:
                y_kind = "time"
            panel = self._make_leaf_panel(
                series_ids=list(node.series_ids),
                series_colors=dict(node.series_colors),
                y_kind=y_kind,
                layout_leaf=node if y_kinds is None else None,
            )
            self._panels.append(panel)
            self._node_widgets.append((node, panel.shell))
            return panel.shell

        orient = (
            Qt.Orientation.Horizontal
            if node.orientation == "horizontal"
            else Qt.Orientation.Vertical
        )
        splitter = QSplitter(orient)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        for child in node.children:
            splitter.addWidget(
                self._build_widget_from_node(child, y_kinds=y_kinds, leaf_counter=leaf_counter)
            )
        self._node_widgets.append((node, splitter))
        if node.sizes and len(node.sizes) == splitter.count():
            total = sum(node.sizes) or 1.0
            base = 1000
            splitter.setSizes([max(50, int(base * s / total)) for s in node.sizes])
        return splitter

    def _make_leaf_panel(
        self,
        *,
        series_ids: list[str],
        series_colors: dict[str, str] | None = None,
        y_kind: str,
        layout_leaf: LayoutNode | None = None,
    ) -> _PlotPanel:
        vb = _NagilizeViewBox()
        plot_widget = pg.PlotWidget(
            viewBox=vb,
            axisItems={
                "bottom": _AutoFitAxis(orientation="bottom"),
                "left": _AutoFitAxis(orientation="left"),
            },
        )
        plot_widget.setBackground("w")
        plot = plot_widget.getPlotItem()
        plot.showGrid(x=True, y=True, alpha=0.3)
        plot.addLegend()
        vb.setMouseMode(pg.ViewBox.PanMode)
        vb.attach_host(self)
        plot.ctrl.fftCheck.setChecked(False)
        plot.ctrl.fftCheck.setEnabled(False)

        v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#888", width=1))
        v_line.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        v_line.setZValue(50)
        plot.addItem(v_line, ignoreBounds=True)

        shell = PanelShell(self, plot_widget)
        colors = {
            k: v
            for k, v in (series_colors or {}).items()
            if k in series_ids
        }
        panel = _PlotPanel(
            plot=plot,
            plot_widget=plot_widget,
            shell=shell,
            v_line=v_line,
            series_ids=list(series_ids),
            series_colors=colors,
            y_kind=y_kind,
            layout_leaf=layout_leaf,
        )
        shell.panel_ref = panel
        shell.set_edge_buttons_enabled(y_kind in ("time", _VIEW_SPECTROGRAM))
        return panel

    def _wire_mouse_proxies(self) -> None:
        self._mouse_proxies.clear()
        for panel in self._panels:
            scene = panel.plot.scene()
            if scene is None:
                continue
            proxy = pg.SignalProxy(
                scene.sigMouseMoved,
                rateLimit=60,
                slot=self._on_mouse_moved,
            )
            self._mouse_proxies.append(proxy)
            scene.sigMouseClicked.connect(self._on_scene_clicked)

    def _prune_workspace_series(self) -> None:
        self._workspace.prune_series(self._project.prune_ids)
        for panel in self._panels:
            panel.series_ids = self._project.prune_ids(panel.series_ids)
            panel.series_colors = {
                k: v for k, v in panel.series_colors.items() if k in panel.series_ids
            }

    def _sync_splitter_sizes_to_tree(self) -> None:
        for node, widget in self._node_widgets:
            if node.is_leaf() or not isinstance(widget, QSplitter):
                continue
            sizes = widget.sizes()
            total = float(sum(sizes)) or 1.0
            node.sizes = [s / total for s in sizes]

    def _active(self) -> _PlotPanel | None:
        if not self._panels:
            return None
        idx = int(np.clip(self._active_panel, 0, len(self._panels) - 1))
        return self._panels[idx]

    def _set_active_panel_from_viewbox(self, vb: pg.ViewBox) -> None:
        for i, panel in enumerate(self._panels):
            if panel.plot.getViewBox() is vb:
                if self._active_panel != i:
                    self._active_panel = i
                    self._update_active_panel_label()
                return

    def _set_display_mode(self, mode: str) -> None:
        if mode == self._display_mode:
            return
        self._display_mode = mode
        self._workspace.display_mode = mode
        if 0 <= self._page_index < len(self._pages):
            self._pages[self._page_index].display_mode = mode
        self._rebuild_panels()
        self._redraw_curves(fit=True)
        self._sync_marker_labels()
        self._update_meta()

    def _sync_layout_menu_enabled(self) -> None:
        time_mode = self._display_mode == _MODE_TIME
        for act in self._layout_actions.values():
            act.setEnabled(time_mode)

    def _spectrum_mode(self) -> bool:
        return self._display_mode != _MODE_TIME

    # --- menu ------------------------------------------------------------

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        open_proj_act = QAction("Open project…", self)
        open_proj_act.setShortcut("Ctrl+Shift+O")
        open_proj_act.triggered.connect(self.open_project_dialog)
        file_menu.addAction(open_proj_act)
        save_proj_act = QAction("Save project…", self)
        save_proj_act.setShortcut(QKeySequence.Save)
        save_proj_act.triggered.connect(self.save_project_dialog)
        file_menu.addAction(save_proj_act)
        file_menu.addSeparator()

        open_act = QAction("Add file…", self)
        open_act.setShortcut(QKeySequence.Open)
        open_act.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_act)

        export_act = QAction("Export selected source CSV…", self)
        export_act.triggered.connect(self.export_csv_dialog)
        file_menu.addAction(export_act)

        file_menu.addSeparator()
        remove_act = QAction("Remove selected source", self)
        remove_act.triggered.connect(self.remove_selected_source)
        file_menu.addAction(remove_act)
        clear_act = QAction("Clear project", self)
        clear_act.triggered.connect(self.clear_project)
        file_menu.addAction(clear_act)

        file_menu.addSeparator()
        dummy_act = QAction("Add dummy sine", self)
        dummy_act.triggered.connect(self.load_dummy)
        file_menu.addAction(dummy_act)
        file_menu.addSeparator()
        quit_act = QAction("Quit", self)
        quit_act.setShortcut(QKeySequence.Quit)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        analyze_menu = self.menuBar().addMenu("&Analyze")
        spectrum_act = QAction("Spectrum (FFT)…", self)
        spectrum_act.setShortcut("Ctrl+Shift+F")
        spectrum_act.triggered.connect(self._show_analysis_page)
        analyze_menu.addAction(spectrum_act)

        view_menu = self.menuBar().addMenu("&View")
        new_page_act = QAction("New page", self)
        new_page_act.setShortcut("Ctrl+T")
        new_page_act.triggered.connect(self._new_view_page)
        view_menu.addAction(new_page_act)
        close_page_act = QAction("Close page", self)
        close_page_act.setShortcut("Ctrl+W")
        close_page_act.triggered.connect(self._close_current_view_page)
        view_menu.addAction(close_page_act)
        rename_page_act = QAction("Rename page…", self)
        rename_page_act.triggered.connect(self._rename_current_view_page)
        view_menu.addAction(rename_page_act)
        view_menu.addSeparator()

        cursor_dock_act = QAction("Cursor values", self)
        cursor_dock_act.setCheckable(True)
        cursor_dock_act.setChecked(False)
        cursor_dock_act.toggled.connect(self._toggle_cursor_dock)
        view_menu.addAction(cursor_dock_act)
        self._cursor_dock_action = cursor_dock_act

        spec_dock_act = QAction("Spectrogram color scale", self)
        spec_dock_act.setCheckable(True)
        spec_dock_act.setChecked(False)
        spec_dock_act.toggled.connect(self._toggle_spectrogram_dock)
        view_menu.addAction(spec_dock_act)
        self._spectrogram_dock_action = spec_dock_act

        measure_dock_act = QAction("Spectrum measure", self)
        measure_dock_act.setCheckable(True)
        measure_dock_act.setChecked(False)
        measure_dock_act.toggled.connect(self._toggle_measure_dock)
        view_menu.addAction(measure_dock_act)
        self._measure_dock_action = measure_dock_act
        view_menu.addSeparator()

        reset_act = QAction("Reset zoom", self)
        reset_act.setShortcut("Ctrl+0")
        reset_act.triggered.connect(self.reset_zoom)
        view_menu.addAction(reset_act)

        link_x_act = QAction("Link X axes", self)
        link_x_act.setCheckable(True)
        link_x_act.setChecked(self._link_x)
        link_x_act.toggled.connect(self._set_link_x)
        view_menu.addAction(link_x_act)
        self._link_x_action = link_x_act
        link_y_act = QAction("Link Y axes", self)
        link_y_act.setCheckable(True)
        link_y_act.setChecked(self._link_y)
        link_y_act.toggled.connect(self._set_link_y)
        view_menu.addAction(link_y_act)
        self._link_y_action = link_y_act

        view_menu.addSeparator()
        save_layout_act = QAction("Save layout…", self)
        save_layout_act.triggered.connect(self._save_layout_dialog)
        view_menu.addAction(save_layout_act)
        load_layout_act = QAction("Load layout…", self)
        load_layout_act.triggered.connect(self._load_layout_dialog)
        view_menu.addAction(load_layout_act)

        layout_menu = view_menu.addMenu("Layout")
        layout_group = QActionGroup(self)
        layout_group.setExclusive(True)
        for pid in preset_ids():
            act = QAction(preset_label(pid), self)
            act.setCheckable(True)
            act.triggered.connect(lambda checked=False, p=pid: self.apply_layout_preset(p))
            layout_group.addAction(act)
            layout_menu.addAction(act)
            self._layout_actions[pid] = act
        if "single" in self._layout_actions:
            self._layout_actions["single"].setChecked(True)

        spectrum_menu = view_menu.addMenu("Spectrum")
        spectrum_group = QActionGroup(self)
        spectrum_group.setExclusive(True)
        for key, label in (
            (_MODE_TIME, "Off (Time)"),
            (_MODE_MAG_PHASE, "Magnitude + Phase"),
            (_MODE_REAL_IMAG, "Real + Imaginary"),
        ):
            act = QAction(label, self)
            act.setCheckable(True)
            act.setChecked(key == _MODE_TIME)
            act.triggered.connect(lambda checked=False, k=key: self._set_display_mode(k))
            spectrum_group.addAction(act)
            spectrum_menu.addAction(act)
            self._spectrum_actions[key] = act

        mag_db_act = QAction("Magnitude in dB", self)
        mag_db_act.setCheckable(True)
        mag_db_act.setChecked(False)
        mag_db_act.toggled.connect(self._set_mag_db_from_menu)
        spectrum_menu.addSeparator()
        spectrum_menu.addAction(mag_db_act)
        self._mag_db_action = mag_db_act

        pick_peaks_act = QAction("Pick spectrum peaks…", self)
        pick_peaks_act.triggered.connect(self._pick_spectrum_peaks)
        spectrum_menu.addAction(pick_peaks_act)

        view_menu.addSeparator()
        add_v = QAction("Add vertical line", self)
        add_v.setShortcut("V")
        add_v.triggered.connect(lambda: self.add_marker(vertical=True))
        view_menu.addAction(add_v)
        add_h = QAction("Add horizontal line", self)
        add_h.setShortcut("H")
        add_h.triggered.connect(lambda: self.add_marker(vertical=False))
        view_menu.addAction(add_h)
        clear_m = QAction("Clear marker lines", self)
        clear_m.setShortcut("Shift+X")
        clear_m.triggered.connect(self.clear_markers)
        view_menu.addAction(clear_m)
        remove_near = QAction("Remove nearest marker", self)
        remove_near.setShortcuts([QKeySequence.Delete, QKeySequence.Backspace])
        remove_near.triggered.connect(self.remove_nearest_marker)
        view_menu.addAction(remove_near)

    def load_dummy(self) -> None:
        self.add_recording(make_sine_with_noise(), reset_layout=False)

    def _toggle_cursor_dock(self, visible: bool) -> None:
        self._cursor_dock.setVisible(visible)
        if visible:
            self._cursor_dock.raise_()

    def _toggle_spectrogram_dock(self, visible: bool) -> None:
        self._spectrogram_dock.setVisible(visible)
        if visible:
            self._spectrogram_dock.raise_()
            self._refresh_spectrogram_dock()

    def _toggle_measure_dock(self, visible: bool) -> None:
        self._measure_dock.setVisible(visible)
        if visible:
            self._measure_dock.raise_()
            self._refresh_measure_dock()

    def _sync_measure_dock_controls(self) -> None:
        if not hasattr(self, "_measure_f_lo_spin"):
            return
        self._updating_measure_dock = True
        try:
            self._mag_linear_btn.setChecked(not self._mag_db)
            self._mag_db_btn.setChecked(self._mag_db)
            if hasattr(self, "_mag_db_action"):
                self._mag_db_action.blockSignals(True)
                self._mag_db_action.setChecked(self._mag_db)
                self._mag_db_action.blockSignals(False)
            self._measure_use_band_cb.setChecked(self._measure_use_band)
            self._measure_f_lo_spin.setValue(float(self._measure_f_lo))
            self._measure_f_hi_spin.setValue(float(self._measure_f_hi))
        finally:
            self._updating_measure_dock = False

    def _on_mag_db_toggled(self, _checked: bool = False) -> None:
        if self._updating_measure_dock:
            return
        self._set_mag_db(bool(self._mag_db_btn.isChecked()))

    def _set_mag_db_from_menu(self, checked: bool) -> None:
        self._set_mag_db(bool(checked))

    def _set_mag_db(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._mag_db:
            self._sync_measure_dock_controls()
            return
        self._mag_db = enabled
        if 0 <= self._page_index < len(self._pages):
            self._pages[self._page_index].mag_db = enabled
        self._sync_measure_dock_controls()
        self._redraw_curves(fit=False)
        self._refresh_cursor_values_panel()
        self._refresh_measure_dock()

    def _on_measure_band_changed(self, *_args) -> None:  # noqa: ANN002
        if self._updating_measure_dock:
            return
        self._measure_use_band = bool(self._measure_use_band_cb.isChecked())
        self._measure_f_lo = float(self._measure_f_lo_spin.value())
        self._measure_f_hi = float(self._measure_f_hi_spin.value())
        if 0 <= self._page_index < len(self._pages):
            page = self._pages[self._page_index]
            page.measure_use_band = self._measure_use_band
            page.measure_f_lo = self._measure_f_lo
            page.measure_f_hi = self._measure_f_hi
        self._refresh_measure_dock()

    def _measure_band_from_markers(self) -> None:
        vals: list[float] = []
        for group in self._synced_markers:
            if group.vertical and group.lines:
                vals.append(float(group.lines[0].value()))
            if len(vals) >= 2:
                break
        if len(vals) < 2:
            QMessageBox.information(
                self,
                "Band from markers",
                "Need at least two vertical markers (V1, V2).",
            )
            return
        lo, hi = sorted(vals[:2])
        self._updating_measure_dock = True
        try:
            self._measure_use_band_cb.setChecked(True)
            self._measure_f_lo_spin.setValue(lo)
            self._measure_f_hi_spin.setValue(hi)
        finally:
            self._updating_measure_dock = False
        self._on_measure_band_changed()

    def _primary_spectrum_for_measure(self) -> Series | None:
        """Prefer Mag panel series; fall back to any frequency panel."""
        candidates: list[_PlotPanel] = []
        active = self._active_cursor_panel()
        if active is not None and active.is_frequency():
            candidates.append(active)
        for panel in self._panels:
            if panel is active:
                continue
            if panel.y_kind == "mag":
                candidates.append(panel)
        for panel in self._panels:
            if panel in candidates:
                continue
            if panel.is_frequency():
                candidates.append(panel)
        for panel in candidates:
            for sid in panel.series_ids:
                series = self._project.get(sid)
                if series is None or series.is_spectrogram():
                    continue
                return series
        return None

    def _spectrum_mag_for_measure(
        self, series: Series
    ) -> tuple[np.ndarray, np.ndarray, str]:
        freq, mag = self._spectrum_xy(series, "mag")
        scale = series_amplitude_scale(series.meta.attrs)
        return freq, mag, scale

    def _refresh_measure_dock(self) -> None:
        if not hasattr(self, "_measure_overall_label"):
            return
        series = self._primary_spectrum_for_measure()
        if series is None:
            self._measure_series_label.setText("Series: —")
            self._measure_overall_label.setText("Overall RMS: —")
            self._measure_band_label.setText("Band RMS: —")
            self._measure_peaks_table.setRowCount(0)
            return
        freq, mag, scale = self._spectrum_mag_for_measure(series)
        unit = series.unit or ""
        unit_txt = f" {unit}" if unit else ""
        self._measure_series_label.setText(
            f"Series: {series.display_name}  (scale={scale})"
        )
        if freq.size == 0:
            self._measure_overall_label.setText("Overall RMS: —")
            self._measure_band_label.setText("Band RMS: —")
            return
        overall = band_rms(freq, mag, amplitude_scale=scale)
        overall_db = level_db(overall)
        self._measure_overall_label.setText(
            f"Overall RMS: {overall:.6g}{unit_txt}  ({overall_db:.2f} dB re{unit_txt or ' 1'})"
        )
        if self._measure_use_band and self._measure_f_hi > self._measure_f_lo:
            band = band_rms(
                freq,
                mag,
                f_lo=self._measure_f_lo,
                f_hi=self._measure_f_hi,
                amplitude_scale=scale,
            )
            band_db = level_db(band)
            self._measure_band_label.setText(
                f"Band RMS [{self._measure_f_lo:.4g}–{self._measure_f_hi:.4g} Hz]: "
                f"{band:.6g}{unit_txt}  ({band_db:.2f} dB re{unit_txt or ' 1'})"
            )
        else:
            self._measure_band_label.setText("Band RMS: (enable band / set f min < f max)")

    def _pick_spectrum_peaks(self) -> None:
        series = self._primary_spectrum_for_measure()
        if series is None:
            QMessageBox.information(
                self,
                "Pick peaks",
                "Assign a spectrum (or time) series to a Mag frequency panel first.",
            )
            return
        if not self._panels:
            return
        n_peaks = (
            int(self._measure_n_peaks_spin.value())
            if hasattr(self, "_measure_n_peaks_spin")
            else 5
        )
        freq, mag, _scale = self._spectrum_mag_for_measure(series)
        peaks = find_spectrum_peaks(freq, mag, n_peaks=n_peaks)
        if not peaks:
            QMessageBox.information(self, "Pick peaks", "No local maxima found.")
            self._measure_peaks_table.setRowCount(0)
            return
        unit = series.unit or ""
        self._measure_peaks_table.setRowCount(len(peaks))
        for r, (f_hz, mag_lin, _idx) in enumerate(peaks):
            if self._mag_db:
                mag_txt = f"{float(to_db(mag_lin)):.3g} dB"
                unit_txt = f"re {unit}" if unit else "re mag"
            else:
                mag_txt = f"{mag_lin:.6g}"
                unit_txt = unit
            self._measure_peaks_table.setItem(r, 0, QTableWidgetItem(f"{f_hz:.6g}"))
            self._measure_peaks_table.setItem(r, 1, QTableWidgetItem(mag_txt))
            self._measure_peaks_table.setItem(r, 2, QTableWidgetItem(unit_txt))
            self.add_marker(vertical=True, value=float(f_hz))
        self._refresh_cursor_values_panel()
        self._refresh_measure_dock()
        if hasattr(self, "_measure_dock_action"):
            self._measure_dock_action.setChecked(True)
        self._measure_dock.setVisible(True)
        self._measure_dock.raise_()
        self.statusBar().showMessage(f"Placed {len(peaks)} peak marker(s)", 4000)

    def open_project_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open project",
            "",
            "Ginkyo project (*.ginkyo);;All files (*)",
        )
        if not path:
            return
        try:
            project = Project.open(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Open project failed", str(exc))
            return
        self._load_project_into_window(project)

    def save_project_dialog(self) -> None:
        suggested = "project.ginkyo"
        if self._project.path is not None:
            suggested = str(self._project.path)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save project",
            suggested,
            "Ginkyo project (*.ginkyo);;All files (*)",
        )
        if not path:
            return
        try:
            self._project.save(path, views=self._serialize_views())
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save project failed", str(exc))
            return
        self.statusBar().showMessage(f"Saved project: {path}", 5000)

    def _serialize_views(self) -> list[dict]:
        self._stash_live_to_page(self._page_index)
        views: list[dict] = []
        for page in self._pages:
            markers: list[dict] = []
            if page.built and page.synced_markers:
                for group in page.synced_markers:
                    if not group.lines:
                        continue
                    markers.append(
                        {
                            "vertical": group.vertical,
                            "value": float(group.lines[0].value()),
                        }
                    )
            else:
                for vertical, value in page.pending_markers:
                    markers.append({"vertical": vertical, "value": value})
            view_ranges: list[dict] = []
            if page.built and page.panels:
                for panel in page.panels:
                    rng = panel.plot.getViewBox().viewRange()
                    view_ranges.append(
                        {"x": [float(rng[0][0]), float(rng[0][1])], "y": [float(rng[1][0]), float(rng[1][1])]}
                    )
            else:
                view_ranges = list(page.pending_view_ranges)
            ws = page.workspace
            if page is self._pages[self._page_index]:
                self._sync_workspace_from_panels()
            views.append(
                {
                    "title": page.title,
                    "display_mode": page.display_mode,
                    "mag_db": bool(page.mag_db),
                    "measure_use_band": bool(page.measure_use_band),
                    "measure_f_lo": float(page.measure_f_lo),
                    "measure_f_hi": float(page.measure_f_hi),
                    "workspace": ws.to_dict(),
                    "markers": markers,
                    "view_ranges": view_ranges,
                    "active_panel": page.active_panel,
                }
            )
        return views

    def _load_project_into_window(self, project: Project) -> None:
        # Tear down all view-page widgets (Analysis lives in the outer mode tab).
        self._suppress_tab_change = True
        for i in range(len(self._pages) - 1, -1, -1):
            page = self._pages[i]
            if page.built and page.root_plot_widget is not None:
                self._destroy_page_runtime(page)
            if 0 <= i < self._tabs.count():
                self._tabs.removeTab(i)
        self._pages.clear()
        self._clear_live_refs()
        self._suppress_tab_change = False

        self._project = project
        self._selected_source_id = project.sources[0].id if project.sources else None
        views = take_pending_views(project)
        if not views:
            views = [
                {
                    "title": "Page 1",
                    "display_mode": _MODE_TIME,
                    "workspace": WorkspaceLayout().to_dict(),
                    "markers": [],
                    "view_ranges": [],
                    "active_panel": 0,
                }
            ]

        self._suppress_tab_change = True
        for i, raw in enumerate(views):
            title = str(raw.get("title") or f"Page {i + 1}")
            try:
                workspace = WorkspaceLayout.from_dict(raw.get("workspace") or {})
            except Exception:  # noqa: BLE001
                workspace = WorkspaceLayout()
            workspace.prune_series(project.prune_ids)
            # Infer leaf view_kind from assigned analysis results.
            for leaf in workspace.leaves():
                kind = self._view_kind_for_series_ids(leaf.series_ids)
                if kind != "time":
                    leaf.view_kind = kind
            mode = str(raw.get("display_mode") or _MODE_TIME)
            if mode not in self._spectrum_actions:
                mode = _MODE_TIME
            markers = []
            for m in raw.get("markers") or []:
                markers.append((bool(m.get("vertical", True)), float(m.get("value") or 0.0)))
            page = _PageState(
                title=title,
                workspace=workspace,
                display_mode=mode,
                mag_db=bool(raw.get("mag_db", False)),
                measure_use_band=bool(raw.get("measure_use_band", False)),
                measure_f_lo=float(raw.get("measure_f_lo") or 0.0),
                measure_f_hi=float(raw.get("measure_f_hi") or 0.0),
                pending_markers=markers,
                pending_view_ranges=list(raw.get("view_ranges") or []),
                active_panel=int(raw.get("active_panel") or 0),
            )
            self._pages.append(page)
            self._tabs.addTab(self._make_page_host(), title)
        self._tabs.setCurrentIndex(0)
        self._suppress_tab_change = False
        self._show_views_workspace()
        if self._pages:
            self._activate_page(0)
        self._analysis_page.refresh_series()
        self._refresh_project_tree()
        self.statusBar().showMessage(f"Opened project: {project.path}", 5000)

    def reset_zoom(self) -> None:
        self._fit_view()

    def open_file_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Add recording to project",
            "",
            "Recordings (*.wav *.WAV *.csv *.CSV *.uff *.UFF *.unv *.UNV);;"
            "WAV (*.wav *.WAV);;CSV (*.csv *.CSV);;UFF/UNV (*.uff *.UFF *.unv *.UNV);;"
            "All files (*)",
        )
        if not path:
            return
        try:
            recording = self._load_path(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Failed to open", str(exc))
            return
        self.add_recording(recording, reset_layout=False)

    def _load_path(self, path: str) -> Recording:
        suffix = Path(path).suffix.lower()
        if suffix == ".wav":
            return read_wav(path)
        if suffix == ".csv":
            return read_csv(path)
        if suffix in {".uff", ".unv"}:
            return read_uff(path)
        raise ValueError(f"Unsupported file type: {suffix or '(none)'}")

    def export_csv_dialog(self) -> None:
        src = self._selected_source()
        if src is None:
            QMessageBox.information(
                self,
                "Export CSV",
                "Select a source in Project data (then export that recording).",
            )
            return
        recording = src.recording
        if recording is None:
            recording = self._recording_from_source(src.id)
        if recording is None or not recording.channels:
            QMessageBox.information(self, "Export CSV", "Selected source has no channels.")
            return
        suggested = "export.csv"
        if recording.source and not recording.source.startswith("dummy:"):
            suggested = Path(recording.source).with_suffix(".csv").name
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export CSV",
            suggested,
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return
        try:
            export_csv(recording, path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Failed to export CSV", str(exc))
            return
        self.statusBar().showMessage(f"Exported CSV: {path}", 5000)

    def _recording_from_source(self, source_id: str) -> Recording | None:
        series_list = [s for s in self._project.all_series() if s.source_id == source_id]
        if not series_list:
            return None
        for s in series_list:
            s.ensure_loaded()
        src = self._project.source_by_id(source_id)
        channels = [
            Channel(name=s.name, data=s.data, unit=s.unit) for s in series_list
        ]
        time = series_list[0].time
        return Recording(
            sample_rate=float(series_list[0].sample_rate),
            channels=channels,
            source=(src.provenance if src is not None else "") or source_id,
            time=time,
        )

    def add_recording(self, recording: Recording, *, reset_layout: bool = False) -> None:
        was_empty = not self._project.series_order
        new_ids = self._project.add_recording(recording)
        if new_ids:
            self._selected_source_id = self._project.get(new_ids[0]).source_id  # type: ignore[union-attr]
        if reset_layout or was_empty:
            self.clear_markers()
            self._set_workspace(
                WorkspaceLayout.default_for_series(list(self._project.series_order))
            )
            self._display_mode = _MODE_TIME
            if 0 <= self._page_index < len(self._pages):
                self._pages[self._page_index].display_mode = _MODE_TIME
            for key, act in self._spectrum_actions.items():
                act.setChecked(key == _MODE_TIME)
            self._rebuild_panels()
        self._refresh_project_tree()
        self._update_active_panel_label()
        self._redraw_curves(fit=bool(reset_layout or was_empty))
        self._update_meta()

    def clear_project(self) -> None:
        self._project.clear()
        self._selected_source_id = None
        self.clear_markers()
        self._set_workspace(WorkspaceLayout())
        self._display_mode = _MODE_TIME
        if 0 <= self._page_index < len(self._pages):
            self._pages[self._page_index].display_mode = _MODE_TIME
        for key, act in self._spectrum_actions.items():
            act.setChecked(key == _MODE_TIME)
        self._rebuild_panels()
        self._refresh_project_tree()
        self._update_active_panel_label()
        self._redraw_curves(fit=True)
        self._update_meta()

    def remove_selected_source(self) -> None:
        src = self._selected_source()
        if src is None:
            QMessageBox.information(self, "Remove source", "Select a source in Project data.")
            return
        self._project.remove_source(src.id)
        if self._selected_source_id == src.id:
            self._selected_source_id = (
                self._project.sources[0].id if self._project.sources else None
            )
        self._prune_workspace_series()
        self._rebuild_panels()
        self._refresh_project_tree()
        self._update_active_panel_label()
        self._redraw_curves(fit=True)
        self._update_meta()

    def _selected_source(self):
        if self._selected_source_id:
            found = self._project.source_by_id(self._selected_source_id)
            if found is not None:
                return found
        if self._project.sources:
            return self._project.sources[0]
        return None

    def _on_project_selection(self) -> None:
        items = self._project_tree.selectedItems()
        if not items:
            return
        item = items[0]
        source_id = item.data(0, ROLE_SOURCE)
        if source_id:
            self._selected_source_id = str(source_id)
            self._update_meta()

    def _project_tree_menu(self, pos) -> None:  # noqa: ANN001
        item = self._project_tree.itemAt(pos)
        if item is None:
            return
        ids = self._project_tree.selected_series_ids()
        if not ids and item.data(0, ROLE_SERIES):
            ids = [str(item.data(0, ROLE_SERIES))]
        if not ids:
            return
        menu = QMenu(self)
        add_analysis = menu.addAction("Add to Analysis…")
        edit = None
        if len(ids) == 1:
            edit = menu.addAction("Edit series metadata…")
        chosen = menu.exec(self._project_tree.viewport().mapToGlobal(pos))
        if chosen is add_analysis:
            self._add_selection_to_analysis(ids)
        elif edit is not None and chosen is edit:
            self._edit_series_meta(ids[0])

    def _add_selection_to_analysis(self, series_ids: list[str] | None = None) -> None:
        ids = series_ids if series_ids is not None else self._project_tree.selected_series_ids()
        added = self._analysis_page.add_series_ids(ids)
        self._show_analysis_page()
        if added:
            self.statusBar().showMessage(
                f"Added {added} series to Analysis set", 4000
            )
        else:
            self.statusBar().showMessage(
                "Nothing new added to Analysis (already listed or not a time series)",
                4000,
            )

    def _edit_series_meta(self, series_id: str) -> None:
        series = self._project.get(series_id)
        if series is None:
            return
        m = series.meta
        fields = [
            ("quantity", m.quantity),
            ("point_id", m.point_id),
            ("point_name", m.point_name),
            ("dof", m.dof),
            ("ref_point_id", m.ref_point_id),
            ("ref_point_name", m.ref_point_name),
            ("ref_dof", m.ref_dof),
            ("provenance", m.provenance),
        ]
        # Simple sequential prompts keep UI tiny for v1.
        for key, current in fields:
            text, ok = QInputDialog.getText(
                self, "Series metadata", f"{key}:", text=str(current)
            )
            if not ok:
                return
            setattr(m, key, text.strip())
        self._refresh_project_tree()

    def _refresh_project_tree(self) -> None:
        filt = (
            self._project_filter.text().strip().lower()
            if hasattr(self, "_project_filter")
            else ""
        )
        sort_key = "name"
        if hasattr(self, "_project_sort"):
            sort_key = str(self._project_sort.currentData() or "name")

        def sort_series(items: list) -> list:
            if sort_key == "point":
                return sorted(
                    items,
                    key=lambda s: (
                        (s.meta.point_name or s.meta.point_id).lower(),
                        s.name.lower(),
                    ),
                )
            if sort_key == "dof":
                return sorted(
                    items,
                    key=lambda s: (s.meta.dof.lower(), s.name.lower()),
                )
            return sorted(items, key=lambda s: s.name.lower())

        self._project_tree.clear()
        for src in self._project.sources:
            parent = QTreeWidgetItem([src.label])
            parent.setData(0, ROLE_SOURCE, src.id)
            self._project_tree.addTopLevelItem(parent)
            children = [s for s in self._project.all_series() if s.source_id == src.id]
            children = sort_series(children)
            visible_children = 0
            for series in children:
                label = series.tree_label()
                if series.unit:
                    label = f"{label} [{series.unit}]"
                if filt and filt not in label.lower() and filt not in src.label.lower():
                    continue
                child = QTreeWidgetItem([label])
                child.setData(0, ROLE_SOURCE, src.id)
                child.setData(0, ROLE_SERIES, series.id)
                parent.addChild(child)
                visible_children += 1
            parent.setExpanded(True)
            parent.setHidden(bool(filt) and visible_children == 0)
            if src.id == self._selected_source_id:
                parent.setSelected(True)
        if hasattr(self, "_analysis_page"):
            self._analysis_page.refresh_series()
        self._update_meta()

    def _save_layout_dialog(self) -> None:
        layouts_dir().mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save layout",
            str(layouts_dir() / "layout.json"),
            "JSON (*.json);;All files (*)",
        )
        if not path:
            return
        self._workspace.display_mode = self._display_mode
        self._sync_workspace_from_panels()
        try:
            save_layout(path, self._workspace)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save layout failed", str(exc))
            return
        self.statusBar().showMessage(f"Saved layout: {path}", 5000)

    def _load_layout_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load layout",
            str(layouts_dir()),
            "JSON (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            layout = load_layout(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Load layout failed", str(exc))
            return
        self._set_workspace(layout)
        self._prune_workspace_series()
        mode = layout.display_mode if layout.display_mode in self._spectrum_actions else _MODE_TIME
        self._display_mode = mode
        if 0 <= self._page_index < len(self._pages):
            self._pages[self._page_index].display_mode = mode
        for key, act in self._spectrum_actions.items():
            act.setChecked(key == mode)
        self._rebuild_panels()
        self._update_active_panel_label()
        self._redraw_curves(fit=True)
        self.statusBar().showMessage(f"Loaded layout: {path}", 5000)

    def _sync_workspace_from_panels(self) -> None:
        if self._spectrum_mode():
            return
        seen: set[int] = set()
        for panel in self._panels:
            leaf = panel.layout_leaf
            if leaf is None:
                continue
            key = id(leaf)
            if key in seen:
                continue
            seen.add(key)
            leaf.series_ids = list(panel.series_ids)
            leaf.series_colors = {
                k: v for k, v in panel.series_colors.items() if k in panel.series_ids
            }
        self._sync_splitter_sizes_to_tree()

    # --- markers (synced across panels) ----------------------------------

    def add_marker(self, *, vertical: bool, value: float | None = None) -> None:
        if not self._panels:
            return
        if value is None:
            panel = self._active() or self._panels[0]
            if self._last_view_pos is None:
                x_range, y_range = panel.plot.getViewBox().viewRange()
                pos = (
                    0.5 * (x_range[0] + x_range[1]),
                    0.5 * (y_range[0] + y_range[1]),
                )
            else:
                pos = self._last_view_pos
            value = float(pos[0] if vertical else pos[1])

        lines: list[_MarkerLine] = []
        for panel in self._panels:
            if vertical:
                line = _MarkerLine(
                    self,
                    vertical=True,
                    pos=value,
                    angle=90,
                    movable=True,
                    pen=pg.mkPen("#c0392b", width=1.5, style=Qt.PenStyle.DashLine),
                    label="{value:.6g}",
                    labelOpts={
                        "position": 0.92,
                        "color": "#c0392b",
                        "fill": (255, 255, 255, 180),
                        "movable": False,
                    },
                )
            else:
                line = _MarkerLine(
                    self,
                    vertical=False,
                    pos=value,
                    angle=0,
                    movable=True,
                    pen=pg.mkPen("#2980b9", width=1.5, style=Qt.PenStyle.DashLine),
                    label="{value:.6g}",
                    labelOpts={
                        "position": 0.92,
                        "color": "#2980b9",
                        "fill": (255, 255, 255, 180),
                        "movable": False,
                    },
                )
            line.sigPositionChanged.connect(self._on_synced_marker_moved)
            line.sigPositionChangeFinished.connect(
                lambda _=None, ln=line: self.select_marker(ln)
            )
            panel.plot.addItem(line, ignoreBounds=True)
            lines.append(line)
        group = _SyncedMarker(vertical=vertical, lines=lines)
        self._synced_markers.append(group)
        self.select_marker(lines[min(self._active_panel, len(lines) - 1)])
        self._refresh_marker_readout()

    def _group_for_line(self, line: _MarkerLine) -> _SyncedMarker | None:
        for group in self._synced_markers:
            if line in group.lines:
                return group
        return None

    def _on_synced_marker_moved(self, line: _MarkerLine = None) -> None:  # noqa: ANN001
        if line is None or getattr(self, "_syncing_markers", False):
            return
        group = self._group_for_line(line)
        if group is None:
            return
        value = float(line.value())
        self._syncing_markers = True
        try:
            for other in group.lines:
                if other is line:
                    continue
                other.blockSignals(True)
                other.setValue(value)
                other.blockSignals(False)
                # setValue with signals blocked skips InfLineLabel updates
                self._refresh_marker_line_label(other)
        finally:
            self._syncing_markers = False
        self._refresh_marker_readout()

    def _refresh_marker_line_label(self, line: _MarkerLine) -> None:
        label = getattr(line, "label", None)
        if label is not None and hasattr(label, "valueChanged"):
            label.valueChanged()
        elif label is not None and hasattr(label, "updatePosition"):
            label.updatePosition()

    def select_marker(self, line: _MarkerLine | None) -> None:
        all_lines = [ln for g in self._synced_markers for ln in g.lines]
        self._selected_marker = line if line in all_lines else None
        group = self._group_for_line(line) if line is not None else None
        for ln in all_lines:
            selected = group is not None and ln in group.lines
            ln.set_selected_style(selected)
        if self._selected_marker is not None:
            self.setFocus(Qt.FocusReason.OtherFocusReason)

    def _on_scene_clicked(self, ev) -> None:  # noqa: ANN001
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        for panel in self._panels:
            scene = panel.plot.scene()
            if scene is None:
                continue
            try:
                items = scene.items(ev.scenePos())
            except Exception:  # noqa: BLE001
                continue
            for item in items:
                if isinstance(item, _MarkerLine):
                    return
                if isinstance(item, InfLineLabel):
                    return
                parent = item.parentItem() if hasattr(item, "parentItem") else None
                if isinstance(parent, (_MarkerLine, InfLineLabel)):
                    return
        self.select_marker(None)

    def nudge_selected_marker(self, key: int, *, fine: bool) -> bool:
        line = self._selected_marker
        all_lines = [ln for g in self._synced_markers for ln in g.lines]
        if line is None or line not in all_lines:
            line = self._pick_nearest_marker()
            if line is None:
                return False
            self.select_marker(line)

        panel = self._active()
        if panel is None:
            return False
        x_range, y_range = panel.plot.getViewBox().viewRange()
        if line.vertical:
            span = float(x_range[1] - x_range[0])
            if key == Qt.Key.Key_Left:
                delta = -1.0
            elif key == Qt.Key.Key_Right:
                delta = 1.0
            else:
                return False
        else:
            span = float(y_range[1] - y_range[0])
            if key == Qt.Key.Key_Down:
                delta = -1.0
            elif key == Qt.Key.Key_Up:
                delta = 1.0
            else:
                return False

        if span <= 0 or not np.isfinite(span):
            return False
        step = span / (1000.0 if fine else 200.0)
        line.setValue(float(line.value()) + delta * step)
        self._on_synced_marker_moved(line)
        return True

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        key = event.key()
        if key in (
            Qt.Key.Key_Left,
            Qt.Key.Key_Right,
            Qt.Key.Key_Up,
            Qt.Key.Key_Down,
        ):
            fine = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            if self.nudge_selected_marker(key, fine=fine):
                event.accept()
                return
        super().keyPressEvent(event)

    def edit_marker_value(self, line: _MarkerLine) -> None:
        value = self._prompt_marker_value(
            vertical=line.vertical,
            current=float(line.value()),
        )
        if value is None:
            return
        line.setValue(value)
        self._on_synced_marker_moved(line)

    def remove_marker(self, line: _MarkerLine) -> None:
        group = self._group_for_line(line)
        if group is None:
            return
        for ln in group.lines:
            for panel in self._panels:
                try:
                    panel.plot.removeItem(ln)
                except Exception:  # noqa: BLE001
                    pass
        self._synced_markers = [g for g in self._synced_markers if g is not group]
        if self._selected_marker in group.lines:
            self._selected_marker = None
        self._refresh_marker_readout()

    def _pick_nearest_marker(self) -> _MarkerLine | None:
        lines = [ln for g in self._synced_markers for ln in g.lines]
        if not lines:
            return None
        if self._last_view_pos is None:
            return lines[-1]
        x, y = self._last_view_pos
        best = lines[0]
        best_d = float("inf")
        for line in lines:
            if line.vertical:
                dist = abs(float(line.value()) - x)
            else:
                dist = abs(float(line.value()) - y)
            if dist < best_d:
                best_d = dist
                best = line
        return best

    def remove_nearest_marker(self) -> None:
        line = self._pick_nearest_marker()
        if line is not None:
            self.remove_marker(line)

    def _prompt_marker_value(
        self, *, vertical: bool, current: float | None
    ) -> float | None:
        if vertical:
            name = "Frequency f" if self._spectrum_mode() else "Time t"
            unit = "Hz" if self._spectrum_mode() else "s"
            title = "Vertical marker"
        else:
            name = "Amplitude y"
            unit = ""
            title = "Horizontal marker"
        prompt = f"{name}" + (f" [{unit}]" if unit else "") + ":"
        default = "" if current is None else f"{current:.6g}"
        text, ok = QInputDialog.getText(self, title, prompt, text=default)
        if not ok:
            return None
        try:
            return float(text.strip())
        except ValueError:
            QMessageBox.warning(self, title, f"Invalid number: {text!r}")
            return None

    def _marker_label_format(self) -> str:
        return "{value:.6g}"

    def _sync_marker_labels(self) -> None:
        fmt = self._marker_label_format()
        for group in self._synced_markers:
            for line in group.lines:
                label = getattr(line, "label", None)
                if label is not None and hasattr(label, "setFormat"):
                    label.setFormat(fmt)

    def clear_markers(self) -> None:
        for group in self._synced_markers:
            for line in group.lines:
                for panel in self._panels:
                    try:
                        panel.plot.removeItem(line)
                    except Exception:  # noqa: BLE001
                        pass
        self._synced_markers.clear()
        self._selected_marker = None

    def _refresh_marker_readout(self) -> None:
        parts: list[str] = []
        x_name = "f" if self._spectrum_mode() else "t"
        for group in self._synced_markers:
            if not group.lines:
                continue
            v = float(group.lines[0].value())
            if group.vertical:
                parts.append(f"V {x_name}={v:.6g}")
            else:
                parts.append(f"H y={v:.6g}")
        if parts:
            self.statusBar().showMessage("Markers: " + " | ".join(parts), 8000)
        self._refresh_cursor_values_panel()

    # --- draw ------------------------------------------------------------

    def _fit_view(self) -> None:
        for panel in self._panels:
            vb = panel.plot.getViewBox()
            vb.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
            vb.autoRange()
            vb.disableAutoRange()

    def _redraw_curves(self, *, fit: bool = False) -> None:
        saved: list = []
        for panel in self._panels:
            saved.append(None if fit else panel.plot.getViewBox().viewRange())

        for panel in self._panels:
            for curve in panel.curves:
                panel.plot.removeItem(curve)
            panel.curves.clear()
            for image in panel.image_items:
                panel.plot.removeItem(image)
            panel.image_items.clear()
            self._clear_panel_colorbars(panel)
            panel.plot.legend = None
            panel.plot.addLegend()
            if panel.v_line.scene() is None:
                panel.plot.addItem(panel.v_line, ignoreBounds=True)

        if not self._project.series_order:
            return

        if self._spectrum_mode():
            self._draw_spectrum_panels()
        else:
            for i, panel in enumerate(self._panels):
                if panel.is_spectrogram():
                    self._draw_spectrogram_panel(panel, i)
                elif panel.is_frequency():
                    self._draw_frequency_panel(panel)
                else:
                    self._draw_time_panel(panel, i)

        if fit:
            self._fit_view()
        else:
            for panel, rng in zip(self._panels, saved):
                if rng is None:
                    continue
                vb = panel.plot.getViewBox()
                vb.setRange(xRange=rng[0], yRange=rng[1], padding=0)
                vb.disableAutoRange()

        self._refresh_spectrogram_dock()
        self._refresh_measure_dock()

    def _draw_time_panels(self) -> None:
        for i, panel in enumerate(self._panels):
            self._draw_time_panel(panel, i)

    def _draw_time_panel(self, panel: _PlotPanel, index: int) -> None:
        units = {
            s.unit
            for sid in panel.series_ids
            for s in [self._project.get(sid)]
            if s is not None and s.unit and not s.is_fft_result()
        }
        left = "Amplitude"
        if len(units) == 1:
            left = f"Amplitude ({next(iter(units))})"

        panel.plot.setLabel("bottom", "Time", units="s")
        panel.plot.setLabel("left", left)
        ids = panel.series_ids
        if not ids:
            panel.plot.setTitle(f"Panel {index + 1} (empty)")
            return
        names = []
        for sid in ids:
            series = self._project.get(sid)
            if series is None:
                continue
            if series.is_spectrogram():
                continue
            names.append(series.display_name)
            color = self._color_for_panel_series(panel, sid)
            if series.is_spectrum():
                # Mag-only if somehow on a time leaf.
                curve = panel.plot.plot(
                    series.time_axis(),
                    series.data,
                    pen=pg.mkPen(color, width=1),
                    name=series.display_name,
                )
            else:
                curve = panel.plot.plot(
                    series.time_axis(),
                    series.data,
                    pen=pg.mkPen(color, width=1),
                    name=series.display_name,
                )
            panel.curves.append(curve)
        panel.plot.setTitle(", ".join(names) if names else f"Panel {index + 1}")

    def _draw_spectrogram_panel(self, panel: _PlotPanel, index: int) -> None:
        y_mode = panel.spectrogram_y_axis
        if y_mode == "angle" and not self._panel_has_spectrogram_angle(panel):
            panel.spectrogram_y_axis = "time"
            y_mode = "time"
        elif y_mode == "rpm" and not self._panel_has_spectrogram_rpm(panel):
            panel.spectrogram_y_axis = "time"
            y_mode = "time"
        panel.plot.setLabel("bottom", "Frequency", units="Hz")
        if y_mode == "angle":
            panel.plot.setLabel("left", "Angle", units="deg")
        elif y_mode == "rpm":
            panel.plot.setLabel("left", "RPM", units="")
        else:
            panel.plot.setLabel("left", "Time", units="s")
        ids = panel.series_ids
        if not ids:
            panel.plot.setTitle(f"Panel {index + 1} (empty)")
            return

        names: list[str] = []
        images: list[pg.ImageItem] = []
        levels = panel.spectrogram_levels
        if levels is None:
            levels = self._spectrogram_levels_for_panel_data(panel)

        for sid in ids:
            series = self._project.get(sid)
            if series is None or not series.is_spectrogram():
                continue
            series.ensure_loaded()
            y_axis = self._spectrogram_y_values(series, y_mode=y_mode)
            freq = series.time
            mag = series.data
            if (
                y_axis is None
                or freq is None
                or mag is None
                or mag.ndim != 2
                or mag.size == 0
            ):
                continue
            names.append(series.display_name)
            display = _spectrogram_db(mag)  # (freq, y) → x=freq, y=time/angle/rpm
            image = pg.ImageItem()
            if levels is not None:
                image.setImage(display, autoLevels=False, levels=list(levels))
            else:
                image.setImage(display, autoLevels=True)
            image.setColorMap("viridis")
            f0, fw = _axis_image_extent(freq)
            y0, yh = _axis_image_extent(y_axis)
            image.setRect(f0, y0, fw, yh)
            image.setZValue(-10)
            panel.plot.addItem(image)
            panel.image_items.append(image)
            images.append(image)

        if images:
            bar_kwargs: dict = {
                "colorMap": "viridis",
                "label": "dB",
                "interactive": True,
                "rounding": 2,  # snap drag adjustments to 2 dB
            }
            if levels is not None:
                bar_kwargs["values"] = levels
            # One color bar for the whole panel (avoids stacked overlapping bars).
            bar = panel.plot.addColorBar(images, **bar_kwargs)
            self._style_spectrogram_colorbar(bar, levels)
            self._connect_spectrogram_colorbar(panel, bar)
            panel.color_bars.append(bar)

        title = ", ".join(names) if names else f"Panel {index + 1}"
        if len(names) < len(ids):
            title = title + " (non-spectrogram series skipped)"
        panel.plot.setTitle(title)

    def _spectrogram_y_values(
        self, series: Series, *, y_mode: str
    ) -> np.ndarray | None:
        if y_mode == "angle":
            angles = series.frame_angle_deg
            if angles is not None and np.asarray(angles).size:
                return np.asarray(angles, dtype=np.float64)
            return None
        if y_mode == "rpm":
            rpms = series.frame_rpm
            if rpms is not None and np.asarray(rpms).size:
                return np.asarray(rpms, dtype=np.float64)
            return None
        times = series.frame_time_s
        if times is None:
            return None
        return np.asarray(times, dtype=np.float64)

    def _panel_has_spectrogram_angle(self, panel: _PlotPanel) -> bool:
        for sid in panel.series_ids:
            series = self._project.get(sid)
            if series is None or not series.is_spectrogram():
                continue
            series.ensure_loaded()
            angles = series.frame_angle_deg
            if angles is not None and np.asarray(angles).size:
                return True
        return False

    def _panel_has_spectrogram_rpm(self, panel: _PlotPanel) -> bool:
        for sid in panel.series_ids:
            series = self._project.get(sid)
            if series is None or not series.is_spectrogram():
                continue
            series.ensure_loaded()
            rpms = series.frame_rpm
            if rpms is not None and np.asarray(rpms).size:
                return True
        return False

    def set_spectrogram_y_axis(self, axis: str) -> None:
        """Switch spectrogram panel Y axis between time, angle, and RPM."""
        panel = self._active_cursor_panel()
        if panel is None or not panel.is_spectrogram():
            return
        if axis == "angle" and not self._panel_has_spectrogram_angle(panel):
            return
        if axis == "rpm" and not self._panel_has_spectrogram_rpm(panel):
            return
        if axis in ("angle", "rpm"):
            panel.spectrogram_y_axis = axis
        else:
            panel.spectrogram_y_axis = "time"
        self._redraw_curves(fit=True)
        self._refresh_cursor_values_panel()

    def _clear_panel_colorbars(self, panel: _PlotPanel) -> None:
        """Remove color bars from the plot layout (prevents stacked ghost bars)."""
        layout = panel.plot.layout
        for bar in list(panel.color_bars):
            try:
                layout.removeItem(bar)
            except Exception:  # noqa: BLE001
                pass
            try:
                bar.setParentItem(None)
            except Exception:  # noqa: BLE001
                pass
        panel.color_bars.clear()
        # Sweep leftovers from previous redraws (same layout cell stacking).
        leftovers: list = []
        try:
            for item in layout.items.keys():  # type: ignore[attr-defined]
                if isinstance(item, pg.ColorBarItem):
                    leftovers.append(item)
        except Exception:  # noqa: BLE001
            leftovers = []
        for item in leftovers:
            try:
                layout.removeItem(item)
            except Exception:  # noqa: BLE001
                pass
        try:
            layout.setColumnFixedWidth(4, 0)
            layout.setColumnFixedWidth(5, 0)
        except Exception:  # noqa: BLE001
            pass

    def _spectrogram_levels_for_panel_data(
        self, panel: _PlotPanel
    ) -> tuple[float, float] | None:
        mins: list[float] = []
        maxs: list[float] = []
        for sid in panel.series_ids:
            series = self._project.get(sid)
            if series is None or not series.is_spectrogram():
                continue
            series.ensure_loaded()
            mag = series.data
            if mag is None or mag.ndim != 2 or mag.size == 0:
                continue
            db = _spectrogram_db(mag)
            mins.append(float(np.min(db)))
            maxs.append(float(np.max(db)))
        if not mins:
            return None
        return _nice_db_levels(float(min(mins)), float(max(maxs)))

    def _displayed_spectrogram_levels(
        self, panel: _PlotPanel
    ) -> tuple[float, float] | None:
        if panel.spectrogram_levels is not None:
            return panel.spectrogram_levels
        if panel.color_bars:
            lo, hi = panel.color_bars[0].levels()
            return float(lo), float(hi)
        return self._spectrogram_levels_for_panel_data(panel)

    def _style_spectrogram_colorbar(
        self, bar: pg.ColorBarItem, levels: tuple[float, float] | None
    ) -> None:
        """Keep color-bar value labels readable (avoid overlapping auto-ticks)."""
        axis = getattr(bar, "axis", None)
        if axis is None:
            return
        axis.setWidth(56)
        # Prefer pyqtgraph auto ticks, but limit density via style.
        axis.setStyle(tickTextOffset=4, autoExpandTextSpace=True)
        if levels is None:
            lo, hi = bar.levels()
        else:
            lo, hi = levels
        ticks = _colorbar_tick_values(float(lo), float(hi), count=9)
        axis.setTicks([[(v, f"{v:g}") for v in ticks], []])

    def _refresh_spectrogram_dock(self) -> None:
        panel = self._active_cursor_panel()
        enabled = panel is not None and panel.is_spectrogram()
        for widget in self._spec_scale_widgets:
            widget.setEnabled(enabled)
        if not enabled:
            self._spec_scale_status.setText("No spectrogram panel selected")
            return
        levels = self._displayed_spectrogram_levels(panel)
        if levels is None:
            self._spec_scale_status.setText("Auto (no data)")
            return
        lo, hi = levels
        auto = panel.spectrogram_levels is None
        self._spec_scale_status.setText(
            "Auto (from data)" if auto else "Manual levels"
        )
        self._updating_spec_dock = True
        self._spec_db_min_spin.setValue(lo)
        self._spec_db_max_spin.setValue(hi)
        self._updating_spec_dock = False

    def _apply_spectrogram_levels_to_panel(
        self, panel: _PlotPanel, lo: float, hi: float
    ) -> None:
        if lo >= hi:
            return
        panel.spectrogram_levels = (float(lo), float(hi))
        for image in panel.image_items:
            image.setLevels((lo, hi))
        for bar in panel.color_bars:
            bar.setLevels(values=(lo, hi))
            self._style_spectrogram_colorbar(bar, (lo, hi))
        self._refresh_spectrogram_dock()

    def _apply_spectrogram_levels_from_dock(self) -> None:
        if self._updating_spec_dock:
            return
        panel = self._active_cursor_panel()
        if panel is None or not panel.is_spectrogram():
            return
        lo = float(self._spec_db_min_spin.value())
        hi = float(self._spec_db_max_spin.value())
        self._apply_spectrogram_levels_to_panel(panel, lo, hi)

    def _on_spec_levels_changed(self, _value: float) -> None:
        self._apply_spectrogram_levels_from_dock()

    def _auto_spectrogram_levels(self) -> None:
        panel = self._active_cursor_panel()
        if panel is None or not panel.is_spectrogram():
            return
        panel.spectrogram_levels = None
        self._redraw_curves(fit=False)
        self._refresh_spectrogram_dock()

    def _connect_spectrogram_colorbar(
        self, panel: _PlotPanel, bar: pg.ColorBarItem
    ) -> None:
        def on_changed(_bar: pg.ColorBarItem) -> None:
            if self._updating_spec_dock:
                return
            lo, hi = bar.levels()
            panel.spectrogram_levels = (float(lo), float(hi))
            self._style_spectrogram_colorbar(bar, (float(lo), float(hi)))
            self._refresh_spectrogram_dock()

        bar.sigLevelsChanged.connect(on_changed)
        bar.sigLevelsChangeFinished.connect(on_changed)

    def show_spectrogram_color_scale(self) -> None:
        """Open the spectrogram color-scale dock (from View menu / right-click)."""
        panel = self._active_cursor_panel()
        if panel is None or not panel.is_spectrogram():
            # Prefer the panel under the context menu ViewBox if available.
            for p in self._panels:
                if p.is_spectrogram():
                    try:
                        self._active_panel = self._panels.index(p)
                    except ValueError:
                        pass
                    break
        if hasattr(self, "_spectrogram_dock_action"):
            self._spectrogram_dock_action.setChecked(True)
        self._spectrogram_dock.setVisible(True)
        self._spectrogram_dock.raise_()
        self._refresh_spectrogram_dock()

    def _draw_frequency_panel(self, panel: _PlotPanel) -> None:
        label_map = {
            "mag": ("Magnitude", ""),
            "phase": ("Phase", "rad"),
            "real": ("Real", ""),
            "imag": ("Imag", ""),
        }
        y_name, y_unit = label_map.get(panel.y_kind, ("Y", ""))
        units = {
            s.unit
            for sid in panel.series_ids
            for s in [self._project.get(sid)]
            if s is not None and s.unit and panel.y_kind != "phase"
        }
        series_unit = next(iter(units)) if len(units) == 1 else ""
        panel.plot.setLabel("bottom", "Frequency", units="Hz")
        if panel.y_kind == "mag" and self._mag_db:
            if series_unit:
                panel.plot.setLabel("left", f"Magnitude (dB re {series_unit})")
            else:
                panel.plot.setLabel("left", "Magnitude (dB)")
        elif y_unit:
            panel.plot.setLabel("left", y_name, units=y_unit)
        elif series_unit:
            panel.plot.setLabel("left", f"{y_name} ({series_unit})")
        else:
            panel.plot.setLabel("left", y_name)
        panel.plot.setTitle(y_name)
        for sid in panel.series_ids:
            series = self._project.get(sid)
            if series is None or series.is_spectrogram():
                continue
            freq, y = self._spectrum_xy(series, panel.y_kind)
            if freq.size == 0:
                continue
            if panel.y_kind == "mag" and self._mag_db:
                y = np.asarray(to_db(y), dtype=np.float64)
            color = self._color_for_panel_series(panel, sid)
            curve = panel.plot.plot(
                freq,
                y,
                pen=pg.mkPen(color, width=1),
                name=series.display_name,
            )
            panel.curves.append(curve)

    def _spectrum_xy(self, series, y_kind: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (freq, y) for a series on a frequency panel."""
        if series.is_spectrogram():
            return np.array([]), np.array([])
        if series.is_spectrum():
            freq = series.time_axis()
            if y_kind == "mag":
                return freq, series.data
            if y_kind == "phase":
                phase = series.phase_rad
                if phase is None:
                    return freq, np.zeros_like(series.data)
                return freq, np.asarray(phase, dtype=np.float64)
            # Real/imag not stored; reconstruct from mag/phase if asked.
            mag = series.data
            phase = series.phase_rad
            if phase is None:
                return freq, np.zeros_like(mag)
            c = mag * np.exp(1j * np.asarray(phase, dtype=np.float64))
            if y_kind == "real":
                return freq, np.real(c).astype(np.float64)
            return freq, np.imag(c).astype(np.float64)
        spec = compute_spectrum(series.data, float(series.sample_rate))
        if y_kind == "mag":
            return spec.frequency_hz, spec.magnitude
        if y_kind == "phase":
            return spec.frequency_hz, spec.phase_rad
        if y_kind == "real":
            return spec.frequency_hz, spec.real
        return spec.frequency_hz, spec.imag

    def _draw_spectrum_panels(self) -> None:
        series_ids = list(self._panels[0].series_ids) if self._panels else []
        leaves = self._workspace.leaves()
        if not series_ids and leaves:
            series_ids = list(leaves[0].series_ids)

        for panel in self._panels:
            panel.series_ids = list(series_ids)
            self._draw_frequency_panel(panel)

    def _update_meta(self) -> None:
        src = self._selected_source()
        n_src = len(self._project.sources)
        n_ser = len(self._project.series_order)
        n_panels = self._workspace.root.leaf_count()
        if src is None:
            self.statusBar().showMessage(
                f"project: {n_src} sources, {n_ser} series | panels={n_panels}"
            )
            self.setWindowTitle("Ginkyo 吟響")
            return
        self.statusBar().showMessage(
            f"project: {n_src} sources, {n_ser} series | panels={n_panels}"
        )
        self.setWindowTitle(f"Ginkyo 吟響 — {src.label} (+{max(0, n_src - 1)} more)")

    def _panel_at_scene_pos(self, scene_pos) -> _PlotPanel | None:  # noqa: ANN001
        for i, panel in enumerate(self._panels):
            vb = panel.plot.getViewBox()
            if vb.sceneBoundingRect().contains(scene_pos):
                if self._active_panel != i:
                    self._active_panel = i
                    self._update_active_panel_label()
                return panel
        return None

    def _on_mouse_moved(self, evt) -> None:  # noqa: ANN001
        if not self._project.series_order:
            return
        pos = evt[0]
        panel = self._panel_at_scene_pos(pos)
        if panel is None:
            return
        mouse_point = panel.plot.getViewBox().mapSceneToView(pos)
        x = float(mouse_point.x())
        y = float(mouse_point.y())
        self._last_view_pos = (x, y)
        for p in self._panels:
            p.v_line.setPos(x)
        self._refresh_cursor_values_panel()

    def _nearest_index(self, axis: np.ndarray, x: float) -> int:
        idx = int(np.clip(np.searchsorted(axis, x), 0, axis.size - 1))
        if idx > 0 and abs(axis[idx - 1] - x) < abs(axis[idx] - x):
            idx -= 1
        return idx

    def _fill_cursor_table(self, rows: list[tuple[str, str, str]]) -> None:
        self._cursor_table.setRowCount(len(rows))
        for r, (at, name, value) in enumerate(rows):
            self._cursor_table.setItem(r, 0, QTableWidgetItem(at))
            self._cursor_table.setItem(r, 1, QTableWidgetItem(name))
            self._cursor_table.setItem(r, 2, QTableWidgetItem(value))

    def _copy_cursor_table(self) -> None:
        lines = []
        for r in range(self._cursor_table.rowCount()):
            a = self._cursor_table.item(r, 0)
            b = self._cursor_table.item(r, 1)
            c = self._cursor_table.item(r, 2)
            lines.append(
                f"{a.text() if a else ''}\t{b.text() if b else ''}\t{c.text() if c else ''}"
            )
        from PySide6.QtWidgets import QApplication

        QApplication.clipboard().setText("At\tSeries\tValue\n" + "\n".join(lines))
        self.statusBar().showMessage("Copied cursor table", 3000)

    def _active_cursor_panel(self) -> _PlotPanel | None:
        if not self._panels:
            return None
        idx = min(max(self._active_panel, 0), len(self._panels) - 1)
        return self._panels[idx]

    def _series_value_at_x(
        self, panel: _PlotPanel, x: float
    ) -> list[tuple[str, str]]:
        """Return (series_name, value_text) at x for the active panel."""
        rows: list[tuple[str, str]] = []
        if panel.is_spectrogram():
            y = self._last_view_pos[1] if self._last_view_pos is not None else 0.0
            y_mode = panel.spectrogram_y_axis
            for sid in panel.series_ids:
                series = self._project.get(sid)
                if series is None or not series.is_spectrogram():
                    continue
                series.ensure_loaded()
                y_axis = self._spectrogram_y_values(series, y_mode=y_mode)
                freq = series.time
                mag = series.data
                if y_axis is None or freq is None or mag is None or mag.ndim != 2:
                    continue
                fi = self._nearest_index(np.asarray(freq, dtype=np.float64), x)
                ti = self._nearest_index(np.asarray(y_axis, dtype=np.float64), y)
                val_db = float(_spectrogram_db(mag)[fi, ti])
                unit = f" {series.unit}" if series.unit else ""
                if y_mode == "angle":
                    y_txt = f"θ={float(y_axis[ti]):.4g} deg"
                elif y_mode == "rpm":
                    y_txt = f"RPM={float(y_axis[ti]):.4g}"
                else:
                    y_txt = f"t={float(y_axis[ti]):.4g} s"
                rows.append(
                    (
                        series.display_name,
                        f"{val_db:.3g} dB re{unit or ' mag'} @ "
                        f"{float(freq[fi]):.4g} Hz, {y_txt}",
                    )
                )
            return rows
        if self._spectrum_mode() or panel.is_frequency():
            for sid in panel.series_ids:
                series = self._project.get(sid)
                if series is None or series.is_spectrogram():
                    continue
                freq, y = self._spectrum_xy(series, panel.y_kind)
                if freq.size == 0 or y.size == 0:
                    continue
                idx = self._nearest_index(freq, x)
                val = float(y[idx])
                name = series.display_name
                if panel.y_kind == "phase":
                    rows.append((name, f"{val:.6g} rad"))
                elif panel.y_kind == "mag" and self._mag_db:
                    db = float(to_db(val))
                    unit = f" {series.unit}" if series.unit else ""
                    rows.append((name, f"{db:.3g} dB re{unit or ' mag'}"))
                else:
                    unit = f" {series.unit}" if series.unit else ""
                    rows.append((name, f"{val:.6g}{unit}"))
            return rows

        for sid in panel.series_ids:
            series = self._project.get(sid)
            if series is None:
                continue
            t = series.time_axis()
            if t.size == 0:
                continue
            idx = self._nearest_index(t, x)
            unit = f" {series.unit}" if series.unit else ""
            val = float(series.data[idx])
            rows.append((series.display_name, f"{val:.6g}{unit}"))
        return rows

    def _refresh_cursor_values_panel(self) -> None:
        """Show mouse cursor values and vertical marker values together."""
        panel = self._active_cursor_panel()
        rows: list[tuple[str, str, str]] = []
        parts: list[str] = []
        spectrogram = panel is not None and panel.is_spectrogram()
        freq_panel = (not spectrogram) and (
            self._spectrum_mode() or (panel is not None and panel.is_frequency())
        )
        if spectrogram or freq_panel:
            x_name, x_unit = "f", "Hz"
        else:
            x_name, x_unit = "t", "s"

        if self._last_view_pos is not None and panel is not None:
            x, y = self._last_view_pos
            at = f"Mouse {x_name}={x:.6g}"
            parts.append(f"{x_name}≈{x:.6g} {x_unit}")
            if spectrogram:
                y_mode = panel.spectrogram_y_axis
                if y_mode == "angle":
                    parts.append(f"θ≈{y:.6g} deg")
                    rows.append((at, "f [Hz]", f"{x:.6g}"))
                    rows.append((at, "θ [deg]", f"{y:.6g}"))
                elif y_mode == "rpm":
                    parts.append(f"RPM≈{y:.6g}")
                    rows.append((at, "f [Hz]", f"{x:.6g}"))
                    rows.append((at, "RPM", f"{y:.6g}"))
                else:
                    parts.append(f"t≈{y:.6g} s")
                    rows.append((at, "f [Hz]", f"{x:.6g}"))
                    rows.append((at, "t [s]", f"{y:.6g}"))
            elif freq_panel:
                y_label = "y [dB]" if (panel is not None and panel.y_kind == "mag" and self._mag_db) else "y"
                parts.append(f"{y_label}={y:.6g}")
                rows.append((at, f"{x_name} [{x_unit}]", f"{x:.6g}"))
                rows.append((at, y_label, f"{y:.6g}"))
            else:
                rows.append((at, f"{x_name} [{x_unit}]", f"{x:.6g}"))
            for name, value in self._series_value_at_x(panel, x):
                parts.append(f"{name}={value}")
                rows.append((at, name, value))

        v_idx = 0
        for group in self._synced_markers:
            if not group.vertical or not group.lines:
                continue
            v_idx += 1
            x = float(group.lines[0].value())
            at = f"V{v_idx} {x_name}={x:.6g}"
            parts.append(at)
            rows.append((at, f"{x_name} [{x_unit}]", f"{x:.6g}"))
            if panel is not None:
                for name, value in self._series_value_at_x(panel, x):
                    rows.append((at, name, value))

        if not rows:
            self._cursor_label.setText("Cursor: (move mouse on plot / add vertical markers)")
            self._fill_cursor_table([])
            return
        self._cursor_label.setText("Cursor: " + " | ".join(parts[:8]))
        self._fill_cursor_table(rows)
