"""Main window: multi-panel viewer + spectrum pairs (M2/M3 minimal)."""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup, QKeyEvent, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QColorDialog,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTabBar,
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

from nagilize.core.dummy import make_sine_with_noise
from nagilize.core.model import Channel, Recording
from nagilize.core.project import Project
from nagilize.core.project_io import take_pending_views
from nagilize.core.spectrum import compute_spectrum
from nagilize.export.csv_export import export_csv
from nagilize.readers.csv_reader import read_csv
from nagilize.readers.uff import read_uff
from nagilize.readers.wav import read_wav
from nagilize.ui.analysis_page import AnalysisPage
from nagilize.ui.layout_state import (
    LayoutNode,
    WorkspaceLayout,
    build_preset,
    layouts_dir,
    load_layout,
    preset_ids,
    preset_label,
    save_layout,
)
from nagilize.ui.panel_shell import ROLE_SERIES, ROLE_SOURCE, PanelShell, SeriesTree

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

_MODE_TIME = "time"
_MODE_MAG_PHASE = "mag_phase"
_MODE_REAL_IMAG = "real_imag"


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
    series_ids: list[str] = field(default_factory=list)
    series_colors: dict[str, str] = field(default_factory=dict)
    y_kind: str = "time"  # time | mag | phase | real | imag
    layout_leaf: LayoutNode | None = None

    @property
    def widget(self) -> PanelShell:
        return self.shell

    def is_frequency(self) -> bool:
        return self.y_kind != "time"


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
        self.setWindowTitle("nagilize")
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
            "Right-click plot → Panel series to remove"
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

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self._tabs.setTabsClosable(True)
        self._tabs.setMovable(False)
        self._tabs.currentChanged.connect(self._on_tab_changed)
        self._tabs.tabCloseRequested.connect(self._close_view_page)
        self._tabs.tabBarDoubleClicked.connect(self._rename_view_page)
        add_page_btn = QToolButton(self._tabs)
        add_page_btn.setText("+")
        add_page_btn.setToolTip("New page")
        add_page_btn.clicked.connect(self._new_view_page)
        self._tabs.setCornerWidget(add_page_btn, Qt.Corner.TopRightCorner)
        self._analysis_page = AnalysisPage(
            get_project=lambda: self._project,
            on_result=self._on_spectrum_result,
        )
        self._tabs.addTab(self._analysis_page, "Analysis")
        # Analysis tab is not closable.
        self._tabs.tabBar().setTabButton(0, QTabBar.ButtonPosition.LeftSide, None)
        self._tabs.tabBar().setTabButton(0, QTabBar.ButtonPosition.RightSide, None)
        self._tabs.addTab(self._make_page_host(), self._pages[0].title)
        self._tabs.setCurrentIndex(1)
        main_split.addWidget(self._tabs)
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

    # --- view pages (tabs) -----------------------------------------------

    def _is_analysis_tab(self, tab_index: int) -> bool:
        return tab_index == 0

    def _page_index_from_tab(self, tab_index: int) -> int | None:
        if tab_index <= 0:
            return None
        return tab_index - 1

    def _tab_index_from_page(self, page_index: int) -> int:
        return page_index + 1

    def _plot_host_for_page(self, page_index: int) -> QWidget | None:
        tab = self._tab_index_from_page(page_index)
        if tab < 0 or tab >= self._tabs.count():
            return None
        return self._tabs.widget(tab)

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
        tab = self._tab_index_from_page(index)
        if 0 <= tab < self._tabs.count():
            page.title = self._tabs.tabText(tab)
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

    def _set_workspace(self, layout: WorkspaceLayout) -> None:
        self._workspace = layout
        if 0 <= self._page_index < len(self._pages):
            self._pages[self._page_index].workspace = layout

    def _activate_page(self, index: int) -> None:
        page = self._pages[index]
        self._page_index = index
        self._workspace = page.workspace
        self._display_mode = page.display_mode
        self._apply_spectrum_menu(page.display_mode)

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
        page_index = self._page_index_from_tab(index)
        if page_index is None:
            if 0 <= self._page_index < len(self._pages):
                self._stash_live_to_page(self._page_index)
            self._analysis_page.refresh_series()
            return
        old = self._page_index
        if old == page_index and self._tabs.currentIndex() == self._tab_index_from_page(old):
            return
        if 0 <= old < len(self._pages):
            self._stash_live_to_page(old)
        self._activate_page(page_index)

    def _show_analysis_page(self) -> None:
        self._tabs.setCurrentIndex(0)
        self._analysis_page.refresh_series()

    def _on_spectrum_result(self, series_id: str) -> None:
        self._refresh_project_tree()
        self._analysis_page.refresh_series()
        series = self._project.get(series_id)
        name = series.name if series is not None else series_id
        self.statusBar().showMessage(
            f"Added spectrum “{name}” — drag it onto a plot cell for Mag+Phase",
            6000,
        )

    def _new_view_page(self) -> None:
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
        if self._is_analysis_tab(index):
            return
        page_index = self._page_index_from_tab(index)
        if page_index is None:
            return
        if len(self._pages) <= 1:
            self.statusBar().showMessage("Cannot close the last page", 3000)
            return
        if page_index < 0 or page_index >= len(self._pages):
            return
        closing_current = page_index == self._page_index and not self._is_analysis_tab(
            self._tabs.currentIndex()
        )
        if closing_current:
            self._stash_live_to_page(page_index)
        page = self._pages[page_index]
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
        del self._pages[page_index]
        if self._page_index > page_index:
            self._page_index -= 1
        elif self._page_index == page_index:
            self._page_index = min(page_index, len(self._pages) - 1)
        new_tab = self._tabs.currentIndex()
        self._suppress_tab_change = False
        new_page = self._page_index_from_tab(new_tab)
        if closing_current:
            if new_page is None:
                self._tabs.setCurrentIndex(self._tab_index_from_page(self._page_index))
                self._activate_page(self._page_index)
            else:
                self._activate_page(new_page)
        elif new_page is not None and new_page != self._page_index:
            self._page_index = new_page
        self.statusBar().showMessage("Page closed", 3000)

    def _rename_view_page(self, index: int) -> None:
        page_index = self._page_index_from_tab(index)
        if page_index is None or page_index < 0 or page_index >= len(self._pages):
            return
        current = self._tabs.tabText(index)
        text, ok = QInputDialog.getText(self, "Rename page", "Page name:", text=current)
        if not ok:
            return
        name = text.strip() or current
        self._pages[page_index].title = name
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
            + (" → Mag+Phase" if new_kind == "mag_phase" else ""),
            3000,
        )

    def _panel_for_viewbox(self, vb) -> _PlotPanel | None:  # noqa: ANN001
        for panel in self._panels:
            if panel.plot.getViewBox() is vb:
                return panel
        return None

    def _update_active_panel_label(self) -> None:
        return

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
        shell.set_edge_buttons_enabled(y_kind == "time")
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
        close_page_act.triggered.connect(
            lambda: self._close_view_page(self._tabs.currentIndex())
        )
        view_menu.addAction(close_page_act)
        rename_page_act = QAction("Rename page…", self)
        rename_page_act.triggered.connect(
            lambda: self._rename_view_page(self._tabs.currentIndex())
        )
        view_menu.addAction(rename_page_act)
        view_menu.addSeparator()

        cursor_dock_act = QAction("Cursor values", self)
        cursor_dock_act.setCheckable(True)
        cursor_dock_act.setChecked(False)
        cursor_dock_act.toggled.connect(self._toggle_cursor_dock)
        view_menu.addAction(cursor_dock_act)
        self._cursor_dock_action = cursor_dock_act
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

    def open_project_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open project",
            "",
            "nagilize project (*.nagproj);;All files (*)",
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
        suggested = "project.nagproj"
        if self._project.path is not None:
            suggested = str(self._project.path)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save project",
            suggested,
            "nagilize project (*.nagproj);;All files (*)",
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
                    "workspace": ws.to_dict(),
                    "markers": markers,
                    "view_ranges": view_ranges,
                    "active_panel": page.active_panel,
                }
            )
        return views

    def _load_project_into_window(self, project: Project) -> None:
        # Tear down all page widgets (keep Analysis tab at index 0).
        self._suppress_tab_change = True
        for i in range(len(self._pages) - 1, -1, -1):
            page = self._pages[i]
            if page.built and page.root_plot_widget is not None:
                self._destroy_page_runtime(page)
            tab = self._tab_index_from_page(i)
            if 0 <= tab < self._tabs.count():
                self._tabs.removeTab(tab)
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
            # Infer mag_phase leaf from assigned spectrum results.
            for leaf in workspace.leaves():
                if self._view_kind_for_series_ids(leaf.series_ids) == "mag_phase":
                    leaf.view_kind = "mag_phase"
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
                pending_markers=markers,
                pending_view_ranges=list(raw.get("view_ranges") or []),
                active_panel=int(raw.get("active_panel") or 0),
            )
            self._pages.append(page)
            self._tabs.addTab(self._make_page_host(), title)
        self._tabs.setCurrentIndex(1 if self._pages else 0)
        self._suppress_tab_change = False
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
        sid = item.data(0, ROLE_SERIES)
        if not sid:
            return
        menu = QMenu(self)
        edit = menu.addAction("Edit series metadata…")
        chosen = menu.exec(self._project_tree.viewport().mapToGlobal(pos))
        if chosen is edit:
            self._edit_series_meta(str(sid))

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
                    label=self._vertical_marker_format(),
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
                    label="y={value:.6g}",
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

    def _vertical_marker_format(self) -> str:
        return "f={value:.6g}" if self._spectrum_mode() else "t={value:.6g}"

    def _sync_marker_labels(self) -> None:
        fmt = self._vertical_marker_format()
        for group in self._synced_markers:
            if not group.vertical:
                continue
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
                if panel.is_frequency():
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

    def _draw_time_panels(self) -> None:
        for i, panel in enumerate(self._panels):
            self._draw_time_panel(panel, i)

    def _draw_time_panel(self, panel: _PlotPanel, index: int) -> None:
        units = {
            s.unit
            for sid in panel.series_ids
            for s in [self._project.get(sid)]
            if s is not None and s.unit and not s.is_spectrum()
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

    def _draw_frequency_panel(self, panel: _PlotPanel) -> None:
        label_map = {
            "mag": ("Magnitude", ""),
            "phase": ("Phase", "rad"),
            "real": ("Real", ""),
            "imag": ("Imag", ""),
        }
        y_name, y_unit = label_map.get(panel.y_kind, ("Y", ""))
        panel.plot.setLabel("bottom", "Frequency", units="Hz")
        if y_unit:
            panel.plot.setLabel("left", y_name, units=y_unit)
        else:
            panel.plot.setLabel("left", y_name)
        panel.plot.setTitle(y_name)
        for sid in panel.series_ids:
            series = self._project.get(sid)
            if series is None:
                continue
            freq, y = self._spectrum_xy(series, panel.y_kind)
            if freq.size == 0:
                continue
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
            self.setWindowTitle("nagilize")
            return
        self.statusBar().showMessage(
            f"project: {n_src} sources, {n_ser} series | panels={n_panels}"
        )
        self.setWindowTitle(f"nagilize — {src.label} (+{max(0, n_src - 1)} more)")

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
        if self._spectrum_mode() or panel.is_frequency():
            for curve in panel.curves:
                data = curve.getData()
                if data is None or data[0] is None or data[1] is None:
                    continue
                xd = np.asarray(data[0], dtype=np.float64).reshape(-1)
                yd = np.asarray(data[1], dtype=np.float64).reshape(-1)
                if xd.size == 0 or yd.size == 0 or xd.size != yd.size:
                    continue
                idx = self._nearest_index(xd, x)
                name = curve.name() or "curve"
                rows.append((name, f"{float(yd[idx]):.6g}"))
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
        freq = self._spectrum_mode() or (panel is not None and panel.is_frequency())
        x_name = "f" if freq else "t"
        x_unit = "Hz" if freq else "s"

        if self._last_view_pos is not None and panel is not None:
            x, y = self._last_view_pos
            at = f"Mouse {x_name}={x:.6g}"
            parts.append(f"{x_name}≈{x:.6g} {x_unit}")
            if freq:
                parts.append(f"y={y:.6g}")
                rows.append((at, f"{x_name} [{x_unit}]", f"{x:.6g}"))
                rows.append((at, "y", f"{y:.6g}"))
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
