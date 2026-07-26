"""Panel shell: drop target for series DnD (layout chrome lives on splitters)."""

from __future__ import annotations

import json
import weakref

from PySide6.QtCore import QMimeData, Qt
from PySide6.QtGui import QDrag
from PySide6.QtWidgets import (
    QFrame,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

SERIES_MIME = "application/x-ginkyo-series-ids"
ROLE_SOURCE = int(Qt.ItemDataRole.UserRole)
ROLE_SERIES = int(Qt.ItemDataRole.UserRole) + 1


class SeriesTree(QTreeWidget):
    """Project series list that starts drags with series ids."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setHeaderLabels(["Series"])
        self.setDragEnabled(True)
        self.setDragDropMode(QTreeWidget.DragOnly)
        self.setDefaultDropAction(Qt.DropAction.CopyAction)

    def selected_series_ids(self) -> list[str]:
        ids: list[str] = []
        seen: set[str] = set()
        for item in self.selectedItems():
            for sid in self._ids_from_item(item):
                if sid not in seen:
                    seen.add(sid)
                    ids.append(sid)
        return ids

    def _ids_from_item(self, item: QTreeWidgetItem) -> list[str]:
        sid = item.data(0, ROLE_SERIES)
        if sid:
            return [str(sid)]
        out: list[str] = []
        for i in range(item.childCount()):
            child_sid = item.child(i).data(0, ROLE_SERIES)
            if child_sid:
                out.append(str(child_sid))
        return out

    def startDrag(self, supportedActions) -> None:  # noqa: N802
        ids = self.selected_series_ids()
        if not ids:
            return
        mime = QMimeData()
        payload = json.dumps(ids).encode("utf-8")
        mime.setData(SERIES_MIME, payload)
        mime.setText(", ".join(ids))
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.CopyAction)


class PanelShell(QFrame):
    """Wraps a plot and accepts series drops."""

    def __init__(self, host: object, plot_widget: QWidget) -> None:
        super().__init__()
        self._host_ref = weakref.ref(host)
        self._plot_widget = plot_widget
        self.panel_ref: object | None = None
        self.setObjectName("PanelShell")
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(plot_widget)

    def host(self):
        return self._host_ref() if self._host_ref is not None else None

    def set_edge_buttons_enabled(self, _enabled: bool) -> None:
        """Kept for call-site compatibility; edge '+' removed in favor of splitter '+'."""

    def hide_plus(self) -> None:
        """No-op (edge '+' removed)."""

    def _set_drop_style(self, active: bool) -> None:
        if active:
            self.setStyleSheet(
                "#PanelShell { border: 2px solid #1f4e79; background: rgba(31,78,121,20); }"
            )
        else:
            self.setStyleSheet("")

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasFormat(SERIES_MIME):
            event.acceptProposedAction()
            self._set_drop_style(True)
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasFormat(SERIES_MIME):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self._set_drop_style(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802
        self._set_drop_style(False)
        raw = bytes(event.mimeData().data(SERIES_MIME)).decode("utf-8")
        try:
            ids = json.loads(raw)
        except json.JSONDecodeError:
            event.ignore()
            return
        if not isinstance(ids, list) or not ids:
            event.ignore()
            return
        host = self.host()
        if host is None or self.panel_ref is None:
            event.ignore()
            return
        replace = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        getattr(host, "assign_series_to_panel")(
            self.panel_ref, [str(x) for x in ids], replace=replace
        )
        event.acceptProposedAction()
