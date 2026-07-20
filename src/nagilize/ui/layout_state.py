"""Workspace layout: nested splitter tree + named presets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


_MAX_LEAVES = 9


@dataclass
class LayoutNode:
    """Binary layout node: leaf holds series; split holds orientation + two children."""

    kind: str = "leaf"  # leaf | split
    series_ids: list[str] = field(default_factory=list)
    series_colors: dict[str, str] = field(default_factory=dict)
    # time = single plot; mag_phase = that cell becomes Mag (top) / Phase (bottom)
    view_kind: str = "time"
    orientation: str = "horizontal"  # horizontal | vertical
    sizes: list[float] = field(default_factory=lambda: [0.5, 0.5])
    children: list[LayoutNode] = field(default_factory=list)

    def is_leaf(self) -> bool:
        return self.kind == "leaf"

    def iter_leaves(self) -> list[LayoutNode]:
        if self.is_leaf():
            return [self]
        out: list[LayoutNode] = []
        for child in self.children:
            out.extend(child.iter_leaves())
        return out

    def leaf_count(self) -> int:
        return len(self.iter_leaves())

    def prune_series_ids(self, prune: Callable[[list[str]], list[str]]) -> None:
        if self.is_leaf():
            self.series_ids = prune(list(self.series_ids))
            self.series_colors = {
                k: v for k, v in self.series_colors.items() if k in self.series_ids
            }
            return
        for child in self.children:
            child.prune_series_ids(prune)

    def to_dict(self) -> dict:
        if self.is_leaf():
            data = {
                "kind": "leaf",
                "series_ids": list(self.series_ids),
                "view_kind": self.view_kind or "time",
            }
            if self.series_colors:
                data["series_colors"] = dict(self.series_colors)
            return data
        return {
            "kind": "split",
            "orientation": self.orientation,
            "sizes": list(self.sizes),
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, data: dict) -> LayoutNode:
        kind = str(data.get("kind") or "leaf")
        if kind == "split":
            children = [cls.from_dict(c) for c in (data.get("children") or [])]
            if len(children) < 2:
                while len(children) < 2:
                    children.append(cls(kind="leaf"))
            sizes = [float(x) for x in (data.get("sizes") or [0.5, 0.5])]
            if len(sizes) != len(children):
                sizes = [1.0 / len(children)] * len(children)
            return cls(
                kind="split",
                orientation=str(data.get("orientation") or "horizontal"),
                sizes=sizes,
                children=children,
            )
        ids = data.get("series_ids")
        if ids is None and "channels" in data:
            ids = []
        colors_raw = data.get("series_colors") or {}
        colors = {str(k): str(v) for k, v in dict(colors_raw).items()}
        view_kind = str(data.get("view_kind") or "time")
        if view_kind not in ("time", "mag_phase"):
            view_kind = "time"
        return cls(
            kind="leaf",
            series_ids=list(ids or []),
            series_colors=colors,
            view_kind=view_kind,
        )

    @classmethod
    def leaf(cls, series_ids: list[str] | None = None) -> LayoutNode:
        return cls(kind="leaf", series_ids=list(series_ids or []))

    @classmethod
    def split(
        cls,
        orientation: str,
        left: LayoutNode,
        right: LayoutNode,
        sizes: list[float] | None = None,
    ) -> LayoutNode:
        return cls(
            kind="split",
            orientation=orientation,
            sizes=list(sizes or [0.5, 0.5]),
            children=[left, right],
        )


def _chain_split(orientation: str, nodes: list[LayoutNode]) -> LayoutNode:
    """Nest binary splits so each leaf gets an equal share (1/n)."""
    if len(nodes) == 1:
        return nodes[0]
    if len(nodes) == 2:
        return LayoutNode.split(orientation, nodes[0], nodes[1], sizes=[0.5, 0.5])
    n = len(nodes)
    rest = _chain_split(orientation, nodes[1:])
    return LayoutNode.split(
        orientation, nodes[0], rest, sizes=[1.0 / n, (n - 1) / float(n)]
    )


def _grid_to_tree(rows: int, cols: int, leaves: list[LayoutNode]) -> LayoutNode:
    """Build nested splits approximating a rows×cols grid (row-major, equal shares)."""
    assert len(leaves) >= rows * cols

    def row_node(r: int) -> LayoutNode:
        cells = [leaves[r * cols + c] for c in range(cols)]
        return _chain_split("horizontal", cells)

    row_nodes = [row_node(r) for r in range(rows)]
    return _chain_split("vertical", row_nodes)


def _fill_leaves(n: int, series_ids: list[str]) -> list[LayoutNode]:
    return [
        LayoutNode.leaf([series_ids[i]] if i < len(series_ids) else [])
        for i in range(n)
    ]


@dataclass
class WorkspaceLayout:
    root: LayoutNode = field(default_factory=LayoutNode.leaf)
    display_mode: str = "time"
    preset_id: str = "single"

    def leaves(self) -> list[LayoutNode]:
        return self.root.iter_leaves()

    def prune_series(self, prune: Callable[[list[str]], list[str]]) -> None:
        self.root.prune_series_ids(prune)

    def to_dict(self) -> dict:
        return {
            "version": 2,
            "display_mode": self.display_mode,
            "preset_id": self.preset_id,
            "root": self.root.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> WorkspaceLayout:
        if "root" in data:
            root = LayoutNode.from_dict(data["root"])
            return cls(
                root=root,
                display_mode=str(data.get("display_mode") or "time"),
                preset_id=str(data.get("preset_id") or ""),
            )
        panels_raw = data.get("panels") or []
        panels: list[LayoutNode] = []
        for p in panels_raw:
            ids = p.get("series_ids")
            if ids is None and "channels" in p:
                ids = []
            panels.append(LayoutNode.leaf(list(ids or [])))
        rows = max(1, int(data.get("rows") or 1))
        cols = max(1, int(data.get("cols") or 1))
        while len(panels) < rows * cols:
            panels.append(LayoutNode.leaf())
        root = _grid_to_tree(rows, cols, panels[: rows * cols])
        return cls(
            root=root,
            display_mode=str(data.get("display_mode") or "time"),
            preset_id="",
        )

    @classmethod
    def default_for_series(cls, series_ids: list[str]) -> WorkspaceLayout:
        n = len(series_ids)
        if n <= 1:
            return build_preset("single", series_ids)
        if n == 2:
            return build_preset("dual_v", series_ids)
        if n == 3:
            return build_preset("left2_right1", series_ids)
        return build_preset("grid_2x2", series_ids)


def _register_presets() -> dict[str, tuple[str, Callable[[list[str]], WorkspaceLayout]]]:
    def single(ids: list[str]) -> WorkspaceLayout:
        lay = WorkspaceLayout(root=_fill_leaves(1, ids)[0], preset_id="single")
        return lay

    def dual_v(ids: list[str]) -> WorkspaceLayout:
        a, b = _fill_leaves(2, ids)
        return WorkspaceLayout(
            root=LayoutNode.split("vertical", a, b), preset_id="dual_v"
        )

    def dual_h(ids: list[str]) -> WorkspaceLayout:
        a, b = _fill_leaves(2, ids)
        return WorkspaceLayout(
            root=LayoutNode.split("horizontal", a, b), preset_id="dual_h"
        )

    def triple_v(ids: list[str]) -> WorkspaceLayout:
        a, b, c = _fill_leaves(3, ids)
        return WorkspaceLayout(
            root=LayoutNode.split(
                "vertical", a, LayoutNode.split("vertical", b, c), sizes=[1 / 3, 2 / 3]
            ),
            preset_id="triple_v",
        )

    def triple_h(ids: list[str]) -> WorkspaceLayout:
        a, b, c = _fill_leaves(3, ids)
        return WorkspaceLayout(
            root=LayoutNode.split(
                "horizontal",
                a,
                LayoutNode.split("horizontal", b, c),
                sizes=[1 / 3, 2 / 3],
            ),
            preset_id="triple_h",
        )

    def left2_right1(ids: list[str]) -> WorkspaceLayout:
        a, b, c = _fill_leaves(3, ids)
        return WorkspaceLayout(
            root=LayoutNode.split(
                "horizontal",
                LayoutNode.split("vertical", a, b),
                c,
                sizes=[0.5, 0.5],
            ),
            preset_id="left2_right1",
        )

    def left1_right2(ids: list[str]) -> WorkspaceLayout:
        a, b, c = _fill_leaves(3, ids)
        return WorkspaceLayout(
            root=LayoutNode.split(
                "horizontal",
                a,
                LayoutNode.split("vertical", b, c),
                sizes=[0.5, 0.5],
            ),
            preset_id="left1_right2",
        )

    def top1_bottom2(ids: list[str]) -> WorkspaceLayout:
        a, b, c = _fill_leaves(3, ids)
        return WorkspaceLayout(
            root=LayoutNode.split(
                "vertical",
                a,
                LayoutNode.split("horizontal", b, c),
                sizes=[0.5, 0.5],
            ),
            preset_id="top1_bottom2",
        )

    def grid_2x2(ids: list[str]) -> WorkspaceLayout:
        leaves = _fill_leaves(4, ids)
        return WorkspaceLayout(root=_grid_to_tree(2, 2, leaves), preset_id="grid_2x2")

    def grid_2x3(ids: list[str]) -> WorkspaceLayout:
        leaves = _fill_leaves(6, ids)
        return WorkspaceLayout(root=_grid_to_tree(2, 3, leaves), preset_id="grid_2x3")

    def grid_3x2(ids: list[str]) -> WorkspaceLayout:
        leaves = _fill_leaves(6, ids)
        return WorkspaceLayout(root=_grid_to_tree(3, 2, leaves), preset_id="grid_3x2")

    return {
        "single": ("1 — Single", single),
        "dual_v": ("2 — Dual vertical", dual_v),
        "dual_h": ("2 — Dual horizontal", dual_h),
        "triple_v": ("3 — Triple vertical", triple_v),
        "triple_h": ("3 — Triple horizontal", triple_h),
        "left2_right1": ("3 — Left 2 + Right 1", left2_right1),
        "left1_right2": ("3 — Left 1 + Right 2", left1_right2),
        "top1_bottom2": ("3 — Top 1 + Bottom 2", top1_bottom2),
        "grid_2x2": ("4 — 2×2 grid", grid_2x2),
        "grid_2x3": ("6 — 2×3 grid", grid_2x3),
        "grid_3x2": ("6 — 3×2 grid", grid_3x2),
    }


LAYOUT_PRESETS = _register_presets()


def preset_ids() -> list[str]:
    return list(LAYOUT_PRESETS.keys())


def preset_label(preset_id: str) -> str:
    return LAYOUT_PRESETS[preset_id][0]


def build_preset(preset_id: str, series_ids: list[str] | None = None) -> WorkspaceLayout:
    if preset_id not in LAYOUT_PRESETS:
        raise KeyError(f"Unknown layout preset: {preset_id}")
    return LAYOUT_PRESETS[preset_id][1](list(series_ids or []))


def layouts_dir() -> Path:
    return Path.home() / ".config" / "nagilize" / "layouts"


def save_layout(path: str | Path, layout: WorkspaceLayout) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(layout.to_dict(), indent=2), encoding="utf-8")


def load_layout(path: str | Path) -> WorkspaceLayout:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Layout file must be a JSON object")
    return WorkspaceLayout.from_dict(data)
