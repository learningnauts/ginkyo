# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for ginkyo (Windows + macOS onedir GUI bundle)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(SPECPATH).resolve().parent
SRC = ROOT / "src"

block_cipher = None

a = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(SRC)],
    binaries=[],
    datas=[],
    hiddenimports=[
        "ginkyo",
        "ginkyo.__main__",
        "ginkyo.ui.main_window",
        "ginkyo.ui.analysis_page",
        "ginkyo.ui.layout_state",
        "ginkyo.ui.panel_shell",
        "ginkyo.core.measure",
        "ginkyo.core.spectrum",
        "ginkyo.core.project",
        "ginkyo.core.project_io",
        "ginkyo.core.model",
        "ginkyo.core.dummy",
        "ginkyo.readers.csv_reader",
        "ginkyo.readers.wav",
        "ginkyo.readers.uff",
        "ginkyo.export.csv_export",
        "pyuff",
    ],
    hookspath=[],
    hooksconfig={
        # Keep the Qt footprint closer to Widgets/GUI (not the whole QML stack).
        "PySide6": {
            "exclude_dlls": [
                "Qt6Quick",
                "Qt6Quick*",
                "Qt6Qml",
                "Qt6Qml*",
                "Qt6VirtualKeyboard*",
                "Qt6WebEngine*",
                "Qt6Pdf*",
                "Qt6Designer*",
            ],
        },
    },
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "IPython",
        "jupyter",
        "pytest",
        "unittest",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ginkyo",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="ginkyo",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="ginkyo.app",
        icon=None,
        bundle_identifier="app.ginkyo.desktop",
        info_plist={
            "CFBundleName": "ginkyo",
            "CFBundleDisplayName": "ginkyo",
            "CFBundleShortVersionString": "0.1.0",
            "NSHighResolutionCapable": True,
        },
    )
