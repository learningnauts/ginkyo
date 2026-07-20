"""Entry point: ``python -m nagilize``."""

from __future__ import annotations

import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from nagilize.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("nagilize")
    app.setOrganizationName("nagilize")

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
