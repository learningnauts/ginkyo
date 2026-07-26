"""Entry point: ``python -m ginkyo``."""

from __future__ import annotations

import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from ginkyo.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("ginkyo")
    app.setOrganizationName("ginkyo")

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
