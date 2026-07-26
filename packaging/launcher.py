"""PyInstaller entry: keep this tiny so the frozen app still finds the package."""

from __future__ import annotations

from ginkyo.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
