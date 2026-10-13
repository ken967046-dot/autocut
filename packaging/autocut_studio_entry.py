"""PyInstaller entry point: opens the AutoCut Studio window (same as ``python -m autocut studio``)."""
import sys

from autocut.gui.app import main

if __name__ == "__main__":
    sys.exit(main())
