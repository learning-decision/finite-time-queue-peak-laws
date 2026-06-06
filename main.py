"""
Root entrypoint: ``python main.py <config>``.

Implementation lives under ``src/qpeak/``. This file prepends ``src`` to ``sys.path``
so you can run without installing the package.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from qpeak.cli import main

if __name__ == "__main__":
    main()
