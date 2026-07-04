"""
pytest fixtures + path setup.

scripts/ is not a package (matches this repo's existing convention -- every
script under scripts/ does `sys.path.insert(0, str(ROOT))` itself and
scripts import each other, when needed, via importlib rather than package
imports). Tests need the same two things on sys.path: ROOT (for `src.*`
imports) and ROOT/scripts (for direct `import factor_backtest` etc).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
