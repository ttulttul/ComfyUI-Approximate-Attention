from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Avoid importing the package __init__ (requires ComfyUI runtime deps).
collect_ignore = ["../__init__.py"]
