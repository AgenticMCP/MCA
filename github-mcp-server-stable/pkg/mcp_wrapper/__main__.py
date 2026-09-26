"""Entry point: ``python -m pkg.mcp_wrapper ...``."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
