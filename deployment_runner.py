#!/usr/bin/env python
"""Run the reproducible synthetic evidence challenge."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from threat_ai.evidence import main


if __name__ == "__main__":
    raise SystemExit(main())
