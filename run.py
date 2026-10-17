"""Build everything: `python run.py` (same as `python -m hri` with src/ on the path)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from hri.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
