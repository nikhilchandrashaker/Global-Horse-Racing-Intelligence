"""`python run.py` rebuilds the whole database: data layer, ratings, market lab, CSV reports."""

from __future__ import annotations

import sys

from .market import build_market, export_reports
from .pipeline import build
from .ratings import build_ratings


def main() -> int:
    print("1/4 building data layer ...");  print("   ", build())
    print("2/4 computing ratings ...");    print("   ", build_ratings())
    print("3/4 market lab ...");           print("   ", build_market())
    print("4/4 exporting reports ...");    print("   ", export_reports())
    print("done. Launch the app with:  streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
