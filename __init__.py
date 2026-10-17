"""Global Horse Racing Intelligence — shared data layer, ratings and market lab."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "hri.duckdb"
REPORT_DIR = ROOT / "reports"
