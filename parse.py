"""Small, well-tested parsers for the messy fields in the raw racing files."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Approximate lengths for the short-margin words used by HKJC and JRA.
_MARGIN_WORDS = {
    "DH": 0.0,
    "N": 0.05,
    "NOSE": 0.05,
    "SH": 0.1,
    "SHD": 0.1,
    "HD": 0.2,
    "NK": 0.3,
}


def parse_margin(value) -> float:
    """Lengths behind the winner. '1-1/4' -> 1.25, 'HD' -> 0.2, unparseable -> NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = str(value).strip().upper()
    if s in {"", "-", "--", "---", "NAN", "NONE"}:
        return np.nan
    if s in _MARGIN_WORDS:
        return _MARGIN_WORDS[s]
    try:
        total = 0.0
        for tok in s.replace("-", " ").split():
            if "/" in tok:
                a, b = tok.split("/")
                total += int(a) / int(b)
            else:
                total += float(tok)
        return total
    except (ValueError, ZeroDivisionError):
        return np.nan


def parse_time(value) -> float:
    """Finish time in seconds. HKJC '1.39.58' and JRA '1:56.0' both -> seconds."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = str(value).strip()
    m = re.fullmatch(r"(\d+)[:.](\d{1,2})\.(\d{1,2})", s)
    if m:
        return int(m[1]) * 60 + int(m[2]) + int(m[3]) / 10 ** len(m[3])
    m = re.fullmatch(r"(\d{1,2})\.(\d{1,2})", s)
    if m:
        return int(m[1]) + int(m[2]) / 10 ** len(m[2])
    return np.nan


def parse_race_no(value) -> float:
    """'Race 7' -> 7. Non-numeric ('oversea') -> NaN."""
    m = re.search(r"(\d+)", str(value))
    return float(m.group(1)) if m else np.nan


def parse_distance(value) -> float:
    """'1650M' or 1650 -> 1650."""
    m = re.search(r"(\d+)", str(value))
    return float(m.group(1)) if m else np.nan


def parse_decimal_comma(series: pd.Series) -> pd.Series:
    """'81,82' -> 81.82 (Ecuador file uses decimal commas)."""
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def hk_surface(track: str) -> tuple[str, str | None]:
    """HKJC track text -> (surface, course letter). 'TURF - "A+3" Course' -> ('TURF', 'A+3')."""
    t = str(track).upper()
    if "ALL WEATHER" in t:
        return "AWT", None
    m = re.search(r'"([^"]+)"', t)
    return "TURF", (m.group(1) if m else None)


def hk_finish(position) -> tuple[float, str]:
    """HKJC result_position -> (finishing position, status). '03' -> (3, 'FIN'); 'WV' -> (nan, 'WV')."""
    s = str(position).strip().upper()
    if s.isdigit():
        return float(int(s)), "FIN"
    return np.nan, s or "UNK"


def jra_finish(position) -> tuple[float, str]:
    """JRA final_position '3rd' -> (3, 'FIN'); missing -> (nan, 'NF')."""
    if position is None or (isinstance(position, float) and np.isnan(position)):
        return np.nan, "NF"
    m = re.match(r"(\d+)", str(position))
    return (float(m.group(1)), "FIN") if m else (np.nan, str(position).upper())
