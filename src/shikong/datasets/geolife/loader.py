"""Load GeoLife ``.plt`` trajectory files into an :class:`EventFrame`.

A ``.plt`` file has a 6-line header (``Geolife trajectory``, ``WGS 84``,
``Altitude is in Feet``, ``Reserved 3``, a track header, ``0``) followed
by one GPS fix per line::

    latitude, longitude, 0, altitude_feet, days, date, time

Altitude is converted from feet to metres (``* 0.3048``). The
``days`` column (days since 1899-12-30) is cross-checked against the
``date``/``time`` columns and then dropped: the timestamp is built
from ``date`` + ``time`` only, so the frame never depends on the
Excel-epoch convention.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import polars as pl

from shikong.core.event_frame import EventFrame
from shikong.core.schema import EventMetadata
from shikong.core.utils import get_logger
from shikong.datasets.geolife.schema import GEOLIFE_SCHEMA, create_geolife_metadata

logger = get_logger(__name__)

_PLT_HEADER_LINES = 6
_FEET_TO_METRES = 0.3048
# Excel epoch used by the .plt "days" column.
_DAYS_EPOCH = date(1899, 12, 30)


def parse_plt(path: str | Path, *, user_id: str) -> pl.DataFrame:
    """Parse one ``.plt`` file into a DataFrame of GPS fixes.

    Args:
        path: Path to the ``.plt`` file.
        user_id: User identifier (the ``Data/<user>/`` folder name).

    Returns:
        DataFrame with ``user_id``, ``trajectory_id`` (file stem),
        ``timestamp``, ``latitude``, ``longitude``, ``altitude_m``.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < _PLT_HEADER_LINES:
        raise ValueError(
            f"{path}: expected a {_PLT_HEADER_LINES}-line header, got {len(lines)} lines"
        )
    if not lines[0].startswith("Geolife trajectory"):
        raise ValueError(f"{path}: unexpected header line 1: {lines[0]!r}")
    if lines[1].strip() != "WGS 84":
        raise ValueError(f"{path}: unexpected header line 2 (expected 'WGS 84'): {lines[1]!r}")

    rows: list[dict[str, object]] = []
    for line_no, line in enumerate(lines[_PLT_HEADER_LINES:], start=_PLT_HEADER_LINES + 1):
        if not line.strip():
            continue
        fields = line.split(",")
        if len(fields) != 7:
            raise ValueError(f"{path}:{line_no}: expected 7 fields, got {len(fields)}: {line!r}")
        lat, lon, _reserved, altitude_ft, days, date_str, time_str = fields
        try:
            timestamp = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: unparseable date/time: {line!r}") from exc
        # Cross-check the Excel-epoch day count against the parsed date.
        days_float = float(days)
        if abs((timestamp.date() - _DAYS_EPOCH).days - days_float) > 1.0:
            raise ValueError(
                f"{path}:{line_no}: 'days' column ({days}) disagrees with "
                f"date {date_str} by more than a day"
            )
        rows.append(
            {
                "user_id": user_id,
                "trajectory_id": path.stem,
                "timestamp": timestamp,
                "latitude": float(lat),
                "longitude": float(lon),
                "altitude_m": float(altitude_ft) * _FEET_TO_METRES,
            }
        )
    if not rows:
        raise ValueError(f"{path}: no GPS fixes after the header")
    frame = pl.DataFrame(rows, schema_overrides={"timestamp": pl.Datetime})
    return frame.sort(["timestamp"])


def load_geolife(
    paths: list[str | Path],
    *,
    user_ids: list[str] | None = None,
    metadata: EventMetadata | None = None,
) -> EventFrame:
    """Load ``.plt`` files into an :class:`EventFrame`.

    Args:
        paths: ``.plt`` files to load, one trajectory each.
        user_ids: User id per file; defaults to the parent folder name
            (the ``Data/<user>/`` folder).
        metadata: Optional metadata override.

    Returns:
        EventFrame with one row per GPS fix, in EPSG:4326.
    """
    if not paths:
        raise ValueError("load_geolife needs at least one .plt path")
    if user_ids is not None and len(user_ids) != len(paths):
        raise ValueError("user_ids must have one entry per path")
    frames = []
    for index, path in enumerate(paths):
        path = Path(path)
        user_id = user_ids[index] if user_ids is not None else path.parent.parent.name
        frames.append(parse_plt(path, user_id=user_id))
    combined = pl.concat(frames, how="vertical").sort(["user_id", "trajectory_id", "timestamp"])
    return EventFrame(
        combined.lazy(),
        GEOLIFE_SCHEMA,
        metadata or create_geolife_metadata(),
    )
