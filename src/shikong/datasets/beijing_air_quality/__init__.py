"""Beijing Multi-Site Air Quality dataset adapter (UCI 501)."""

from shikong.datasets.beijing_air_quality.mapping import load_beijing_air_quality
from shikong.datasets.beijing_air_quality.schema import (
    BEIJING_AIR_QUALITY_SCHEMA,
    MEASUREMENT_COLS,
    POLLUTANT_COLS,
    STATION_COORDS,
)

__all__ = [
    "BEIJING_AIR_QUALITY_SCHEMA",
    "MEASUREMENT_COLS",
    "POLLUTANT_COLS",
    "STATION_COORDS",
    "load_beijing_air_quality",
]
