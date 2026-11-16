"""Schema definition for the Beijing Multi-Site Air Quality dataset (UCI 501).

Source: Chen, S. (2017). Beijing Multi-Site Air Quality [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C5RK5G

The 12 per-station CSVs carry no coordinates, so station locations
come from the published station table in Ma et al. (2023), "Spatial
and temporal characteristics analysis and prediction model of PM2.5
concentration based on SpatioTemporal-Informer model", PLOS ONE,
Table 1 (degrees/minutes/seconds, converted to decimal degrees here).
That table lists 11 of the 12 stations — Shunyi is absent — so the
Shunyi coordinate comes from Zhang et al. (2020), "Health risk
assessment of PM2.5 bound components in Beijing, China during
2013-2015", Atmospheric Environment: X, Table S1 (116.720 E,
40.144 N). Independent sources disagree by up to ~0.04 degrees
(~4 km) for some stations; the coordinates are only used to satisfy
the EventFrame schema (latitude/longitude) and to document where each
station is — no step in the v1 recipe consumes them, so that
imprecision cannot leak into any feature.
"""

from collections.abc import Iterable, Mapping
from typing import Any

from eventflow.core.schema import EventMetadata, EventSchema

# Station name -> (longitude, latitude) in decimal degrees (EPSG:4326).
# See the module docstring for the sources and their disagreement.
STATION_COORDS: dict[str, tuple[float, float]] = {
    "Aotizhongxin": (116.400841, 39.985156),
    "Changping": (116.231034, 40.220952),
    "Dingling": (116.225759, 40.294843),
    "Dongsi": (116.415517, 39.929562),
    "Guanyuan": (116.356819, 39.932774),
    "Gucheng": (116.185036, 39.913067),
    "Huairou": (116.631974, 40.317003),
    "Nongzhanguan": (116.46513, 39.94052),
    "Tiantan": (116.410829, 39.881913),
    "Wanliu": (116.298073, 39.967155),
    "Wanshouxigong": (116.294561, 39.909492),
    # Absent from Ma et al. (2023) Table 1; Zhang et al. (2020) Table S1.
    "Shunyi": (116.72, 40.144),
}

STATIONS: tuple[str, ...] = tuple(sorted(STATION_COORDS))

# Six air pollutants (ug/m^3, except CO in mg/m^3).
POLLUTANT_COLS: list[str] = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]

# Five numeric meteorological variables. Wind direction ("wd") is
# categorical and listed separately.
METEOROLOGICAL_COLS: list[str] = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]

# Every numeric measurement in the files: pollutants + meteorology.
# These are measurements, never counts — the recipe declares them as
# CompletePanelStep measure_cols so a missing reading stays null.
MEASUREMENT_COLS: list[str] = POLLUTANT_COLS + METEOROLOGICAL_COLS

# Beijing Multi-Site Air Quality Event Schema.
# The panel unit is the station; the timestamp is the hourly bin.
# "count" is one per observed station-hour (added by the loader) and is
# the only zero-filled column at panel completion.
BEIJING_AIR_QUALITY_SCHEMA = EventSchema(
    timestamp_col="timestamp",
    lat_col="latitude",
    lon_col="longitude",
    categorical_cols=[
        "station",
        "wd",
    ],
    numeric_cols=MEASUREMENT_COLS + ["count"],
)


def create_beijing_metadata(
    *,
    dataset_name: str = "beijing_air_quality",
    crs: str = "EPSG:4326",
    time_zone: str = "Asia/Shanghai",
    time_bin: str | None = None,
    date_range: tuple[str, str] | None = None,
    custom: Mapping[str, Any] | None = None,
    feature_catalog: Mapping[str, Any] | None = None,
    output_modalities: Iterable[str] | None = None,
) -> EventMetadata:
    """Create typed metadata tailored for the Beijing Air Quality dataset.

    Timestamps in the UCI files are naive local time (no timezone
    attached); ``time_zone`` records the local zone they were taken in.
    """
    metadata = EventMetadata(
        dataset_name=dataset_name,
        crs=crs,
        time_zone=time_zone,
        time_bin=time_bin,
        date_range=date_range,
        custom=dict(custom or {}),
        feature_catalog=dict(feature_catalog or {}),
    )

    if output_modalities is not None:
        metadata = metadata.model_copy(update={"output_modalities": set(output_modalities)})

    return metadata
