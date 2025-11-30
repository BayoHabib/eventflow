"""Schema definition for Chicago Crime dataset."""

from collections.abc import Iterable, Mapping
from typing import Any

from eventflow.core.schema import EventMetadata, EventSchema

# Chicago Crime Event Schema
# Maps to Socrata "Crimes - 2001 to Present" dataset
CHICAGO_CRIME_SCHEMA = EventSchema(
    timestamp_col="date",
    lat_col="latitude",
    lon_col="longitude",
    categorical_cols=[
        "primary_type",
        "description",
        "location_description",
        "arrest",
        "domestic",
        "beat",
        "district",
        "ward",
        "community_area",
        "fbi_code",
    ],
    numeric_cols=[
        "x_coordinate",
        "y_coordinate",
    ],
)


def create_chicago_metadata(
    *,
    dataset_name: str = "chicago_crime",
    crs: str = "EPSG:4326",
    time_zone: str = "America/Chicago",
    time_bin: str | None = None,
    grid_size_m: float | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    date_range: tuple[str, str] | None = None,
    custom: Mapping[str, Any] | None = None,
    feature_catalog: Mapping[str, Any] | None = None,
    output_modalities: Iterable[str] | None = None,
) -> EventMetadata:
    """Create typed metadata tailored for the Chicago Crime dataset."""
    metadata = EventMetadata(
        dataset_name=dataset_name,
        crs=crs,
        time_zone=time_zone,
        time_bin=time_bin,
        grid_size_m=grid_size_m,
        bounds=bounds,
        date_range=date_range,
        custom=dict(custom or {}),
        feature_catalog=dict(feature_catalog or {}),
    )

    if output_modalities is not None:
        metadata = metadata.model_copy(update={"output_modalities": set(output_modalities)})

    return metadata
