"""Schema definition for Chicago Crime dataset."""

from eventflow.core.schema import EventSchema, EventMetadata

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


def create_chicago_metadata(**kwargs) -> EventMetadata:
    """
    Create metadata for Chicago Crime dataset.
    
    Args:
        **kwargs: Additional metadata fields
        
    Returns:
        EventMetadata instance
    """
    defaults = {
        "dataset_name": "chicago_crime",
        "crs": "EPSG:4326",  # Raw data is in WGS84
        "time_zone": "America/Chicago",
    }
    defaults.update(kwargs)
    return EventMetadata(**defaults)
