"""GeoLife GPS trajectories dataset (Microsoft Research Asia).

Source: "Geolife Trajectories 1.3.zip", Microsoft Download Center
(version 1.2.2, published 2024-07-15), 182 users, 17,621 trajectories,
April 2007 - August 2012. Each trajectory is one ``.plt`` file with a
6-line header followed by rows::

    latitude, longitude, 0, altitude_feet, days_since_1899_12_30, date, time

Representation in Shikong: one GPS fix is one event. The moving entity
is the user (``user_id``); a trajectory (``trajectory_id``, the ``.plt``
file stem) groups the fixes of one continuous log. There is no
categorical column in the raw files and no fixed spatial unit: the
entity moves, so fixed-cell semantics are evaluated per step in
``docs/design-notes/phase3-geolife-boundary.md`` — they do not all
transfer from the Chicago crime panel.
"""

from shikong.core.schema import EventMetadata, EventSchema

GEOLIFE_SCHEMA = EventSchema(
    timestamp_col="timestamp",
    lat_col="latitude",
    lon_col="longitude",
    categorical_cols=[],
    numeric_cols=["altitude_m"],
)

# Beijing sits in UTM zone 50N; all GeoLife fixes in the replay subset
# are in Beijing (lon ~116.3-116.5 E).
GEOLIFE_TARGET_CRS = "EPSG:32650"


def create_geolife_metadata(
    *,
    dataset_name: str = "geolife",
    crs: str = "EPSG:4326",
    # The GeoLife release does not declare a time zone for the .plt
    # date/time columns; they are read as naive datetimes and labelled
    # Asia/Shanghai (Beijing local time) by assumption, with no
    # conversion performed. Flag for maintainer review.
    time_zone: str = "Asia/Shanghai",
) -> EventMetadata:
    """Create typed metadata tailored for the GeoLife dataset."""
    return EventMetadata(
        dataset_name=dataset_name,
        crs=crs,
        time_zone=time_zone,
    )
