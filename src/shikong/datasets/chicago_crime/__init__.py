"""Chicago Crime dataset adapter."""

from shikong.datasets.chicago_crime.mapping import load_chicago_crime
from shikong.datasets.chicago_crime.schema import CHICAGO_CRIME_SCHEMA

__all__ = [
    "load_chicago_crime",
    "CHICAGO_CRIME_SCHEMA",
]
