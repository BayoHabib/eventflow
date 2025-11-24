"""Chicago Crime dataset adapter."""

from eventflow.datasets.chicago_crime.mapping import load_chicago_crime
from eventflow.datasets.chicago_crime.schema import CHICAGO_CRIME_SCHEMA

__all__ = [
    "load_chicago_crime",
    "CHICAGO_CRIME_SCHEMA",
]
