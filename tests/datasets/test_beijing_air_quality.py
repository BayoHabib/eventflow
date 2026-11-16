"""Tests for the Beijing Multi-Site Air Quality dataset (UCI 501).

Covers the loader, the station schema/coordinates, and the v1 recipe,
including the exit-criterion-3 invariants:

- NA count per measurement column: after completion it grows by exactly
  the number of generated pairs (each generated pair contributes one
  null per measurement) — and is exactly identical when the panel is
  already complete, as the shipped UCI data is. No observed null ever
  becomes 0, and no generated pair ever carries a fabricated 0.0.
- No leakage across stations: one station's features never use another
  station's observations.
- Hand-checks: lags, rolling means and calendar encodings re-derived
  by hand from the raw values.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from shikong.core.event_frame import EventFrame
from shikong.core.schema import EventMetadata, RecipeConfig
from shikong.datasets.beijing_air_quality.mapping import (
    clean_beijing_data,
    load_beijing_air_quality,
)
from shikong.datasets.beijing_air_quality.recipes.beijing_air_quality_v1 import (
    BeijingAirQualityV1Recipe,
)
from shikong.datasets.beijing_air_quality.schema import (
    BEIJING_AIR_QUALITY_SCHEMA,
    MEASUREMENT_COLS,
    STATION_COORDS,
)

BASE = datetime(2024, 1, 1, 0, 0)
N_HOURS = 30
WIND_DIRS = ["N", "NE", "E", "SE"]


def _row(station: str, hour: int, base: int) -> dict[str, object]:
    """One synthetic station-hour with hand-computable values."""
    return {
        "station": station,
        "timestamp": BASE + timedelta(hours=hour),
        "count": 1,
        "PM2.5": float(base + hour),
        "PM10": float(2 * base + hour),
        "SO2": float(10 + hour),
        "NO2": float(20 + hour),
        "CO": 1.0 + 0.1 * hour,
        "O3": float(30 + hour),
        "TEMP": 5.0 + 0.1 * hour,
        "PRES": 1010.0 + 0.01 * hour,
        "DEWP": 0.0 + 0.1 * hour,
        "RAIN": 0.0,
        "WSPM": 2.0,
        "wd": WIND_DIRS[hour % len(WIND_DIRS)],
        "latitude": STATION_COORDS[station][1],
        "longitude": STATION_COORDS[station][0],
    }


def make_beijing_frame() -> EventFrame:
    """Two stations x 30 hours, with planted NAs and a 3-hour gap.

    Dongsi: all 30 hours, PM2.5 missing at hour 5, TEMP missing at
    hour 3. Tiantan: hours 10, 11, 12 missing entirely (station down),
    PM2.5 missing at hour 20. Expected completed panel: 60 rows,
    3 generated pairs, count sum 57.
    """
    rows: list[dict[str, object]] = []
    for hour in range(N_HOURS):
        row = _row("Dongsi", hour, 100)
        if hour == 5:
            row["PM2.5"] = None
        if hour == 3:
            row["TEMP"] = None
        rows.append(row)
    for hour in range(N_HOURS):
        if hour in (10, 11, 12):
            continue
        row = _row("Tiantan", hour, 200)
        if hour == 20:
            row["PM2.5"] = None
        rows.append(row)
    lf = pl.LazyFrame(rows)
    metadata = EventMetadata(dataset_name="test-beijing", crs="EPSG:4326", time_bin="1h")
    return EventFrame(lf, BEIJING_AIR_QUALITY_SCHEMA, metadata)


def make_recipe() -> BeijingAirQualityV1Recipe:
    config = RecipeConfig(
        dataset="beijing_air_quality",
        recipe="beijing_air_quality_v1",
        temporal={"time_bin": "1h"},
    )
    return BeijingAirQualityV1Recipe(config)


def _na_counts(df: pl.DataFrame, cols: list[str]) -> dict[str, int]:
    return {c: df[c].null_count() for c in cols}


class TestStationSchema:
    def test_twelve_stations_with_coordinates(self) -> None:
        assert len(STATION_COORDS) == 12
        for name, (lon, lat) in STATION_COORDS.items():
            assert 116.0 < lon < 117.0, name
            assert 39.8 < lat < 40.4, name

    def test_coordinates_match_published_dms_table(self) -> None:
        # Ma et al. (2023), PLOS ONE, Table 1 (DMS -> decimal degrees).
        dms = {
            "Aotizhongxin": ((116, 24, 3.028), (39, 59, 6.562)),
            "Changping": ((116, 13, 51.722), (40, 13, 15.427)),
            "Dingling": ((116, 13, 32.732), (40, 17, 41.435)),
            "Dongsi": ((116, 24, 55.862), (39, 55, 46.423)),
            "Guanyuan": ((116, 21, 24.548), (39, 55, 57.986)),
            "Gucheng": ((116, 11, 6.130), (39, 54, 47.041)),
            "Huairou": ((116, 37, 55.106), (40, 19, 1.211)),
            "Nongzhanguan": ((116, 27, 54.468), (39, 56, 25.872)),
            "Tiantan": ((116, 24, 38.984), (39, 52, 54.887)),
            "Wanliu": ((116, 17, 53.063), (39, 58, 1.758)),
            "Wanshouxigong": ((116, 17, 40.420), (39, 54, 34.171)),
        }
        for name, ((d1, m1, s1), (d2, m2, s2)) in dms.items():
            lon, lat = STATION_COORDS[name]
            assert lon == pytest.approx(d1 + m1 / 60 + s1 / 3600, abs=1e-6), name
            assert lat == pytest.approx(d2 + m2 / 60 + s2 / 3600, abs=1e-6), name
        # Shunyi is absent from that table; Zhang et al. (2020) Table S1.
        assert STATION_COORDS["Shunyi"] == (116.72, 40.144)

    def test_schema_names_station_panel(self) -> None:
        assert BEIJING_AIR_QUALITY_SCHEMA.timestamp_col == "timestamp"
        assert BEIJING_AIR_QUALITY_SCHEMA.lat_col == "latitude"
        assert BEIJING_AIR_QUALITY_SCHEMA.lon_col == "longitude"
        assert "station" in BEIJING_AIR_QUALITY_SCHEMA.categorical_cols
        assert set(MEASUREMENT_COLS) <= set(BEIJING_AIR_QUALITY_SCHEMA.numeric_cols)


class TestLoader:
    def _write_station_csv(self, root: Path, station: str, rows: list[dict[str, object]]) -> None:
        header = (
            "No,year,month,day,hour,PM2.5,PM10,SO2,NO2,CO,O3," "TEMP,PRES,DEWP,RAIN,wd,WSPM,station"
        )
        lines = [header]
        for i, r in enumerate(rows, start=1):
            ts = r["timestamp"]
            assert isinstance(ts, datetime)
            vals = [
                str(i),
                str(ts.year),
                str(ts.month),
                str(ts.day),
                str(ts.hour),
                "NA" if r["PM2.5"] is None else str(r["PM2.5"]),
                str(r["PM10"]),
                str(r["SO2"]),
                str(r["NO2"]),
                str(r["CO"]),
                str(r["O3"]),
                str(r["TEMP"]),
                str(r["PRES"]),
                str(r["DEWP"]),
                str(r["RAIN"]),
                str(r["wd"]),
                str(r["WSPM"]),
                station,
            ]
            lines.append(",".join(vals))
        (root / f"PRSA_Data_{station}_20130301-20170228.csv").write_text("\n".join(lines))

    def test_loader_reads_station_csvs(self, tmp_path: Path) -> None:
        rows = [_row("Dongsi", h, 100) for h in range(3)]
        rows[1]["PM2.5"] = None  # written as the literal "NA"
        self._write_station_csv(tmp_path, "Dongsi", rows)
        result = load_beijing_air_quality(str(tmp_path)).collect()
        assert result.height == 3
        assert result["timestamp"].to_list() == [BASE + timedelta(hours=h) for h in range(3)]
        assert result["count"].to_list() == [1, 1, 1]
        assert result["PM2.5"].to_list() == [100.0, None, 102.0]
        lon, lat = STATION_COORDS["Dongsi"]
        assert result["longitude"].to_list() == [lon] * 3
        assert result["latitude"].to_list() == [lat] * 3

    def test_loader_rejects_unknown_station(self, tmp_path: Path) -> None:
        self._write_station_csv(tmp_path, "Nowhere", [_row("Dongsi", 0, 100)])
        with pytest.raises(ValueError, match="Unknown Beijing station"):
            load_beijing_air_quality(str(tmp_path))

    def test_loader_rejects_empty_directory(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No PRSA_Data_"):
            load_beijing_air_quality(str(tmp_path))

    def test_clean_rejects_duplicate_station_hour(self) -> None:
        rows = [_row("Dongsi", 0, 100), _row("Dongsi", 0, 100)]
        frame = EventFrame(
            pl.LazyFrame(rows),
            BEIJING_AIR_QUALITY_SCHEMA,
            EventMetadata(dataset_name="test-beijing", crs="EPSG:4326"),
        )
        with pytest.raises(ValueError, match="Duplicate \\(station, hour\\)"):
            clean_beijing_data(frame.lazy_frame)


class TestRecipeMissingStaysMissing:
    """Exit criterion 3: a missing measurement is never turned into 0."""

    def test_na_counts_grow_only_by_generated_pairs(self) -> None:
        frame = make_beijing_frame()
        before = _na_counts(frame.collect(), MEASUREMENT_COLS)
        result = make_recipe().run(frame).collect()
        after = _na_counts(result, MEASUREMENT_COLS)
        # 3 generated pairs (Tiantan hours 10-12), each contributes one
        # null per measurement column; nothing else changes.
        n_generated = 3
        assert result.height == 60
        assert result["count"].sum() == 57
        for col in MEASUREMENT_COLS:
            assert after[col] == before[col] + n_generated, col
        assert before["PM2.5"] == 2 and after["PM2.5"] == 5
        assert before["TEMP"] == 1 and after["TEMP"] == 4
        assert before["RAIN"] == 0 and after["RAIN"] == 3

    def test_generated_pairs_have_zero_count_and_null_measurements(self) -> None:
        result = make_recipe().run(make_beijing_frame()).collect()
        generated = result.filter(
            (pl.col("station") == "Tiantan")
            & (
                pl.col("timestamp").is_between(
                    BASE + timedelta(hours=10), BASE + timedelta(hours=12)
                )
            )
        )
        assert generated.height == 3
        assert generated["count"].to_list() == [0, 0, 0]
        for col in MEASUREMENT_COLS:
            assert generated[col].null_count() == 3, col
        # Wind direction varies within a station, so it is joined and
        # null on generated pairs too.
        assert generated["wd"].null_count() == 3

    def test_observed_nulls_stay_null_and_no_zero_is_fabricated(self) -> None:
        result = make_recipe().run(make_beijing_frame()).collect()
        observed = result.filter(
            (pl.col("station") == "Dongsi") & (pl.col("timestamp") == BASE + timedelta(hours=5))
        )
        assert observed["count"].item() == 1
        assert observed["PM2.5"].item() is None
        # No PM2.5 value in the fixture is 0, so any 0.0 after the run
        # would be a fabricated reading.
        assert (result["PM2.5"] == 0.0).sum() == 0
        assert (result["PM10"] == 0.0).sum() == 0

    def test_complete_panel_on_already_complete_data_keeps_na_counts(self) -> None:
        # The shipped UCI data is a complete panel (no missing
        # station-hours), so completion adds no rows and the NA counts
        # are exactly identical — the literal criterion-3 check.
        rows = [_row("Dongsi", h, 100) for h in range(N_HOURS)]
        rows[5]["PM2.5"] = None
        frame = EventFrame(
            pl.LazyFrame(rows),
            BEIJING_AIR_QUALITY_SCHEMA,
            EventMetadata(dataset_name="test-beijing", crs="EPSG:4326", time_bin="1h"),
        )
        before = _na_counts(frame.collect(), MEASUREMENT_COLS)
        result = make_recipe().run(frame).collect()
        assert result.height == N_HOURS
        assert _na_counts(result, MEASUREMENT_COLS) == before


class TestRecipeHandChecks:
    """Lags, rolling means and calendar encodings re-derived by hand."""

    def _result(self) -> pl.DataFrame:
        return make_recipe().run(make_beijing_frame()).collect()

    def _cell(self, df: pl.DataFrame, station: str, hour: int, col: str) -> object:
        row = df.filter(
            (pl.col("station") == station) & (pl.col("timestamp") == BASE + timedelta(hours=hour))
        )
        assert row.height == 1
        return row[col].item()

    def test_lag_24_reads_same_station_yesterday_same_hour(self) -> None:
        df = self._result()
        # Dongsi hour 25 <- Dongsi hour 1 (101.0); Tiantan hour 25 <- 201.0.
        assert self._cell(df, "Dongsi", 25, "PM2.5_lag_24") == 101.0
        assert self._cell(df, "Tiantan", 25, "PM2.5_lag_24") == 201.0
        assert self._cell(df, "Dongsi", 24, "PM2.5_lag_24") == 100.0

    def test_lag_1_of_a_missing_reading_stays_null(self) -> None:
        df = self._result()
        # Dongsi hour 5 is missing, so the 1h lag at hour 6 is unknown.
        assert self._cell(df, "Dongsi", 6, "PM2.5_lag_1") is None
        assert self._cell(df, "Dongsi", 6, "PM2.5_lag_24") is None  # before hour 0
        assert self._cell(df, "Dongsi", 1, "PM2.5_lag_1") == 100.0

    def test_lag_across_the_gap_stays_null(self) -> None:
        df = self._result()
        # Tiantan hours 10-12 are generated (null), so the 1h lag at
        # hour 11 and hour 13 (which reads hour 12) are null, while the
        # lag at hour 10 reads hour 9 (209.0).
        assert self._cell(df, "Tiantan", 10, "PM2.5_lag_1") == 209.0
        assert self._cell(df, "Tiantan", 11, "PM2.5_lag_1") is None
        assert self._cell(df, "Tiantan", 13, "PM2.5_lag_1") is None

    def test_rolling_mean_24_closed_left_skips_nulls(self) -> None:
        df = self._result()
        # Dongsi hour 24: mean of hours 0..23, hour 5 missing (skipped).
        values = [100.0 + h for h in range(24) if h != 5]
        expected = sum(values) / len(values)
        assert self._cell(df, "Dongsi", 24, "PM2.5_ma_24") == pytest.approx(expected)
        # Tiantan hour 24: mean of hours 0..23, hours 10-12 missing
        # (station down) and hour 20 missing (null reading) — both skipped.
        values = [200.0 + h for h in range(24) if h not in (10, 11, 12, 20)]
        expected = sum(values) / len(values)
        assert self._cell(df, "Tiantan", 24, "PM2.5_ma_24") == pytest.approx(expected)

    def test_calendar_encodings(self) -> None:
        df = self._result()
        # 2024-01-01 is a Monday; hour 6 -> sin(2*pi*6/24) = 1.
        assert self._cell(df, "Dongsi", 6, "hour_sin") == pytest.approx(1.0)
        assert self._cell(df, "Dongsi", 0, "hour_cos") == pytest.approx(1.0)
        assert self._cell(df, "Dongsi", 0, "hour_of_day") == 0
        assert self._cell(df, "Dongsi", 0, "day_of_week") == 1  # Monday
        assert self._cell(df, "Dongsi", 0, "is_weekend") is False
        assert self._cell(df, "Dongsi", 0, "month") == 1

    def test_recipe_rejects_non_hourly_time_bin(self) -> None:
        config = RecipeConfig(
            dataset="beijing_air_quality",
            recipe="beijing_air_quality_v1",
            temporal={"time_bin": "6h"},
        )
        with pytest.raises(ValueError, match="must be '1h'"):
            BeijingAirQualityV1Recipe(config).build_pipeline()


class TestNoLeakageAcrossStations:
    """A station's features never use another station's observations."""

    def test_changing_tiantan_leaves_dongsi_features_untouched(self) -> None:
        base = make_beijing_frame()
        base_result = make_recipe().run(base).collect()

        rows = base.collect().to_dicts()
        for row in rows:
            if row["station"] == "Tiantan":
                row["PM2.5"] = 999.0
                row["PM10"] = 999.0
        perturbed = EventFrame(
            pl.LazyFrame(rows),
            BEIJING_AIR_QUALITY_SCHEMA,
            EventMetadata(dataset_name="test-beijing", crs="EPSG:4326", time_bin="1h"),
        )
        perturbed_result = make_recipe().run(perturbed).collect()

        feature_cols = [c for c in base_result.columns if c not in {"station", "timestamp"}]
        dongsi_before = base_result.filter(pl.col("station") == "Dongsi").select(feature_cols)
        dongsi_after = perturbed_result.filter(pl.col("station") == "Dongsi").select(feature_cols)
        assert dongsi_before.equals(dongsi_after)

        # ... while Tiantan's own features do change.
        tiantan_before = base_result.filter(pl.col("station") == "Tiantan").select(
            ["PM2.5_lag_24", "PM2.5_ma_24"]
        )
        tiantan_after = perturbed_result.filter(pl.col("station") == "Tiantan").select(
            ["PM2.5_lag_24", "PM2.5_ma_24"]
        )
        assert not tiantan_before.equals(tiantan_after)
