"""STPP exact-event emitters: extraction, splits, Gatech/catalog/tick.

All fixtures are synthetic, built in memory with hand-computed
expected values. No real data lives in ``tests/`` (the CI has no CSVs).

Hand-computed reference sequence (5 events, 3 marks):

    t = 0s    THEFT     (x, y) from lon/lat below
    t = 10s   BATTERY
    t = 30s   THEFT
    t = 35s   ASSAULT
    t = 100s  BATTERY

Codebook (sorted): ASSAULT=0, BATTERY=1, THEFT=2.

Gatech fields for that sequence:

    idx  tss   tsl  tsls  type
    0    0     0    0     2
    1    10    10   0     1
    2    30    20   30    2
    3    35    5    0     0
    4    100   65   90    1
"""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from pyproj import Transformer

from shikong.core.event_frame import EventFrame
from shikong.core.schema import EventMetadata, EventSchema
from shikong.stpp.emitters import (
    CATALOG_COLUMNS,
    GATECH_KEYS,
    NO_FITTED_MODEL_STATEMENT,
    SINGLE_MARK_LABEL,
    ChronologicalSplits,
    SplitBoundary,
    build_gatech_sequence,
    build_manifest,
    build_tick_payload,
    chronological_event_splits,
    emit_catalog,
    emit_gatech,
    emit_tick,
    events_to_txy,
    extract_exact_events,
    write_manifest,
)

T0 = datetime(2024, 1, 1, 0, 0, 0)
# (seconds offset, mark, lon, lat) — lon/lat near downtown Chicago.
_REF_ROWS = [
    (0, "THEFT", -87.6298, 41.8781),
    (10, "BATTERY", -87.6300, 41.8782),
    (30, "THEFT", -87.6290, 41.8790),
    (35, "ASSAULT", -87.6310, 41.8770),
    (100, "BATTERY", -87.6280, 41.8800),
]
_CODEBOOK = ["ASSAULT", "BATTERY", "THEFT"]
_TARGET_CRS = "EPSG:26971"


def _frame(rows: list[dict], *, mark_col: str | None = "primary_type") -> EventFrame:
    schema = EventSchema(
        timestamp_col="date",
        lat_col="latitude",
        lon_col="longitude",
        categorical_cols=[mark_col] if mark_col else [],
    )
    metadata = EventMetadata(dataset_name="test-stpp", crs="EPSG:4326", time_zone="America/Chicago")
    return EventFrame(pl.LazyFrame(rows), schema, metadata)


def _ref_frame() -> EventFrame:
    rows = [
        {
            "date": T0 + timedelta(seconds=s),
            "latitude": lat,
            "longitude": lon,
            "primary_type": mark,
        }
        for s, mark, lon, lat in _REF_ROWS
    ]
    return _frame(rows)


def _ref_events() -> pl.DataFrame:
    return extract_exact_events(_ref_frame(), mark_col="primary_type", target_crs=_TARGET_CRS)


def _independent_xy(lon: float, lat: float) -> tuple[float, float]:
    """Projection computed independently (test-side transformer)."""
    transformer = Transformer.from_crs("EPSG:4326", _TARGET_CRS, always_xy=True)
    x_m, y_m = transformer.transform([lon], [lat])
    return float(x_m[0]) / 1000.0, float(y_m[0]) / 1000.0


def _sha256_tree(directory: Path) -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.iterdir())
        if p.is_file()
    }


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


class TestExtract:
    def test_columns_order_and_chronological_sort(self) -> None:
        # Deliberately shuffled input: output must be time-sorted.
        rows = [
            {
                "date": T0 + timedelta(seconds=s),
                "latitude": lat,
                "longitude": lon,
                "primary_type": m,
            }
            for s, m, lon, lat in reversed(_REF_ROWS)
        ]
        out = extract_exact_events(_frame(rows), mark_col="primary_type", target_crs=_TARGET_CRS)
        assert out.columns == [
            "timestamp",
            "longitude",
            "latitude",
            "mark",
            "mark_label",
            "x",
            "y",
        ]
        assert out["timestamp"].to_list() == [T0 + timedelta(seconds=s) for s, _, _, _ in _REF_ROWS]
        assert out["mark"].to_list() == [2, 1, 2, 0, 1]
        assert out["mark_label"].to_list() == ["THEFT", "BATTERY", "THEFT", "ASSAULT", "BATTERY"]

    def test_projection_matches_independent_pyproj(self) -> None:
        out = _ref_events()
        for row, (_, _, lon, lat) in zip(out.iter_rows(named=True), _REF_ROWS, strict=True):
            x, y = _independent_xy(lon, lat)
            assert row["x"] == pytest.approx(x)
            assert row["y"] == pytest.approx(y)
            assert row["longitude"] == lon
            assert row["latitude"] == lat

    def test_codebook_is_sorted_unique(self) -> None:
        out = _ref_events()
        assert sorted(set(out["mark_label"].to_list())) == _CODEBOOK
        assert out.filter(pl.col("mark_label") == "ASSAULT")["mark"].to_list() == [0]

    def test_no_mark_column_gives_single_fix_mark(self) -> None:
        rows = [
            {"date": T0 + timedelta(seconds=s), "latitude": lat, "longitude": lon}
            for s, _, lon, lat in _REF_ROWS
        ]
        out = extract_exact_events(
            _frame(rows, mark_col=None), mark_col=None, target_crs=_TARGET_CRS
        )
        assert out["mark"].to_list() == [0] * 5
        assert out["mark_label"].to_list() == [SINGLE_MARK_LABEL] * 5

    def test_null_timestamp_rejected(self) -> None:
        rows = [{"date": None, "latitude": 41.8781, "longitude": -87.6298, "primary_type": "THEFT"}]
        with pytest.raises(ValueError, match="null"):
            extract_exact_events(_frame(rows), mark_col="primary_type", target_crs=_TARGET_CRS)

    def test_null_coordinate_rejected(self) -> None:
        rows = [
            {"date": T0, "latitude": None, "longitude": -87.6298, "primary_type": "THEFT"},
            {"date": T0, "latitude": 41.8781, "longitude": None, "primary_type": "THEFT"},
        ]
        for row in rows:
            with pytest.raises(ValueError, match="null"):
                extract_exact_events(_frame([row]), mark_col="primary_type", target_crs=_TARGET_CRS)

    def test_null_mark_rejected(self) -> None:
        rows = [{"date": T0, "latitude": 41.8781, "longitude": -87.6298, "primary_type": None}]
        with pytest.raises(ValueError, match="null"):
            extract_exact_events(_frame(rows), mark_col="primary_type", target_crs=_TARGET_CRS)

    def test_non_finite_coordinate_rejected(self) -> None:
        rows = [{"date": T0, "latitude": float("inf"), "longitude": -87.6298, "primary_type": "X"}]
        with pytest.raises(ValueError, match="non-finite"):
            extract_exact_events(_frame(rows), mark_col="primary_type", target_crs=_TARGET_CRS)

    def test_missing_mark_column_rejected(self) -> None:
        with pytest.raises(ValueError, match="mark column"):
            extract_exact_events(_ref_frame(), mark_col="no_such_column", target_crs=_TARGET_CRS)

    def test_missing_timestamp_column_rejected(self) -> None:
        schema = EventSchema(timestamp_col="missing_time", lat_col="latitude", lon_col="longitude")
        metadata = EventMetadata(dataset_name="test-stpp", crs="EPSG:4326")
        frame = EventFrame(pl.LazyFrame([{"latitude": 41.0, "longitude": -87.0}]), schema, metadata)
        with pytest.raises(ValueError, match="not present"):
            extract_exact_events(frame, mark_col=None, target_crs=_TARGET_CRS)

    def test_schema_without_lat_lon_rejected(self) -> None:
        schema = EventSchema(timestamp_col="date", geometry_col="geom")
        metadata = EventMetadata(dataset_name="test-stpp", crs="EPSG:4326")
        frame = EventFrame(pl.LazyFrame([{"date": T0, "geom": "POINT (0 0)"}]), schema, metadata)
        with pytest.raises(ValueError, match="no lat/lon"):
            extract_exact_events(frame, mark_col=None, target_crs=_TARGET_CRS)

    def test_empty_frame_rejected(self) -> None:
        schema = EventSchema(timestamp_col="date", lat_col="latitude", lon_col="longitude")
        metadata = EventMetadata(dataset_name="test-stpp", crs="EPSG:4326")
        frame = EventFrame(
            pl.LazyFrame(
                [],
                schema={"date": pl.Datetime, "latitude": pl.Float64, "longitude": pl.Float64},
            ),
            schema,
            metadata,
        )
        with pytest.raises(ValueError, match="empty"):
            extract_exact_events(frame, mark_col=None, target_crs=_TARGET_CRS)


# ---------------------------------------------------------------------------
# Gatech
# ---------------------------------------------------------------------------


class TestGatech:
    _EXPECTED = [
        {
            "idx_event": 0,
            "time_since_start": 0.0,
            "time_since_last_event": 0.0,
            "time_since_last_same_event": 0.0,
            "type_event": 2,
        },
        {
            "idx_event": 1,
            "time_since_start": 10.0,
            "time_since_last_event": 10.0,
            "time_since_last_same_event": 0.0,
            "type_event": 1,
        },
        {
            "idx_event": 2,
            "time_since_start": 30.0,
            "time_since_last_event": 20.0,
            "time_since_last_same_event": 30.0,
            "type_event": 2,
        },
        {
            "idx_event": 3,
            "time_since_start": 35.0,
            "time_since_last_event": 5.0,
            "time_since_last_same_event": 0.0,
            "type_event": 0,
        },
        {
            "idx_event": 4,
            "time_since_start": 100.0,
            "time_since_last_event": 65.0,
            "time_since_last_same_event": 90.0,
            "type_event": 1,
        },
    ]

    def test_sequence_fields_hand_computed(self) -> None:
        seq = build_gatech_sequence(_ref_events())
        assert seq == self._EXPECTED

    def test_sequence_keys_exact(self) -> None:
        for event in build_gatech_sequence(_ref_events()):
            assert list(event.keys()) == list(GATECH_KEYS)

    def test_first_of_each_type_has_zero_same_gap(self) -> None:
        seq = build_gatech_sequence(_ref_events())
        first_seen: dict[int, dict] = {}
        for event in seq:
            first_seen.setdefault(event["type_event"], event)
        for event in first_seen.values():
            assert event["time_since_last_same_event"] == 0.0

    def test_time_since_start_non_decreasing(self) -> None:
        seq = build_gatech_sequence(_ref_events())
        starts = [e["time_since_start"] for e in seq]
        assert starts == sorted(starts)

    def test_empty_split_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            build_gatech_sequence(_ref_events().head(0))

    def test_emit_round_trip_pickle_and_json(self, tmp_path: Path) -> None:
        splits = {"train": _ref_events(), "test": _ref_events().head(2)}
        written = emit_gatech(splits, output_dir=tmp_path, dim_process=3)
        assert {p.name for p in written} == {
            "train.pkl",
            "train.json",
            "test.pkl",
            "test.json",
        }
        # THP-style read: pickle.load with latin-1.
        with (tmp_path / "train.pkl").open("rb") as f:
            payload = pickle.load(f, encoding="latin-1")
        assert payload["dim_process"] == 3
        assert payload["train"] == [self._EXPECTED]
        assert set(payload.keys()) == {"dim_process", "train"}
        # JSON holds the same content.
        assert json.loads((tmp_path / "train.json").read_text(encoding="utf-8")) == payload
        with (tmp_path / "test.pkl").open("rb") as f:
            test_payload = pickle.load(f, encoding="latin-1")
        assert test_payload["test"] == [self._EXPECTED[:2]]


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def _split_events() -> pl.DataFrame:
    """12 extracted events, one per day 2024-01-01 .. 2024-01-12, mark 0."""
    rows = [
        {
            "timestamp": datetime(2024, 1, 1) + timedelta(days=i),
            "longitude": -87.6,
            "latitude": 41.8,
            "mark": 0,
            "mark_label": "THEFT",
            "x": 350.0 + i,
            "y": 578.0,
        }
        for i in range(12)
    ]
    return pl.DataFrame(rows, schema_overrides={"timestamp": pl.Datetime})


class TestSplits:
    _BOUNDARIES = [
        SplitBoundary("aux", datetime(2024, 1, 3)),
        SplitBoundary("train", datetime(2024, 1, 7), timedelta(days=2)),
        SplitBoundary("dev", datetime(2024, 1, 10), timedelta(days=1)),
        SplitBoundary("test", datetime(2024, 1, 13)),
    ]

    def test_counts_and_embargoes(self) -> None:
        # aux [01-01, 01-03): 2; train [01-03, 01-07): 4;
        # embargo 01-07..01-09: 2; dev [01-09, 01-10): 1;
        # embargo 01-10..01-11: 1; test [01-11, 01-13): 2.
        result = chronological_event_splits(_split_events(), boundaries=self._BOUNDARIES)
        assert isinstance(result, ChronologicalSplits)
        assert {k: v.height for k, v in result.splits.items()} == {
            "aux": 2,
            "train": 4,
            "dev": 1,
            "test": 2,
        }
        assert result.embargo_counts == {"embargo_after_train": 2, "embargo_after_dev": 1}
        assert result.embargo_intervals == {
            "embargo_after_train": (datetime(2024, 1, 7), datetime(2024, 1, 9)),
            "embargo_after_dev": (datetime(2024, 1, 10), datetime(2024, 1, 11)),
        }
        assert result.trailing_count == 0
        total = sum(v.height for v in result.splits.values()) + sum(result.embargo_counts.values())
        assert total == 12

    def test_chronology_no_event_in_two_splits(self) -> None:
        result = chronological_event_splits(_split_events(), boundaries=self._BOUNDARIES)
        ordered = [result.splits[name] for name in ("aux", "train", "dev", "test")]
        for earlier, later in zip(ordered, ordered[1:], strict=False):
            assert earlier["timestamp"].max() < later["timestamp"].min()
        seen: set[datetime] = set()
        for frame in ordered:
            for ts in frame["timestamp"].to_list():
                assert ts not in seen
                seen.add(ts)

    def test_no_boundaries_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one boundary"):
            chronological_event_splits(_split_events(), boundaries=[])

    def test_non_increasing_ends_rejected(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            chronological_event_splits(
                _split_events(),
                boundaries=[
                    SplitBoundary("a", datetime(2024, 1, 5)),
                    SplitBoundary("b", datetime(2024, 1, 5)),
                ],
            )

    def test_nonzero_gap_on_last_boundary_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be zero"):
            chronological_event_splits(
                _split_events(),
                boundaries=[SplitBoundary("a", datetime(2024, 1, 5), timedelta(days=1))],
            )

    def test_trailing_events_rejected(self) -> None:
        with pytest.raises(ValueError, match="would be dropped"):
            chronological_event_splits(
                _split_events(),
                boundaries=[SplitBoundary("a", datetime(2024, 1, 5))],
            )

    def test_empty_split_rejected(self) -> None:
        # First cutoff before the earliest event -> empty aux side.
        with pytest.raises(ValueError, match="empty"):
            chronological_event_splits(
                _split_events(),
                boundaries=[
                    SplitBoundary("a", datetime(2023, 12, 1)),
                    SplitBoundary("b", datetime(2024, 1, 13)),
                ],
            )

    def test_empty_last_split_rejected(self) -> None:
        # Events fall at midnight; nothing falls in [2024-01-05 12:00, 2024-01-05 12:00:01).
        with pytest.raises(ValueError, match="is empty"):
            chronological_event_splits(
                _split_events(),
                boundaries=[
                    SplitBoundary("a", datetime(2024, 1, 5, 12, 0, 0)),
                    SplitBoundary("b", datetime(2024, 1, 5, 12, 0, 1)),
                ],
            )

    def test_empty_frame_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            chronological_event_splits(
                _split_events().head(0),
                boundaries=[SplitBoundary("a", datetime(2024, 1, 13))],
            )

    def test_missing_time_column_rejected(self) -> None:
        with pytest.raises(ValueError, match="not present"):
            chronological_event_splits(
                _split_events(),
                time_col="no_such_col",
                boundaries=[SplitBoundary("a", datetime(2024, 1, 13))],
            )


# ---------------------------------------------------------------------------
# Catalog + (t, x, y)
# ---------------------------------------------------------------------------


class TestCatalog:
    def test_csv_columns_exact_and_time_format(self, tmp_path: Path) -> None:
        splits = {"train": _ref_events()}
        written = emit_catalog(splits, output_dir=tmp_path)
        assert {p.name for p in written} == {"train_catalog.csv", "train_txy.npy"}
        with (tmp_path / "train_catalog.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert list(rows[0].keys()) == list(CATALOG_COLUMNS)
        assert len(rows) == 5
        assert rows[0]["time"] == "2024-01-01 00:00:00.000"
        assert rows[1]["time"] == "2024-01-01 00:00:10.000"
        assert rows[0]["mark"] == "2" and rows[0]["mark_label"] == "THEFT"
        assert rows[0]["id"] == "0" and rows[4]["id"] == "4"

    def test_txy_matches_csv_row_for_row(self, tmp_path: Path) -> None:
        events = _ref_events()
        emit_catalog({"train": events}, output_dir=tmp_path)
        txy = np.load(tmp_path / "train_txy.npy")
        assert txy.dtype == np.float64
        assert txy.shape == (5, 3)
        # t = seconds since first event, hand-computed.
        assert txy[:, 0].tolist() == [0.0, 10.0, 30.0, 35.0, 100.0]
        with (tmp_path / "train_catalog.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for i, row in enumerate(rows):
            assert txy[i, 1] == pytest.approx(float(row["x"]))
            assert txy[i, 2] == pytest.approx(float(row["y"]))
            assert txy[i, 1] == pytest.approx(events["x"][i])
            assert txy[i, 2] == pytest.approx(events["y"][i])

    def test_txy_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            events_to_txy(_ref_events().head(0))

    def test_emit_catalog_empty_split_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="empty"):
            emit_catalog({"train": _ref_events().head(0)}, output_dir=tmp_path)


# ---------------------------------------------------------------------------
# tick
# ---------------------------------------------------------------------------


class TestTick:
    def test_payload_shape_and_union(self) -> None:
        payload = build_tick_payload(
            _ref_events(), split_end=datetime(2024, 1, 1, 0, 2, 0), codebook=_CODEBOOK
        )
        assert payload["time_unit"] == "s"
        assert payload["node_labels"] == _CODEBOOK
        assert payload["end_times"] == [120.0]
        realizations = payload["events"]
        assert len(realizations) == 1
        nodes = realizations[0]
        assert len(nodes) == 3
        # Node 0 (ASSAULT): 35s; node 1 (BATTERY): 10s, 100s; node 2 (THEFT): 0s, 30s.
        assert [a.tolist() for a in nodes] == [[35.0], [10.0, 100.0], [0.0, 30.0]]
        for arr in nodes:
            assert arr.dtype == np.float64
            assert (arr == sorted(arr)).all()
        union = sorted(float(v) for arr in nodes for v in arr)
        assert union == [0.0, 10.0, 30.0, 35.0, 100.0]

    def test_emit_round_trip(self, tmp_path: Path) -> None:
        splits = {"train": _ref_events()}
        ends = {"train": datetime(2024, 1, 1, 0, 2, 0)}
        written = emit_tick(splits, output_dir=tmp_path, split_ends=ends, codebook=_CODEBOOK)
        assert [p.name for p in written] == ["train_tick.pkl"]
        with (tmp_path / "train_tick.pkl").open("rb") as f:
            payload = pickle.load(f, encoding="latin-1")
        assert payload["node_labels"] == _CODEBOOK
        assert payload["end_times"] == [120.0]
        assert [a.tolist() for a in payload["events"][0]] == [[35.0], [10.0, 100.0], [0.0, 30.0]]

    def test_split_end_before_first_event_rejected(self) -> None:
        with pytest.raises(ValueError, match="before the first event"):
            build_tick_payload(_ref_events(), split_end=datetime(2023, 12, 31), codebook=_CODEBOOK)

    def test_empty_split_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            build_tick_payload(
                _ref_events().head(0),
                split_end=datetime(2024, 1, 1, 0, 2, 0),
                codebook=_CODEBOOK,
            )

    def test_missing_split_end_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no split end"):
            emit_tick(
                {"train": _ref_events()}, output_dir=tmp_path, split_ends={}, codebook=_CODEBOOK
            )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class TestManifest:
    def test_build_manifest_records_no_fitted_model(self) -> None:
        manifest = build_manifest(
            dataset="test",
            n_events=5,
            n_raw=6,
            n_removed_by_cleaning=1,
            source_crs="EPSG:4326",
            target_crs=_TARGET_CRS,
            time_zone="America/Chicago",
            codebook=_CODEBOOK,
            splits={"train": {"start": "2024-01-01", "end": "2024-01-02", "n_events": 5}},
            embargoes={},
            files=["train.pkl"],
        )
        assert manifest["fitted_model"] == "none"
        assert manifest["model"] == NO_FITTED_MODEL_STATEMENT
        assert manifest["dim_process"] == 3
        assert manifest["library"] == "shikong"
        assert manifest["library_version"] == "0.1.0"
        assert manifest["time_unit"] == "s"

    def test_write_manifest_is_sorted_and_round_trips(self, tmp_path: Path) -> None:
        manifest = build_manifest(
            dataset="test",
            n_events=5,
            n_raw=5,
            n_removed_by_cleaning=0,
            source_crs="EPSG:4326",
            target_crs=_TARGET_CRS,
            time_zone="America/Chicago",
            codebook=_CODEBOOK,
            splits={},
            embargoes={},
            files=["b.pkl", "a.pkl"],
            notes="a note",
        )
        path = write_manifest(tmp_path, manifest)
        assert path.name == "manifest.json"
        text = path.read_text(encoding="utf-8")
        assert text == json.dumps(json.loads(text), indent=2, sort_keys=True) + "\n"
        assert json.loads(text)["files"] == ["a.pkl", "b.pkl"]
        assert json.loads(text)["notes"] == "a note"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_two_emissions_are_byte_identical(self, tmp_path: Path) -> None:
        events = _ref_events()
        splits = {"train": events, "dev": events.head(3)}
        ends = {"train": datetime(2024, 1, 1, 0, 2, 0), "dev": datetime(2024, 1, 1, 0, 1, 0)}
        hashes = []
        for run in ("run1", "run2"):
            out_dir = tmp_path / run
            written = []
            written += emit_gatech(splits, output_dir=out_dir, dim_process=3)
            written += emit_catalog(splits, output_dir=out_dir)
            written += emit_tick(splits, output_dir=out_dir, split_ends=ends, codebook=_CODEBOOK)
            manifest = build_manifest(
                dataset="test",
                n_events=8,
                n_raw=8,
                n_removed_by_cleaning=0,
                source_crs="EPSG:4326",
                target_crs=_TARGET_CRS,
                time_zone="America/Chicago",
                codebook=_CODEBOOK,
                splits={},
                embargoes={},
                files=[p.name for p in written],
            )
            written.append(write_manifest(out_dir, manifest))
            hashes.append(_sha256_tree(out_dir))
        assert hashes[0] == hashes[1]
        assert len(hashes[0]) == 11  # 2 gatech x2 + 2 catalog x2 + 2 tick + manifest
