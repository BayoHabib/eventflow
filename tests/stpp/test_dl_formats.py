"""Deep-learning STPP formats: wrapper .npz, NSTPP windows, DSTPP/SMASH pickles, DeepSTPP CSV.

All fixtures are synthetic, built in memory with hand-computed
expected values. No real data lives in ``tests/`` (the CI has no CSVs).

Reference fixture (5 events, 3 marks, seconds offsets from T0):

    t = 0s    THEFT     (x, y) from lon/lat below
    t = 10s   BATTERY
    t = 30s   THEFT
    t = 35s   ASSAULT
    t = 100s  BATTERY

Codebook (sorted): ASSAULT=0, BATTERY=1, THEFT=2.

Long fixture (25 events, 1 s apart, single mark THEFT) — for the
sliding-window round-trip: lookback=20 / lookahead=1 gives
25 - 20 - 1 + 1 = 5 windows.

Multi-day fixture (events at day 0, day 1, day 100) — for NSTPP
windowing: anchor 2024-01-01, 30-day windows, 7-day step → 15 windows,
several of them empty.
"""

from __future__ import annotations

import hashlib
import math
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from pyproj import Transformer

from shikong.core.event_frame import EventFrame
from shikong.core.schema import EventMetadata, EventSchema
from shikong.stpp.dl_formats import (
    NSTPP_WINDOW_DAYS,
    emit_deepstpp_csv,
    emit_dstpp_pickles,
    emit_nstpp_npz,
    emit_smash_pickles,
    emit_wrapper_npz,
    window_events_nstpp,
)
from shikong.stpp.emitters import events_to_txy, extract_exact_events

T0 = datetime(2024, 1, 1, 0, 0, 0)
_REF_ROWS = [
    (0, "THEFT", -87.6298, 41.8781),
    (10, "BATTERY", -87.6300, 41.8782),
    (30, "THEFT", -87.6290, 41.8790),
    (35, "ASSAULT", -87.6310, 41.8770),
    (100, "BATTERY", -87.6280, 41.8800),
]
_TARGET_CRS = "EPSG:26971"


def _frame(rows: list[dict], *, mark_col: str | None = "primary_type") -> EventFrame:
    schema = EventSchema(
        timestamp_col="date",
        lat_col="latitude",
        lon_col="longitude",
        categorical_cols=[mark_col] if mark_col else [],
    )
    metadata = EventMetadata(
        dataset_name="test-dl-stpp", crs="EPSG:4326", time_zone="America/Chicago"
    )
    return EventFrame(pl.LazyFrame(rows), schema, metadata)


def _ref_frame() -> EventFrame:
    rows = [
        {"date": T0 + timedelta(seconds=s), "latitude": lat, "longitude": lon, "primary_type": mark}
        for s, mark, lon, lat in _REF_ROWS
    ]
    return _frame(rows)


def _ref_events() -> pl.DataFrame:
    return extract_exact_events(_ref_frame(), mark_col="primary_type", target_crs=_TARGET_CRS)


def _long_events(n: int = 25) -> pl.DataFrame:
    rows = [
        {
            "date": T0 + timedelta(seconds=s),
            "latitude": 41.8781,
            "longitude": -87.6298,
            "primary_type": "THEFT",
        }
        for s in range(n)
    ]
    return extract_exact_events(_frame(rows), mark_col="primary_type", target_crs=_TARGET_CRS)


def _multiday_events() -> pl.DataFrame:
    rows = [
        {
            "date": T0 + timedelta(days=d),
            "latitude": 41.8781,
            "longitude": -87.6298,
            "primary_type": "THEFT",
        }
        for d in (0, 1, 100)
    ]
    return extract_exact_events(_frame(rows), mark_col="primary_type", target_crs=_TARGET_CRS)


def _ref_splits() -> dict[str, pl.DataFrame]:
    events = _ref_events()
    return {
        "aux": events.slice(0, 1),
        "train": events.slice(0, 3),
        "dev": events.slice(3, 2),
        "test": events.slice(4, 1),
    }


def _independent_xy(lon: float, lat: float) -> tuple[float, float]:
    transformer = Transformer.from_crs("EPSG:4326", _TARGET_CRS, always_xy=True)
    x_m, y_m = transformer.transform([lon], [lat])
    return float(x_m[0]) / 1000.0, float(y_m[0]) / 1000.0


def _sha256_tree(directory: Path) -> dict[str, str]:
    return {
        str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def _wrapper_windows(seq: np.ndarray, *, lookback: int = 20, lookahead: int = 1):
    """Faithful re-implementation of SlidingWindowWrapper's windowing.

    ``dataset.py`` (DeepSTPP) / ``sliding_window.py`` (AutoSTPP):
    column 0 -> gaps (``np.diff(prepend=0)``), ``roll`` moves time last
    (``[x, y, dt]``), then sliding windows stride 1.
    """
    s = seq.copy()
    s[:, 0] = np.diff(seq[:, 0], prepend=0)
    s = np.roll(s, -1, axis=1)
    xs, ys = [], []
    for i in range(lookback, len(s) + 1 - lookahead):
        xs.append(s[i - lookback : i])
        ys.append(s[i : i + lookahead])
    return np.stack(xs), np.stack(ys)


def _dstpp_transform(data: list) -> list:
    """Faithful re-implementation of DSTPP ``app.py`` data_loader transform."""
    out = []
    for seq in data:
        transformed = [
            [i[0], i[0] - seq[idx - 1][0] if idx > 0 else i[0]] + i[1:] for idx, i in enumerate(seq)
        ]
        out.append(transformed)
    return out


def _smash_transform(data: list, *, log_normalization: int = 1) -> list:
    """Faithful re-implementation of SMASH ``app.py`` data_loader transform."""
    out = []
    for seq in data:
        transformed = []
        for idx, i in enumerate(seq):
            gap = i[0] - seq[idx - 1][0] if idx > 0 else i[0]
            if log_normalization:
                gap = math.log(max(gap, 1e-4))
            transformed.append([i[0], gap, i[1] + 1] + i[2:])
        out.append(transformed)
    return out


# ---------------------------------------------------------------------------
# Wrapper .npz (DeepSTPP / AutoSTPP shared contract)
# ---------------------------------------------------------------------------


class TestWrapperNpz:
    def test_keys_shapes_dtypes_and_values(self, tmp_path: Path) -> None:
        splits = _ref_splits()
        (path,) = emit_wrapper_npz(splits, output_dir=tmp_path, name="test")
        assert path.name == "test.npz"
        loaded = np.load(path, allow_pickle=True)
        assert sorted(loaded.files) == ["test", "train", "val"]
        for key, split_name in (("train", "train"), ("val", "dev"), ("test", "test")):
            seqs = loaded[key]
            assert seqs.dtype == object and len(seqs) == 1
            arr = seqs[0]
            assert arr.dtype == np.float64
            expected = events_to_txy(splits[split_name])
            assert arr.shape == expected.shape
            np.testing.assert_array_equal(arr, expected)
            # Time starts at 0 (seconds since the split's first event).
            assert arr[0, 0] == 0.0
            assert np.all(np.diff(arr[:, 0]) >= 0)

    def test_aux_excluded_and_no_normalisation(self, tmp_path: Path) -> None:
        splits = _ref_splits()
        (path,) = emit_wrapper_npz(splits, output_dir=tmp_path, name="test")
        loaded = np.load(path, allow_pickle=True)
        assert "aux" not in loaded.files
        # Raw values, not min-max scaled: x/y are projected km, not [0, 1].
        arr = loaded["train"][0]
        x0, y0 = _independent_xy(-87.6298, 41.8781)
        assert arr[0, 1] == pytest.approx(x0)
        assert arr[0, 2] == pytest.approx(y0)
        assert not (arr[:, 1:].min() >= 0.0 and arr[:, 1:].max() <= 1.0 and arr.shape[0] > 2)

    def test_wrapper_windowing_round_trip(self, tmp_path: Path) -> None:
        events = _long_events(25)
        splits = {"train": events, "dev": events.slice(0, 5), "test": events.slice(0, 5)}
        (path,) = emit_wrapper_npz(splits, output_dir=tmp_path, name="long")
        loaded = np.load(path, allow_pickle=True)
        seq = loaded["train"][0]
        assert seq.shape == (25, 3)
        st_x, st_y = _wrapper_windows(seq, lookback=20, lookahead=1)
        assert st_x.shape == (5, 20, 3)
        assert st_y.shape == (5, 1, 3)
        # First gap is t0 = 0.0 (our sequences start at t = 0); rolled columns [x, y, dt].
        assert st_x[0, 0, 2] == 0.0
        assert st_x[0, 1, 2] == 1.0
        assert st_y[0, 0, 2] == 1.0
        # The cumulative (un-rolled, un-diffed) view keeps absolute times.
        assert seq[0, 0] == 0.0 and seq[-1, 0] == 24.0

    def test_missing_or_empty_split_rejected(self, tmp_path: Path) -> None:
        splits = _ref_splits()
        with pytest.raises(ValueError, match="missing required split"):
            emit_wrapper_npz(
                {k: v for k, v in splits.items() if k != "dev"}, output_dir=tmp_path, name="x"
            )
        empty = _ref_events().slice(0, 0)
        with pytest.raises(ValueError, match="empty split"):
            emit_wrapper_npz(
                {"train": empty, "dev": splits["dev"], "test": splits["test"]},
                output_dir=tmp_path,
                name="x",
            )

    def test_deterministic(self, tmp_path: Path) -> None:
        splits = _ref_splits()
        emit_wrapper_npz(splits, output_dir=tmp_path / "a", name="test")
        emit_wrapper_npz(splits, output_dir=tmp_path / "b", name="test")
        assert _sha256_tree(tmp_path / "a") == _sha256_tree(tmp_path / "b")


# ---------------------------------------------------------------------------
# NSTPP windowing
# ---------------------------------------------------------------------------


class TestNstppWindows:
    def test_window_keys_and_values_hand_computed(self) -> None:
        events = _multiday_events()
        windows = window_events_nstpp(events)
        # Anchor 2024-01-01, 7-day steps while start < 2024-04-10 (day 100):
        # 01-01, 01-08, 01-15, 01-22, 01-29, 02-05, 02-12, 02-19, 02-26,
        # 03-04, 03-11, 03-18, 03-25, 04-01, 04-08 -> 15 windows.
        assert len(windows) == 15
        assert sorted(windows)[:3] == ["20240101", "20240108", "20240115"]
        assert sorted(windows)[-1] == "20240408"
        w0 = windows["20240101"]
        assert w0.shape == (2, 3) and w0.dtype == np.float64
        np.testing.assert_array_equal(w0[:, 0], [0.0, 1.0])  # days since window start
        np.testing.assert_array_equal(w0[:, 1], [-87.6298, -87.6298])  # lon, degrees
        np.testing.assert_array_equal(w0[:, 2], [41.8781, 41.8781])  # lat, degrees
        # Empty windows are included (faithful to the preprocessor).
        assert windows["20240108"].shape == (0, 3)
        assert windows["20240115"].shape == (0, 3)
        # Window 20240318 [day 77, 107) holds the day-100 event at t = 23 days.
        w = windows["20240318"]
        assert w.shape == (1, 3)
        assert w[0, 0] == pytest.approx(23.0)
        # Every window's times are within [0, 30) days.
        for arr in windows.values():
            assert arr.shape[1] == 3
            assert np.all((arr[:, 0] >= 0.0) & (arr[:, 0] < NSTPP_WINDOW_DAYS))

    def test_overlap_counts_events_more_than_once(self) -> None:
        events = _multiday_events()
        windows = window_events_nstpp(events)
        total_in_windows = sum(arr.shape[0] for arr in windows.values())
        assert total_in_windows > events.height  # overlapping windows double-count
        # ... but every event appears in at least one window.
        assert total_in_windows >= events.height

    def test_invalid_parameters_rejected(self) -> None:
        events = _multiday_events()
        with pytest.raises(ValueError, match="window_days must be positive"):
            window_events_nstpp(events, window_days=0)
        with pytest.raises(ValueError, match="step_days must be positive"):
            window_events_nstpp(events, step_days=0)
        with pytest.raises(ValueError, match="must not exceed window_days"):
            window_events_nstpp(events, window_days=7, step_days=30)
        with pytest.raises(ValueError, match="empty frame"):
            window_events_nstpp(events.slice(0, 0))

    def test_emit_nstpp_npz(self, tmp_path: Path) -> None:
        events = _multiday_events()
        (path,) = emit_nstpp_npz(events, output_dir=tmp_path, name="test")
        assert path.name == "test_nstpp.npz"
        loaded = np.load(path)  # no pickled objects
        assert len(loaded.files) == 15
        assert loaded["20240101"].dtype == np.float64

    def test_nstpp_deterministic(self, tmp_path: Path) -> None:
        events = _multiday_events()
        emit_nstpp_npz(events, output_dir=tmp_path / "a", name="test")
        emit_nstpp_npz(events, output_dir=tmp_path / "b", name="test")
        assert _sha256_tree(tmp_path / "a") == _sha256_tree(tmp_path / "b")


# ---------------------------------------------------------------------------
# DSTPP pickles
# ---------------------------------------------------------------------------


class TestDstppPickles:
    def test_layout_and_positional_schema(self, tmp_path: Path) -> None:
        splits = _ref_splits()
        written = emit_dstpp_pickles(splits, output_dir=tmp_path, name="test")
        assert [p.name for p in written] == ["data_test.pkl", "data_train.pkl", "data_val.pkl"]
        assert (tmp_path / "dataset" / "test" / "data_train.pkl").exists()
        with (tmp_path / "dataset" / "test" / "data_train.pkl").open("rb") as f:
            data = pickle.load(f)
        assert isinstance(data, list) and len(data) == 1  # one sequence per split
        seq = data[0]
        assert len(seq) == 3  # train split has 3 events
        assert seq[0] == [0.0, -87.6298, 41.8781]  # [t, lng, lat], t seconds
        assert seq[1][0] == 10.0 and seq[2][0] == 30.0
        assert all(isinstance(v, float) for ev in seq for v in ev)
        # dev -> val mapping: val holds the 2 dev events (t = 35s, 100s -> 0s, 65s).
        with (tmp_path / "dataset" / "test" / "data_val.pkl").open("rb") as f:
            val = pickle.load(f)[0]
        assert [ev[0] for ev in val] == [0.0, 65.0]

    def test_loader_transform_and_tie_assert(self, tmp_path: Path) -> None:
        splits = _ref_splits()
        written = emit_dstpp_pickles(splits, output_dir=tmp_path, name="test")
        for p in written:
            with p.open("rb") as f:
                data = pickle.load(f)
            transformed = _dstpp_transform(data)
            gaps = [ev[1] for seq in transformed for ev in seq]
            # First gap of each sequence is its absolute t (0.0 here):
            # the reference loader's `assert Min[1] > 0` FAILS on it.
            assert min(gaps) == 0.0
        # The assert's logic, isolated: it fails iff the first t is 0 or
        # any two consecutive events share a timestamp.
        tied = [[[0.0, -87.0, 41.0], [0.0, -87.0, 41.0]]]
        gaps_tied = [ev[1] for seq in _dstpp_transform(tied) for ev in seq]
        assert min(gaps_tied) == 0.0  # -> `assert Min[1] > 0` would fail
        zero_start = [[[0.0, -87.0, 41.0], [5.0, -87.0, 41.0]]]
        gaps_zero_start = [ev[1] for seq in _dstpp_transform(zero_start) for ev in seq]
        assert min(gaps_zero_start) == 0.0  # first gap is t0 = 0.0 here too
        assert gaps_zero_start[1] == 5.0
        nonzero_start = [[[5.0, -87.0, 41.0], [15.0, -87.0, 41.0]]]
        gaps_nonzero = [ev[1] for seq in _dstpp_transform(nonzero_start) for ev in seq]
        assert min(gaps_nonzero) == 5.0  # only this shape passes the assert

    def test_deterministic(self, tmp_path: Path) -> None:
        splits = _ref_splits()
        emit_dstpp_pickles(splits, output_dir=tmp_path / "a", name="test")
        emit_dstpp_pickles(splits, output_dir=tmp_path / "b", name="test")
        assert _sha256_tree(tmp_path / "a") == _sha256_tree(tmp_path / "b")


# ---------------------------------------------------------------------------
# SMASH pickles
# ---------------------------------------------------------------------------


class TestSmashPickles:
    def test_layout_and_marked_schema(self, tmp_path: Path) -> None:
        splits = _ref_splits()
        written = emit_smash_pickles(splits, output_dir=tmp_path, name="test")
        assert [p.name for p in written] == ["data_test.pkl", "data_train.pkl", "data_val.pkl"]
        with (tmp_path / "dataset" / "test" / "data_train.pkl").open("rb") as f:
            data = pickle.load(f)
        assert isinstance(data, list) and len(data) == 1
        seq = data[0]
        assert len(seq) == 3
        # [t, mark, lng, lat]: THEFT=2, BATTERY=1 (codebook ASSAULT=0, BATTERY=1, THEFT=2).
        assert seq[0] == [0.0, 2, -87.6298, 41.8781]
        assert seq[1] == [10.0, 1, -87.6300, 41.8782]
        assert seq[2] == [30.0, 2, -87.6290, 41.8790]
        assert all(isinstance(ev[1], int) for ev in seq)  # mark is an int, 0-based
        # num_types = max mark + 1 over all splits.
        all_marks = [ev[1] for p in written for seq in pickle.load(p.open("rb")) for ev in seq]
        assert max(all_marks) + 1 == 3

    def test_loader_transform_log_gap_handles_ties(self, tmp_path: Path) -> None:
        splits = _ref_splits()
        written = emit_smash_pickles(splits, output_dir=tmp_path, name="test")
        path = tmp_path / "dataset" / "test" / "data_train.pkl"
        assert path in written
        with path.open("rb") as f:
            data = pickle.load(f)
        transformed = _smash_transform(data, log_normalization=1)
        for seq in transformed:
            for ev in seq:
                assert math.isfinite(ev[1])  # log(max(gap, 1e-4)) never explodes
        # First gap of the test sequence: log(max(0.0, 1e-4)) = log(1e-4).
        with (tmp_path / "dataset" / "test" / "data_test.pkl").open("rb") as f:
            test_data = pickle.load(f)
        first = _smash_transform(test_data, log_normalization=1)[0][0]
        assert first[1] == pytest.approx(math.log(1e-4))
        assert first[2] == 1 + 1  # mark 1 (BATTERY) shifted to 1-based by the loader
        # Tied events: log floor keeps the gap finite (no crash, unlike DSTPP).
        tied = [[[0.0, 1, -87.0, 41.0], [0.0, 1, -87.0, 41.0]]]
        gaps = [ev[1] for seq in _smash_transform(tied, log_normalization=1) for ev in seq]
        assert gaps == pytest.approx([math.log(1e-4), math.log(1e-4)])

    def test_deterministic(self, tmp_path: Path) -> None:
        splits = _ref_splits()
        emit_smash_pickles(splits, output_dir=tmp_path / "a", name="test")
        emit_smash_pickles(splits, output_dir=tmp_path / "b", name="test")
        assert _sha256_tree(tmp_path / "a") == _sha256_tree(tmp_path / "b")


# ---------------------------------------------------------------------------
# DeepSTPP CSV ([x, y, t])
# ---------------------------------------------------------------------------


class TestDeepstppCsv:
    def test_columns_order_format_and_values(self, tmp_path: Path) -> None:
        splits = _ref_splits()
        written = emit_deepstpp_csv(splits, output_dir=tmp_path)
        assert [p.name for p in written] == [
            "aux_deepstpp.csv",
            "dev_deepstpp.csv",
            "test_deepstpp.csv",
            "train_deepstpp.csv",
        ]
        text = (tmp_path / "train_deepstpp.csv").read_text(encoding="utf-8")
        rows = [line.split(",") for line in text.strip().splitlines()]
        assert len(rows) == 3
        x0, y0 = _independent_xy(-87.6298, 41.8781)
        assert rows[0] == [f"{x0:.6f}", f"{y0:.6f}", "0.000000"]  # [x, y, t], %f
        assert rows[1][2] == "10.000000" and rows[2][2] == "30.000000"
        # Reloadable by np.loadtxt, the synthetic.py::load shape.
        arr = np.loadtxt(tmp_path / "train_deepstpp.csv", delimiter=",")
        assert arr.shape == (3, 3)
        np.testing.assert_array_almost_equal(arr[:, 2], [0.0, 10.0, 30.0])

    def test_deterministic(self, tmp_path: Path) -> None:
        splits = _ref_splits()
        emit_deepstpp_csv(splits, output_dir=tmp_path / "a")
        emit_deepstpp_csv(splits, output_dir=tmp_path / "b")
        assert _sha256_tree(tmp_path / "a") == _sha256_tree(tmp_path / "b")
