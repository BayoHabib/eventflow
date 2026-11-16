"""Synthetic ground-truth generator for Phase 4 evaluation (criterion 7).

The generator produces a small unit x time-bin panel with KNOWN planted
structure, plus the event-level rows the panel counts summarise, plus the
planted parameters themselves. Recovery tests run the real library Steps
on this panel and check the planted parameters come back within the
tolerances stated in ``test_synthetic_ground_truth.py`` (and justified in
``docs/design-notes/phase4-synthetic-ground-truth.md``).

Planted structure (``signal=True``, the default):

- Spatial hotspot: a 2x2 block of units in the centre of the grid has a
  baseline rate ``hotspot_multiplier`` times the background rate.
- Temporal seasonality: weekend bins (Saturday/Sunday) have a rate
  ``weekend_multiplier`` times the weekday rate.
- Lag-1 autocorrelation: a latent AR(1) process with coefficient
  ``lag1_phi`` drives each unit's rate; counts are Poisson draws around
  that rate, so the observed count lag-1 correlation is attenuated
  below ``lag1_phi`` by Poisson noise (it is positive, never equal).
- Category mix: every event carries a ``primary_type`` drawn from a
  known categorical distribution (``category_probs``).

With ``signal=False`` every multiplier is 1 and the latent process is
white noise (``lag1_phi = 0``): the panel is pure Poisson noise at a
constant rate, and the steps must recover ~no signal.

Determinism: every draw comes from a single ``numpy.random.Generator``
seeded by ``seed``; the same seed reproduces the same panel bit for bit.

Layout: ``n_grid_rows`` x ``n_grid_cols`` units on a regular grid
(``grid_id = row * n_grid_cols + col + 1``, rows/cols 0-based) crossed
with ``n_bins`` consecutive daily bins starting on ``start`` (a Monday
by default, so the first two bins are weekdays). The panel is small by
design (default 36 units x 56 bins = 2,016 rows) so the recovery tests
run in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl

from eventflow.core.event_frame import EventFrame
from eventflow.core.schema import EventMetadata, EventSchema

# Default planted parameters. These are the "ground truth" the recovery
# tests check against; they are constants of the generator, not fitted
# to any run.
DEFAULT_BASELINE_RATE = 2.0
DEFAULT_HOTSPOT_MULTIPLIER = 3.0
DEFAULT_WEEKEND_MULTIPLIER = 1.6
DEFAULT_LAG1_PHI = 0.6
DEFAULT_LATENT_SCALE = 0.5
DEFAULT_CATEGORY_PROBS: dict[str, float] = {
    "THEFT": 0.40,
    "BATTERY": 0.25,
    "ASSAULT": 0.20,
    "BURGLARY": 0.15,
}
# Centre 2x2 block of the default 6x6 grid, as (row, col) pairs.
DEFAULT_HOTSPOT_CELLS: tuple[tuple[int, int], ...] = ((2, 2), (2, 3), (3, 2), (3, 3))


@dataclass(frozen=True)
class PlantedParams:
    """The planted parameters of one generated panel."""

    seed: int
    signal: bool
    n_grid_rows: int
    n_grid_cols: int
    n_bins: int
    start: date
    baseline_rate: float
    hotspot_units: tuple[int, ...]
    hotspot_multiplier: float
    weekend_multiplier: float
    # Day-of-week rate factors, Monday (0) .. Sunday (6).
    dow_factors: tuple[float, ...]
    lag1_phi: float
    latent_scale: float
    category_probs: dict[str, float]
    top_categories: tuple[str, ...]

    @property
    def n_units(self) -> int:
        """Number of spatial units in the panel."""
        return self.n_grid_rows * self.n_grid_cols


@dataclass
class SyntheticData:
    """One generated synthetic dataset: panel + events + ground truth."""

    panel: pl.DataFrame
    events: pl.DataFrame
    params: PlantedParams
    # (unit, time-bin) pairs dropped by make_sparse_panel, if any.
    dropped_pairs: list[tuple[int, datetime]] = field(default_factory=list)


def _grid_id(row: int, col: int, n_cols: int) -> int:
    return row * n_cols + col + 1


def generate_synthetic_panel(
    *,
    seed: int = 42,
    n_grid_rows: int = 6,
    n_grid_cols: int = 6,
    n_bins: int = 56,
    start: date = date(2024, 1, 1),
    signal: bool = True,
    baseline_rate: float = DEFAULT_BASELINE_RATE,
    hotspot_cells: tuple[tuple[int, int], ...] = DEFAULT_HOTSPOT_CELLS,
    hotspot_multiplier: float = DEFAULT_HOTSPOT_MULTIPLIER,
    weekend_multiplier: float = DEFAULT_WEEKEND_MULTIPLIER,
    lag1_phi: float = DEFAULT_LAG1_PHI,
    latent_scale: float = DEFAULT_LATENT_SCALE,
    category_probs: dict[str, float] | None = None,
) -> SyntheticData:
    """Generate a synthetic unit x time-bin panel with planted structure.

    Args:
        seed: Seed for the single RNG driving every draw.
        n_grid_rows: Grid rows (spatial units = rows x cols).
        n_grid_cols: Grid columns.
        n_bins: Number of consecutive daily time bins.
        start: First bin date; defaults to Monday 2024-01-01.
        signal: If False, plant no signal (constant rate, white-noise
            latent process). The category mix is still drawn from the
            known distribution; the null checks concern the
            temporal/spatial structure, not the categories.
        baseline_rate: Mean events per unit-bin before multipliers.
        hotspot_cells: (row, col) cells with the elevated rate.
        hotspot_multiplier: Hotspot rate multiplier (1.0 when no signal).
        weekend_multiplier: Saturday/Sunday rate multiplier.
        lag1_phi: AR(1) coefficient of the latent rate process.
        latent_scale: Scale of the latent log-rate process.
        category_probs: Category distribution; defaults to
            ``DEFAULT_CATEGORY_PROBS``.

    Returns:
        SyntheticData with the full panel, the event-level rows, and
        the planted parameters.
    """
    probs = dict(category_probs) if category_probs is not None else dict(DEFAULT_CATEGORY_PROBS)
    if abs(sum(probs.values()) - 1.0) > 1e-9:
        raise ValueError(f"category_probs must sum to 1, got {sum(probs.values())}")
    if n_grid_rows < 1 or n_grid_cols < 1:
        raise ValueError("grid dimensions must be >= 1")
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2 (a lag-1 structure needs at least 2 bins)")
    if not 0.0 <= lag1_phi < 1.0:
        raise ValueError(f"lag1_phi must lie in [0, 1), got {lag1_phi}")

    rng = np.random.default_rng(seed)
    phi = lag1_phi if signal else 0.0
    hot_mult = hotspot_multiplier if signal else 1.0
    wknd_mult = weekend_multiplier if signal else 1.0

    dow_factors = tuple(wknd_mult if d >= 5 else 1.0 for d in range(7))
    hotspot_units = tuple(sorted(_grid_id(r, c, n_grid_cols) for r, c in hotspot_cells))
    hotspot_set = set(hotspot_cells)

    # Latent AR(1) per unit. The log-rate scale is normalised so that
    # E[exp(latent_scale * x)] == 1: the planted multipliers keep their
    # meaning as mean-rate multipliers instead of drifting with the
    # latent variance (Var(x) = 1 / (1 - phi^2) for unit innovations).
    var_x = 1.0 / (1.0 - phi * phi) if phi > 0 else 1.0
    lognorm = np.exp(0.5 * latent_scale * latent_scale * var_x) if signal else 1.0

    bins = [datetime.combine(start + timedelta(days=t), datetime.min.time()) for t in range(n_bins)]
    n_units = n_grid_rows * n_grid_cols

    counts = np.zeros((n_units, n_bins), dtype=np.int64)
    latent = np.zeros(n_units)
    for t, bin_dt in enumerate(bins):
        dow_factor = dow_factors[bin_dt.weekday()]
        latent = phi * latent + rng.normal(size=n_units)
        log_rate = latent_scale * latent if signal else np.zeros(n_units)
        for unit in range(n_units):
            row, col = divmod(unit, n_grid_cols)
            mult = hot_mult if (row, col) in hotspot_set else 1.0
            rate = baseline_rate * mult * dow_factor * float(np.exp(log_rate[unit])) / lognorm
            counts[unit, t] = rng.poisson(rate)

    grid_ids = [_grid_id(r, c, n_grid_cols) for r in range(n_grid_rows) for c in range(n_grid_cols)]
    panel = pl.DataFrame(
        {
            "grid_id": [g for _ in bins for g in grid_ids],
            "grid_row": [r for _ in bins for r in range(n_grid_rows) for _ in range(n_grid_cols)],
            "grid_col": [c for _ in bins for _ in range(n_grid_rows) for c in range(n_grid_cols)],
            "time_bin": [b for b in bins for _ in grid_ids],
            "count": counts.T.ravel().tolist(),
            "latitude": [
                41.8781 + r * 0.01
                for _ in bins
                for r in range(n_grid_rows)
                for _ in range(n_grid_cols)
            ],
            "longitude": [
                -87.6298 + c * 0.01
                for _ in bins
                for _ in range(n_grid_rows)
                for c in range(n_grid_cols)
            ],
        }
    )

    # Event-level rows: one row per counted event, category drawn from
    # the planted distribution. The panel counts are the per-(unit, bin)
    # sums of these rows by construction.
    categories = sorted(probs)
    cat_p = np.array([probs[c] for c in categories], dtype=float)
    event_rows: dict[str, list] = {
        "timestamp": [],
        "grid_id": [],
        "time_bin": [],
        "primary_type": [],
        "latitude": [],
        "longitude": [],
    }
    for unit in range(n_units):
        row, col = divmod(unit, n_grid_cols)
        for t, bin_dt in enumerate(bins):
            n_events = int(counts[unit, t])
            if n_events == 0:
                continue
            draws = rng.choice(categories, size=n_events, p=cat_p)
            for cat in draws:
                event_rows["timestamp"].append(bin_dt)
                event_rows["grid_id"].append(grid_ids[unit])
                event_rows["time_bin"].append(bin_dt)
                event_rows["primary_type"].append(str(cat))
                event_rows["latitude"].append(41.8781 + row * 0.01)
                event_rows["longitude"].append(-87.6298 + col * 0.01)
    events = pl.DataFrame(event_rows)

    top_categories = tuple(sorted(probs, key=lambda c: (-probs[c], c)))
    params = PlantedParams(
        seed=seed,
        signal=signal,
        n_grid_rows=n_grid_rows,
        n_grid_cols=n_grid_cols,
        n_bins=n_bins,
        start=start,
        baseline_rate=baseline_rate,
        hotspot_units=hotspot_units,
        hotspot_multiplier=hot_mult,
        weekend_multiplier=wknd_mult,
        dow_factors=dow_factors,
        lag1_phi=phi,
        latent_scale=latent_scale,
        category_probs=probs,
        top_categories=top_categories,
    )
    return SyntheticData(panel=panel, events=events, params=params)


def make_sparse_panel(data: SyntheticData, *, drop_fraction: float, seed: int) -> SyntheticData:
    """Drop a seeded random fraction of (unit, bin) pairs from the panel.

    Used to exercise CompletePanelStep: the dropped pairs must come back
    as explicit zeros. The event rows are left untouched (the sparse
    panel is an *aggregated* view with missing pairs, not missing events).
    """
    if not 0.0 <= drop_fraction < 1.0:
        raise ValueError(f"drop_fraction must lie in [0, 1), got {drop_fraction}")
    rng = np.random.default_rng(seed)
    keys = data.panel.select(["grid_id", "time_bin"]).to_numpy()
    keep = rng.random(len(keys)) >= drop_fraction
    dropped = [(int(g), t) for (g, t), k in zip(keys, keep, strict=True) if not k]
    sparse = data.panel.filter(pl.Series(keep))
    return SyntheticData(
        panel=sparse, events=data.events, params=data.params, dropped_pairs=dropped
    )


def panel_event_frame(panel: pl.DataFrame) -> EventFrame:
    """Wrap a synthetic panel as an EventFrame (timestamp = time_bin)."""
    schema = EventSchema(
        timestamp_col="time_bin",
        lat_col="latitude",
        lon_col="longitude",
        numeric_cols=["count"],
        categorical_cols=[],
    )
    metadata = EventMetadata(dataset_name="synthetic-panel", crs="EPSG:4326", time_bin="1d")
    return EventFrame(panel.lazy(), schema, metadata)


def events_event_frame(events: pl.DataFrame) -> EventFrame:
    """Wrap synthetic event rows as an EventFrame."""
    schema = EventSchema(
        timestamp_col="timestamp",
        lat_col="latitude",
        lon_col="longitude",
        numeric_cols=[],
        categorical_cols=["primary_type"],
    )
    metadata = EventMetadata(dataset_name="synthetic-events", crs="EPSG:4326", time_bin="1d")
    return EventFrame(events.lazy(), schema, metadata)
