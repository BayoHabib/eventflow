"""Regression tests for the downloader-to-Chicago-feature pipeline."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
import yaml
from pyproj import Transformer
from typer.testing import CliRunner

from shikong.cli.main import app
from shikong.core.event_frame import EventFrame
from shikong.core.schema import DatasetConfig, EventMetadata, EventSchema, RecipeConfig
from shikong.core.spatial import assign_to_grid, assign_to_zones, create_grid, transform_crs
from shikong.core.steps.spatial import AssignToGridStep, AssignToZonesStep, TransformCRSStep
from shikong.core.temporal import extract_temporal_components
from shikong.datasets.chicago_crime.mapping import clean_chicago_data, load_chicago_crime
from shikong.recipes.registry import get_recipe, list_datasets


@pytest.fixture
def raw_events() -> pl.DataFrame:
    """Use Socrata types and points whose expected metric cells are known."""
    longitude, latitude = Transformer.from_crs("EPSG:26971", "EPSG:4326", always_xy=True).transform(
        [350050.0, 350050.0, 351050.0, 350050.0], [570050.0] * 4
    )
    return pl.DataFrame(
        {
            "date": [
                "2024-01-05T05:59:59.000",
                "2024-01-05T06:00:00.000",
                "2024-01-05T06:30:00.000",
                "2024-01-05T06:10:00.000",
            ],
            "latitude": latitude,
            "longitude": longitude,
            "arrest": [True, False, True, False],
            "domestic": [False, True, False, True],
        }
    ).with_columns(pl.col("latitude", "longitude").cast(pl.String))


@pytest.fixture
def recipe_config() -> RecipeConfig:
    return RecipeConfig(
        dataset="chicago_crime",
        recipe="chicago_crime_v1",
        grid={"size_m": 500},
        temporal={"time_bin": "6h", "components": ["hour_of_day", "day_of_week", "is_weekend"]},
    )


@pytest.mark.parametrize("suffix", ["csv", "parquet"])
@pytest.mark.parametrize("target_crs", ["EPSG:4326", "EPSG:26971"])
def test_realistic_chunks_to_six_hour_counts(
    tmp_path: Path,
    raw_events: pl.DataFrame,
    recipe_config: RecipeConfig,
    suffix: str,
    target_crs: str,
) -> None:
    root = tmp_path / "daily" / "2024-01-05"
    root.mkdir(parents=True)
    for index, data in enumerate([raw_events.head(2), raw_events.tail(2)]):
        path = root / f"chunk_{index}.{suffix}"
        if suffix == "csv":
            data.write_csv(path)
        else:
            data.write_parquet(path)
    config = DatasetConfig(dataset_name="chicago_crime", raw_root=str(tmp_path), crs=target_crs)
    events = load_chicago_crime(str(tmp_path), config=config)
    result = get_recipe("chicago_crime", "chicago_crime_v1", recipe_config).run(events)
    data = result.collect()

    assert data["count"].to_list() == [1, 2, 1]
    assert data["grid_id"].to_list() == [6774440, 6774440, 6789174]
    assert data["time_bin"].to_list() == [
        datetime(2024, 1, 5),
        datetime(2024, 1, 5, 6),
        datetime(2024, 1, 5, 6),
    ]
    assert data["is_weekend"].to_list() == [False] * 3
    assert data[result.schema.lon_col].to_list() == [350250.0, 350250.0, 351250.0]
    assert data[result.schema.lat_col].to_list() == [570250.0] * 3
    assert data[result.schema.timestamp_col].equals(data["time_bin"])
    assert set(result.schema.numeric_cols) <= set(data.columns)
    assert result.metadata.crs == "EPSG:26971"
    assert result.metadata.time_bin == "6h"


def test_cell_ids_are_stable_across_batches(
    tmp_path: Path,
    raw_events: pl.DataFrame,
    recipe_config: RecipeConfig,
) -> None:
    path = tmp_path / "events.parquet"
    recipe = get_recipe("chicago_crime", "chicago_crime_v1", recipe_config)
    raw_events.write_parquet(path)
    combined = recipe.run(load_chicago_crime(str(path))).collect()
    raw_events.head(1).write_parquet(path)
    subset = recipe.run(load_chicago_crime(str(path))).collect()
    assert subset["grid_id"][0] == combined["grid_id"][0] == 6774440


def test_empty_input_produces_valid_empty_features(
    tmp_path: Path, raw_events: pl.DataFrame
) -> None:
    path = tmp_path / "empty.parquet"
    raw_events.head(0).write_parquet(path)
    result = get_recipe("chicago_crime", "chicago_crime_v1").run(load_chicago_crime(str(path)))
    data = result.collect()
    assert data.height == 0
    assert {"grid_id", "time_bin", "count", result.schema.timestamp_col} <= set(data.columns)


def test_loader_rejects_missing_data(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No CSV or Parquet"):
        load_chicago_crime(str(tmp_path))


def test_loader_rejects_mixed_formats(tmp_path: Path, raw_events: pl.DataFrame) -> None:
    raw_events.write_parquet(tmp_path / "chunk.parquet")
    raw_events.write_csv(tmp_path / "chunk.csv")
    with pytest.raises(ValueError, match="only CSV or only Parquet"):
        load_chicago_crime(str(tmp_path))


def test_metric_grid_uses_projected_columns(raw_events: pl.DataFrame) -> None:
    frame = EventFrame(
        clean_chicago_data(raw_events.lazy()),
        EventSchema(timestamp_col="date", lat_col="latitude", lon_col="longitude"),
        EventMetadata(dataset_name="metric-grid"),
    )
    projected = transform_crs(frame, "EPSG:26971")
    grid = create_grid((350000, 570000, 352000, 571000), 500, crs="EPSG:26971")
    assert assign_to_grid(projected, grid).collect()["grid_id"].to_list() == [0, 0, 14, 0]


@pytest.mark.parametrize(
    "date",
    [
        "2024-01-05T06:00:00.000",
        "2024-01-05 06:00:00",
        "01/05/2024 06:00:00 AM",
        datetime(2024, 1, 5, 6),
        datetime(2024, 1, 5).date(),
    ],
)
def test_cleaning_accepts_api_legacy_and_native_dates(
    raw_events: pl.DataFrame, date: object
) -> None:
    data = raw_events.head(1).with_columns(pl.lit(date).alias("date"))
    cleaned = clean_chicago_data(data.lazy()).collect()
    assert cleaned.height == 1
    assert cleaned["date"].dtype == pl.Datetime
    assert cleaned["arrest"][0] is True
    assert cleaned["domestic"][0] is False


def test_cleaning_normalizes_boolean_strings(raw_events: pl.DataFrame) -> None:
    data = raw_events.with_columns(pl.Series("arrest", ["True", "FALSE", "1", "0"]))
    assert clean_chicago_data(data.lazy()).collect()["arrest"].to_list() == [
        True,
        False,
        True,
        False,
    ]


def test_cleaning_filters_invalid_records(raw_events: pl.DataFrame) -> None:
    data = raw_events.with_columns(pl.Series("latitude", ["41.8", "invalid", "91", None]))
    assert clean_chicago_data(data.lazy()).collect().height == 1


@pytest.mark.parametrize("suffix", ["csv", "parquet"])
@pytest.mark.parametrize("use_dataset_config", [False, True])
def test_cli_runs_and_writes_output(
    tmp_path: Path,
    raw_events: pl.DataFrame,
    recipe_config: RecipeConfig,
    suffix: str,
    use_dataset_config: bool,
) -> None:
    source = tmp_path / "raw.parquet"
    raw_events.write_parquet(source)
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(yaml.safe_dump(recipe_config.model_dump()), encoding="utf-8")
    output = tmp_path / "processed" / f"features.{suffix}"
    argv = [
        "run",
        "--dataset",
        "chicago_crime",
        "--recipe",
        "chicago_crime_v1",
        "--config",
        str(config_path),
        "--output",
        str(output),
    ]
    if use_dataset_config:
        dataset_path = tmp_path / "dataset.yaml"
        dataset_path.write_text(
            yaml.safe_dump(
                {"dataset_name": "chicago_crime", "raw_root": str(source), "crs": "EPSG:26971"}
            ),
            encoding="utf-8",
        )
        argv.extend(["--dataset-config", str(dataset_path)])
    else:
        argv.extend(["--input", str(source)])

    result = CliRunner().invoke(app, argv)

    assert result.exit_code == 0, result.output
    data = pl.read_parquet(output) if suffix == "parquet" else pl.read_csv(output)
    assert data["count"].to_list() == [1, 2, 1]
    assert data["count"].sum() == 4


def test_cli_discovers_chicago_recipe() -> None:
    assert "chicago_crime" in list_datasets()
    result = CliRunner().invoke(app, ["list-recipes", "--dataset", "chicago_crime"])
    assert result.exit_code == 0
    assert "chicago_crime_v1" in result.output


def test_cli_validate_accepts_shipped_chicago_config() -> None:
    """A valid config passes validation (positive control for validate)."""
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "configs" / "recipes" / "chicago_crime_v1.yaml"
    result = CliRunner().invoke(app, ["validate", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    # Phase 2: the shipped YAML enables the full feature set (21 steps),
    # up from the 17-step phase-1 pipeline.
    assert "pipeline builds: 21 steps" in result.output


def test_phase2_time_steps_use_populated_timestamp_column() -> None:
    """Phase 2 replay bug: time-dependent steps must use "date", not "time_bin".

    On the completed panel the pre-completion "time_bin" column is null
    on every generated (explicit-zero) row, while the schema timestamp
    column "date" holds the bin datetime on all rows. Building the
    near-repeat and STKDE steps on "time_bin" silently computed their
    windows over the sparse observed-bins-only series (wrong values,
    no error) — e.g. count_nr_r1_7d = 60 instead of 50 on the busiest
    cell at 2024-02-17 12:00. The recipe must wire the populated
    timestamp column into both steps (and the steps themselves now
    reject a null time column, see their validation tests).
    """
    from shikong.core.steps.density import SpaceTimeKernelDensityStep
    from shikong.core.steps.near_repeat import NearRepeatStep

    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "configs" / "recipes" / "chicago_crime_v1.yaml"
    config = RecipeConfig(**yaml.safe_load(config_path.read_text(encoding="utf-8")))
    steps = get_recipe("chicago_crime", "chicago_crime_v1", config).get_pipeline().steps
    by_type = {type(step): step for step in steps}
    assert by_type[NearRepeatStep].time_col == "date"
    assert by_type[SpaceTimeKernelDensityStep].time_col == "date"


def test_cli_validate_rejects_unknown_configured_step(tmp_path: Path) -> None:
    """Issue #29: validate must check the pipeline run() will execute.

    A configured-but-unknown step in features.steps must fail validation
    instead of silently validating the recipe's built-in pipeline.
    """
    path = tmp_path / "unknown-step.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "dataset": "chicago_crime",
                "recipe": "chicago_crime_v1",
                "features": {"steps": ["not_a_real_step"]},
            }
        ),
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["validate", "--config", str(path)])
    assert result.exit_code == 1, result.output
    assert "not_a_real_step" in result.output


def test_cli_validate_rejects_unknown_temporal_component(tmp_path: Path) -> None:
    """Issue #30: a typo in temporal.components must fail validation."""
    path = tmp_path / "unknown-component.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "dataset": "chicago_crime",
                "recipe": "chicago_crime_v1",
                "temporal": {"components": ["hour_of_dya"]},
            }
        ),
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["validate", "--config", str(path)])
    assert result.exit_code == 1, result.output
    assert "hour_of_dya" in result.output


def test_cli_missing_input_fails_without_output(
    tmp_path: Path, recipe_config: RecipeConfig
) -> None:
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe_config.model_dump()), encoding="utf-8")
    output = tmp_path / "features.parquet"
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--dataset",
            "chicago_crime",
            "--recipe",
            "chicago_crime_v1",
            "--config",
            str(path),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 1
    assert "--input or --dataset-config" in result.output
    assert not output.exists()


def geographic_frame() -> EventFrame:
    return EventFrame(
        pl.DataFrame(
            {
                "date": [datetime(2024, 1, 5)] * 2,
                "latitude": [41.8, 41.9],
                "longitude": [-87.7, -87.6],
            }
        ).lazy(),
        EventSchema(timestamp_col="date", lat_col="latitude", lon_col="longitude"),
        EventMetadata(dataset_name="spatial-regression"),
    )


@pytest.mark.parametrize("use_step", [False, True])
def test_projection_updates_active_columns(use_step: bool) -> None:
    source = geographic_frame()
    projected = (
        TransformCRSStep("EPSG:26971").run(source)
        if use_step
        else transform_crs(source, "EPSG:26971")
    )
    assert projected.schema.lon_col == "longitude_proj"
    assert projected.schema.lat_col == "latitude_proj"
    ids = AssignToGridStep(500, origin=(0, 0)).run(projected).collect()["grid_id"]
    assert ids.n_unique() == 2
    assert source.schema.lon_col == "longitude"
    assert source.metadata.crs == "EPSG:4326"


@pytest.mark.parametrize("crs", ["EPSG:4326", "EPSG:3435"])
def test_grid_creation_rejects_non_metric_crs(crs: str) -> None:
    with pytest.raises(ValueError, match="projected CRS in meters"):
        create_grid((-87.94, 41.64, -87.52, 42.02), 500, crs=crs)


def test_grid_assignment_rejects_degrees() -> None:
    with pytest.raises(ValueError, match="projected CRS in meters"):
        AssignToGridStep(500).run(geographic_frame())


@pytest.mark.parametrize("use_step", [False, True])
def test_projection_can_be_collected_on_other_threads(use_step: bool) -> None:
    frame = geographic_frame()
    projected = (
        TransformCRSStep("EPSG:26971").run(frame)
        if use_step
        else transform_crs(frame, "EPSG:26971")
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: projected.collect(), range(4)))
    assert all(result.equals(results[0]) for result in results)
    assert results[0]["longitude_proj"][0] == pytest.approx(352634.941673, abs=0.01)


@pytest.mark.parametrize("use_step", [False, True])
def test_zone_assignment_finds_containing_polygon(use_step: bool) -> None:
    zones = pl.DataFrame(
        {
            "zone_id": [7],
            "geometry": [
                "POLYGON ((-87.8 41.7, -87.65 41.7, -87.65 41.85, -87.8 41.85, -87.8 41.7))"
            ],
        }
    )
    frame = geographic_frame()
    result = AssignToZonesStep(zones).run(frame) if use_step else assign_to_zones(frame, zones)
    assert result.collect()["zone_id"].to_list() == [7, None]


def test_legacy_weekend_flags_are_iso_correct() -> None:
    frame = geographic_frame().with_lazy_frame(
        pl.DataFrame(
            {
                "date": [datetime(2024, 1, 5), datetime(2024, 1, 6), datetime(2024, 1, 7)],
                "latitude": [41.8] * 3,
                "longitude": [-87.7] * 3,
            }
        ).lazy()
    )
    result = extract_temporal_components(frame, ["day_of_week", "is_weekend"]).collect()
    assert result["day_of_week"].to_list() == [5, 6, 7]
    assert result["is_weekend"].to_list() == [False, True, True]


def test_legacy_extract_rejects_unknown_component() -> None:
    """An unknown component must fail loudly, not be silently omitted (issue #33)."""
    frame = geographic_frame().with_lazy_frame(
        pl.DataFrame(
            {
                "date": [datetime(2024, 1, 5), datetime(2024, 1, 6), datetime(2024, 1, 7)],
                "latitude": [41.8] * 3,
                "longitude": [-87.7] * 3,
            }
        ).lazy()
    )
    with pytest.raises(ValueError, match="hour_of_dya"):
        extract_temporal_components(frame, ["hour_of_dya"])


def test_legacy_extract_rejects_unimplemented_is_holiday() -> None:
    """is_holiday was documented but never implemented; requesting it must fail (issue #33)."""
    frame = geographic_frame().with_lazy_frame(
        pl.DataFrame(
            {
                "date": [datetime(2024, 1, 5), datetime(2024, 1, 6), datetime(2024, 1, 7)],
                "latitude": [41.8] * 3,
                "longitude": [-87.7] * 3,
            }
        ).lazy()
    )
    with pytest.raises(ValueError, match="is_holiday"):
        extract_temporal_components(frame, ["is_holiday"])


def test_legacy_extract_produces_all_supported_components() -> None:
    """Positive control: every component the legacy function implements is produced."""
    frame = geographic_frame().with_lazy_frame(
        pl.DataFrame(
            {
                "date": [datetime(2024, 1, 5), datetime(2024, 1, 6), datetime(2024, 1, 7)],
                "latitude": [41.8] * 3,
                "longitude": [-87.7] * 3,
            }
        ).lazy()
    )
    components = [
        "hour_of_day",
        "day_of_week",
        "day_of_month",
        "day_of_year",
        "month",
        "year",
        "is_weekend",
    ]
    result = extract_temporal_components(frame, components).collect()
    for component in components:
        assert component in result.columns


def test_recipe_preserves_support_for_queen_lag():
    from shikong.core.steps.spatial import SpatialLagStep

    x, y = Transformer.from_crs("EPSG:26971", "EPSG:4326", always_xy=True).transform(
        [350050.0, 350050.0, 350550.0], [570050.0] * 3
    )
    df = pl.DataFrame({"longitude": x, "latitude": y, "date": [datetime(2024, 1, 5, 1)] * 3})
    events = EventFrame(
        df.lazy(),
        EventSchema(timestamp_col="date", lon_col="longitude", lat_col="latitude"),
        EventMetadata(dataset_name="recipe-support"),
    )
    recipe = get_recipe(
        "chicago_crime",
        "chicago_crime_v1",
        RecipeConfig(
            dataset="chicago_crime",
            recipe="chicago_crime_v1",
            grid={"size_m": 500},
            temporal={"time_bin": "1h"},
        ),
    )
    aggregated = recipe.run(events)
    out = (
        SpatialLagStep(["count"], group_cols=["time_bin"])
        .run(aggregated)
        .collect()
        .sort("grid_col")
    )
    assert out["count"].to_list() == [2, 1]
    assert out["count_spatial_lag"].to_list() == [1.0, 2.0]
    assert {"grid_row", "grid_col"} <= set(out.columns)
    assert aggregated.metadata.custom["grid_spec"]["id_scheme"] == "signed_cantor_v1"
