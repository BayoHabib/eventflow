"""RasterAdapter: Convert EventFrames to 2D/3D arrays for CNNs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from eventflow.core.adapters.base import (
    AdapterMetadata,
    BaseModalityAdapter,
    SerializationFormat,
)
from eventflow.core.adapters.configs import RasterAdapterConfig
from eventflow.core.utils import get_logger

if TYPE_CHECKING:
    from eventflow.core.event_frame import EventFrame

logger = get_logger(__name__)


@dataclass
class RasterOutput:
    """Output from RasterAdapter conversion.

    Attributes:
        raster: 4D array of shape (n_timesteps, n_channels, height, width)
               or (n_timesteps, height, width, n_channels) if channel_last
        timestamps: Timestamps for each time step
        channel_names: Names of each channel (feature)
        grid_shape: (height, width) of the spatial grid
        channel_first: Whether channels are first dimension
        dtypes: Data type information
    """

    raster: np.ndarray
    timestamps: list[Any] = field(default_factory=list)
    channel_names: list[str] = field(default_factory=list)
    grid_shape: tuple[int, int] = (0, 0)
    channel_first: bool = True
    dtypes: dict[str, str] = field(default_factory=dict)

    @property
    def n_timesteps(self) -> int:
        """Number of time steps."""
        return int(self.raster.shape[0])

    @property
    def n_channels(self) -> int:
        """Number of channels."""
        if self.channel_first:
            return int(self.raster.shape[1])
        return int(self.raster.shape[3])

    @property
    def height(self) -> int:
        """Height of the raster."""
        if self.channel_first:
            return int(self.raster.shape[2])
        return int(self.raster.shape[1])

    @property
    def width(self) -> int:
        """Width of the raster."""
        if self.channel_first:
            return int(self.raster.shape[3])
        return int(self.raster.shape[2])

    def to_torch(self) -> dict[str, Any]:
        """Convert to PyTorch tensors."""
        try:
            import torch

            return {
                "raster": torch.from_numpy(self.raster),
                "n_timesteps": self.n_timesteps,
                "n_channels": self.n_channels,
                "height": self.height,
                "width": self.width,
            }
        except ImportError as e:
            raise ImportError("PyTorch required for to_torch()") from e

    def to_tensorflow(self) -> np.ndarray:
        """Convert to TensorFlow format (channel last)."""
        if self.channel_first:
            # (T, C, H, W) -> (T, H, W, C)
            return np.transpose(self.raster, (0, 2, 3, 1))
        return self.raster


class RasterAdapter(BaseModalityAdapter[RasterOutput]):
    """Convert EventFrames to 2D/3D raster arrays for CNN models.

    Creates multi-channel grids per timestep, suitable for image-based
    models like CNNs or ConvLSTMs.
    """

    def __init__(
        self,
        config: RasterAdapterConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize RasterAdapter.

        Args:
            config: Configuration object or None for defaults
            **kwargs: Configuration parameters (used if config is None)
        """
        if config is None:
            config = RasterAdapterConfig(**kwargs)
        self.config = config

    @property
    def modality(self) -> str:
        return "raster"

    def convert(self, event_frame: EventFrame) -> RasterOutput:
        """Convert EventFrame to raster format.

        Args:
            event_frame: The EventFrame to convert

        Returns:
            RasterOutput with 4D raster array
        """
        logger.info("Converting EventFrame to raster format")

        # Get configuration
        grid_col = self.config.grid_col or "grid_id"
        # Check if this is an EventFrame (has EventSchema) vs plain DataFrame/LazyFrame
        has_eventframe_schema = False
        event_schema = None
        is_polars_lazy = isinstance(event_frame, pl.LazyFrame)
        is_polars_dataframe = isinstance(event_frame, pl.DataFrame)
        
        if not is_polars_lazy and not is_polars_dataframe:
            if hasattr(event_frame, "schema"):
                event_schema = event_frame.schema
                has_eventframe_schema = hasattr(event_schema, "timestamp_col")
        
        timestamp_col = self.config.timestamp_col
        if timestamp_col is None:
            if has_eventframe_schema and event_schema and event_schema.timestamp_col:
                timestamp_col = event_schema.timestamp_col
            else:
                timestamp_col = "timestamp"

        # Collect data
        if hasattr(event_frame, "collect"):
            df = event_frame.collect()
        else:
            df = event_frame  # type: ignore[assignment]

        if grid_col not in df.columns:
            raise ValueError(f"Grid column '{grid_col}' not found in EventFrame")

        # Determine feature columns
        if self.config.feature_cols is not None:
            feature_cols = list(self.config.feature_cols)
        else:
            exclude = {grid_col, timestamp_col}
            if has_eventframe_schema and event_schema:
                if event_schema.lat_col:
                    exclude.add(event_schema.lat_col)
                if event_schema.lon_col:
                    exclude.add(event_schema.lon_col)
                if event_schema.geometry_col:
                    exclude.add(event_schema.geometry_col)

            feature_cols = [
                col
                for col in df.columns
                if col not in exclude
                and df[col].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
            ]

        # Determine grid shape
        if self.config.grid_shape is not None:
            grid_height, grid_width = self.config.grid_shape
        else:
            # Try to infer from grid_id format (e.g., "row_col" or numeric)
            unique_grids = df[grid_col].unique().to_list()
            if unique_grids and isinstance(unique_grids[0], str):
                # Try parsing "row_col" format
                try:
                    coords = [tuple(map(int, g.split("_"))) for g in unique_grids]
                    max_row = max(c[0] for c in coords) + 1
                    max_col = max(c[1] for c in coords) + 1
                    grid_height, grid_width = max_row, max_col
                except (ValueError, IndexError):
                    # Fallback to square grid
                    n_cells = len(unique_grids)
                    side = int(np.ceil(np.sqrt(n_cells)))
                    grid_height, grid_width = side, side
            else:
                n_cells = len(unique_grids)
                side = int(np.ceil(np.sqrt(n_cells)))
                grid_height, grid_width = side, side

        # Get unique timestamps
        timestamps = df[timestamp_col].unique().sort().to_list()
        if self.config.time_steps is not None:
            timestamps = timestamps[: self.config.time_steps]

        n_timesteps = len(timestamps)
        n_channels = len(feature_cols)

        # Determine dtype
        np_dtype = np.float32 if self.config.dtype == "float32" else np.float64

        # Create raster array
        if self.config.channel_first:
            raster = np.full(
                (n_timesteps, n_channels, grid_height, grid_width),
                self.config.fill_value,
                dtype=np_dtype,
            )
        else:
            raster = np.full(
                (n_timesteps, grid_height, grid_width, n_channels),
                self.config.fill_value,
                dtype=np_dtype,
            )

        # Build grid index mapping
        unique_grids = df[grid_col].unique().to_list()
        grid_to_idx = {}
        for grid_id in unique_grids:
            if isinstance(grid_id, str) and "_" in grid_id:
                try:
                    row, col = map(int, grid_id.split("_"))
                    grid_to_idx[grid_id] = (row, col)
                except ValueError:
                    pass

        # If no mapping, create linear mapping
        if not grid_to_idx:
            for i, grid_id in enumerate(sorted(unique_grids)):
                row = i // grid_width
                col = i % grid_width
                grid_to_idx[grid_id] = (row, col)

        # Fill raster
        timestamp_to_idx = {ts: i for i, ts in enumerate(timestamps)}

        for data_row in df.iter_rows(named=True):
            ts = data_row[timestamp_col]
            grid_id = data_row[grid_col]

            if ts not in timestamp_to_idx:
                continue
            if grid_id not in grid_to_idx:
                continue

            t_idx = timestamp_to_idx[ts]
            r_idx, c_idx = grid_to_idx[grid_id]

            if r_idx >= grid_height or c_idx >= grid_width:
                continue

            for ch_idx, feat in enumerate(feature_cols):
                val = data_row.get(feat)
                if val is not None:
                    if self.config.channel_first:
                        raster[t_idx, ch_idx, r_idx, c_idx] = val
                    else:
                        raster[t_idx, r_idx, c_idx, ch_idx] = val

        # Normalize if requested
        if self.config.normalize:
            for ch_idx in range(n_channels):
                if self.config.channel_first:
                    channel_data = raster[:, ch_idx, :, :]
                else:
                    channel_data = raster[:, :, :, ch_idx]

                mean = np.nanmean(channel_data)
                std = np.nanstd(channel_data)
                if std > 0:
                    if self.config.channel_first:
                        raster[:, ch_idx, :, :] = (channel_data - mean) / std
                    else:
                        raster[:, :, :, ch_idx] = (channel_data - mean) / std

        logger.info(
            f"Created raster with shape {raster.shape}: "
            f"{n_timesteps} timesteps, {n_channels} channels, "
            f"{grid_height}x{grid_width} grid"
        )

        return RasterOutput(
            raster=raster,
            timestamps=timestamps,
            channel_names=feature_cols,
            grid_shape=(grid_height, grid_width),
            channel_first=self.config.channel_first,
            dtypes={"raster": str(np_dtype)},
        )

    def get_metadata(self, output: RasterOutput) -> AdapterMetadata:
        """Extract metadata from raster output."""
        return AdapterMetadata(
            modality=self.modality,
            feature_names=output.channel_names,
            shapes={
                "raster": output.raster.shape,
                "grid": output.grid_shape,
            },
            dtypes=output.dtypes,
            spatial_info={
                "grid_height": output.height,
                "grid_width": output.width,
            },
            temporal_info={
                "n_timesteps": output.n_timesteps,
            },
            extra={
                "channel_first": output.channel_first,
                "n_channels": output.n_channels,
            },
        )

    def serialize(
        self,
        output: RasterOutput,
        path: Path | str,
        fmt: SerializationFormat | str,
    ) -> None:
        """Serialize raster output to disk."""
        path = Path(path)
        fmt = SerializationFormat(fmt) if isinstance(fmt, str) else fmt

        path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == SerializationFormat.NUMPY:
            np.savez(
                path,
                raster=output.raster,
                timestamps=np.array(output.timestamps, dtype=object),
                channel_names=np.array(output.channel_names),
                grid_shape=np.array(output.grid_shape),
                channel_first=np.array(output.channel_first),
            )

        elif fmt == SerializationFormat.PYTORCH:
            try:
                import torch

                data = output.to_torch()
                data["timestamps"] = output.timestamps
                data["channel_names"] = output.channel_names
                data["grid_shape"] = output.grid_shape
                data["channel_first"] = output.channel_first
                torch.save(data, path)
            except ImportError as e:
                raise ImportError("PyTorch required for pytorch format") from e

        elif fmt == SerializationFormat.ARROW:
            # Flatten for storage
            n_t, n_c, h, w = (
                (
                    output.raster.shape[0],
                    output.raster.shape[1],
                    output.raster.shape[2],
                    output.raster.shape[3],
                )
                if output.channel_first
                else (
                    output.raster.shape[0],
                    output.raster.shape[3],
                    output.raster.shape[1],
                    output.raster.shape[2],
                )
            )

            flat_data: dict[str, Any] = {
                "timestep": np.repeat(np.arange(n_t), h * w),
                "row": np.tile(np.repeat(np.arange(h), w), n_t),
                "col": np.tile(np.tile(np.arange(w), h), n_t),
            }

            for ch_idx, name in enumerate(output.channel_names):
                if output.channel_first:
                    flat_data[name] = output.raster[:, ch_idx, :, :].flatten()
                else:
                    flat_data[name] = output.raster[:, :, :, ch_idx].flatten()

            df = pl.DataFrame(flat_data)
            df.write_ipc(path)

            meta_path = path.with_suffix(".meta.json")
            meta = self.get_metadata(output)
            meta_path.write_text(json.dumps(meta.to_dict(), indent=2))

        else:
            raise ValueError(f"Unsupported format for RasterAdapter: {fmt}")

        logger.info(f"Serialized raster to {path} as {fmt}")

    def deserialize(
        self,
        path: Path | str,
        fmt: SerializationFormat | str,
    ) -> RasterOutput:
        """Deserialize raster output from disk."""
        path = Path(path)
        fmt = SerializationFormat(fmt) if isinstance(fmt, str) else fmt

        if fmt == SerializationFormat.NUMPY:
            loaded = np.load(path, allow_pickle=True)
            return RasterOutput(
                raster=loaded["raster"],
                timestamps=list(loaded["timestamps"]),
                channel_names=list(loaded["channel_names"]),
                grid_shape=tuple(loaded["grid_shape"]),
                channel_first=bool(loaded["channel_first"]),
            )

        elif fmt == SerializationFormat.PYTORCH:
            try:
                import torch

                data = torch.load(path, weights_only=False)
                return RasterOutput(
                    raster=data["raster"].numpy(),
                    timestamps=data["timestamps"],
                    channel_names=data["channel_names"],
                    grid_shape=data["grid_shape"],
                    channel_first=data["channel_first"],
                )
            except ImportError as e:
                raise ImportError("PyTorch required for pytorch format") from e

        else:
            raise ValueError(f"Unsupported format for deserialization: {fmt}")
