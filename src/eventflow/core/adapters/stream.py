"""StreamAdapter: Convert EventFrames to continuous-time sequences for Neural ODEs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from eventflow.core.adapters.base import (
    AdapterMetadata,
    BaseModalityAdapter,
    SerializationFormat,
)
from eventflow.core.adapters.configs import StreamAdapterConfig
from eventflow.core.utils import get_logger

if TYPE_CHECKING:
    from eventflow.core.event_frame import EventFrame

logger = get_logger(__name__)


@dataclass
class StreamOutput:
    """Output from StreamAdapter conversion.

    Attributes:
        timestamps: Continuous timestamps (n_events,) in seconds from origin
        states: State vectors at each timestamp (n_events, n_state_dims)
        inter_times: Inter-event times (n_events,) - time since previous event
        event_types: Event type indices (n_events,) for marked point processes
        intensity_features: Features for intensity function (n_events, n_intensity)
        origin: Origin timestamp for relative time
        time_scale_params: Parameters used for time scaling (mean, std, etc.)
        state_names: Names of state dimensions
        event_type_names: Names of event types
        dtypes: Data type information
    """

    timestamps: np.ndarray
    states: np.ndarray
    inter_times: np.ndarray
    event_types: np.ndarray | None = None
    intensity_features: np.ndarray | None = None
    origin: float = 0.0
    time_scale_params: dict[str, float] = field(default_factory=dict)
    state_names: list[str] = field(default_factory=list)
    event_type_names: list[str] = field(default_factory=list)
    dtypes: dict[str, str] = field(default_factory=dict)

    @property
    def n_events(self) -> int:
        """Number of events."""
        return len(self.timestamps)

    @property
    def n_state_dims(self) -> int:
        """Number of state dimensions."""
        return self.states.shape[1] if len(self.states.shape) > 1 else 1

    @property
    def n_event_types(self) -> int:
        """Number of unique event types."""
        if self.event_types is None:
            return 1
        return len(np.unique(self.event_types))

    def to_torch(self) -> dict[str, Any]:
        """Convert to PyTorch tensors."""
        try:
            import torch  # type: ignore

            result: dict[str, Any] = {
                "timestamps": torch.from_numpy(self.timestamps),
                "states": torch.from_numpy(self.states),
                "inter_times": torch.from_numpy(self.inter_times),
            }
            if self.event_types is not None:
                result["event_types"] = torch.from_numpy(self.event_types).long()
            if self.intensity_features is not None:
                result["intensity_features"] = torch.from_numpy(self.intensity_features)
            return result
        except ImportError as e:
            raise ImportError("PyTorch required for to_torch()") from e

    def get_sequences_by_type(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Split into sequences by event type.

        Returns:
            Dictionary mapping event type to (timestamps, states) arrays
        """
        if self.event_types is None:
            return {0: (self.timestamps, self.states)}

        result = {}
        for etype in np.unique(self.event_types):
            mask = self.event_types == etype
            result[int(etype)] = (self.timestamps[mask], self.states[mask])
        return result


class StreamAdapter(BaseModalityAdapter[StreamOutput]):
    """Convert EventFrames to continuous-time sequences for Neural ODEs.

    Creates (timestamp, state) sequences suitable for continuous-time models
    like Neural ODEs, Neural Hawkes processes, and other point process models.
    """

    def __init__(
        self,
        config: StreamAdapterConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize StreamAdapter.

        Args:
            config: Configuration object or None for defaults
            **kwargs: Configuration parameters (used if config is None)
        """
        if config is None:
            config = StreamAdapterConfig(**kwargs)
        self.config = config

    @property
    def modality(self) -> str:
        return "stream"

    def convert(self, event_frame: EventFrame) -> StreamOutput:
        """Convert EventFrame to stream format.

        Args:
            event_frame: The EventFrame to convert

        Returns:
            StreamOutput with continuous-time sequences
        """
        logger.info("Converting EventFrame to stream format")

        # Get configuration
        # Check if this is an EventFrame (has EventSchema with timestamp_col) vs plain DataFrame/LazyFrame
        # Avoid accessing .schema on Polars LazyFrame to prevent PerformanceWarning
        has_eventframe_schema = False
        event_schema = None
        # Polars LazyFrame has collect method but EventFrame does not (it wraps a DataFrame)
        is_polars_lazy = isinstance(event_frame, pl.LazyFrame)
        is_polars_dataframe = isinstance(event_frame, pl.DataFrame)

        if not is_polars_lazy and not is_polars_dataframe:
            # Must be EventFrame - access schema
            if hasattr(event_frame, "schema"):
                event_schema = event_frame.schema
                has_eventframe_schema = hasattr(event_schema, "timestamp_col")

        timestamp_col = self.config.timestamp_col
        if timestamp_col is None:
            if has_eventframe_schema and event_schema and event_schema.timestamp_col:
                timestamp_col = event_schema.timestamp_col
            else:
                timestamp_col = "timestamp"

        # Collect data and sort by timestamp
        if hasattr(event_frame, "collect"):
            df = event_frame.collect().sort(timestamp_col)
        else:
            df = event_frame.sort(timestamp_col)  # type: ignore[assignment]

        # Truncate if max_events specified
        if self.config.max_events is not None:
            df = df.head(self.config.max_events)

        # Determine state columns
        if self.config.state_cols is not None:
            state_cols = list(self.config.state_cols)
        else:
            exclude = {timestamp_col}
            if self.config.event_type_col:
                exclude.add(self.config.event_type_col)
            if has_eventframe_schema and event_schema:
                if event_schema.lat_col:
                    exclude.add(event_schema.lat_col)
                if event_schema.lon_col:
                    exclude.add(event_schema.lon_col)
                if event_schema.geometry_col:
                    exclude.add(event_schema.geometry_col)

            state_cols = [
                col
                for col in df.columns
                if col not in exclude
                and df[col].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
            ]

        # Determine dtype
        np_dtype = np.float32 if self.config.dtype == "float32" else np.float64

        # Convert timestamps to numeric (seconds from origin)
        ts_series = df[timestamp_col]
        if ts_series.dtype == pl.Datetime:
            # Convert to epoch seconds
            ts_numeric = ts_series.dt.epoch("s").to_numpy().astype(np_dtype)
        elif ts_series.dtype in (pl.Float32, pl.Float64, pl.Int64, pl.Int32):
            ts_numeric = ts_series.to_numpy().astype(np_dtype)
        else:
            raise ValueError(f"Unsupported timestamp type: {ts_series.dtype}")

        # Determine origin
        if self.config.time_origin == "first":
            origin = ts_numeric[0] if len(ts_numeric) > 0 else 0.0
        elif self.config.time_origin == "min":
            origin = ts_numeric.min() if len(ts_numeric) > 0 else 0.0
        elif self.config.time_origin == "custom" and self.config.custom_origin:
            try:
                origin_dt = datetime.fromisoformat(self.config.custom_origin)
                origin = origin_dt.timestamp()
            except ValueError:
                origin = float(self.config.custom_origin)
        else:
            origin = 0.0

        # Make timestamps relative to origin
        origin_val: float = float(origin) if origin is not None else 0.0
        timestamps = ts_numeric - origin_val  # type: ignore[operator]

        # Apply time scaling
        time_scale_params: dict[str, float] = {"origin": origin}

        if self.config.time_scale == "normalize":
            t_mean = float(np.mean(timestamps))
            t_std = float(np.std(timestamps))
            if t_std > 0:
                timestamps = (timestamps - t_mean) / t_std
            time_scale_params["mean"] = t_mean
            time_scale_params["std"] = t_std

        elif self.config.time_scale == "log":
            # Log-transform (shift to ensure positive)
            t_min = float(np.min(timestamps))
            timestamps = np.log1p(timestamps - t_min + 1.0)
            time_scale_params["min_shift"] = t_min

        # Compute inter-event times
        inter_times = np.zeros_like(timestamps)
        if len(timestamps) > 1:
            inter_times[1:] = np.diff(timestamps)

        # Extract states
        if state_cols:
            states = df.select(state_cols).to_numpy().astype(np_dtype)
        else:
            # Use timestamps as single state dimension
            states = timestamps.reshape(-1, 1)
            state_cols = ["time"]

        # Extract event types if specified
        event_types = None
        event_type_names: list[str] = []
        if self.config.event_type_col and self.config.event_type_col in df.columns:
            type_series = df[self.config.event_type_col]
            unique_types = type_series.unique().sort().to_list()
            type_to_idx = {t: i for i, t in enumerate(unique_types)}
            event_types = np.array([type_to_idx[t] for t in type_series.to_list()], dtype=np.int64)
            event_type_names = [str(t) for t in unique_types]

        # Extract intensity features if specified
        intensity_features = None
        if self.config.intensity_cols:
            intensity_cols = [c for c in self.config.intensity_cols if c in df.columns]
            if intensity_cols:
                intensity_features = df.select(intensity_cols).to_numpy().astype(np_dtype)

        logger.info(
            f"Created stream with {len(timestamps)} events, " f"{len(state_cols)} state dims"
        )

        return StreamOutput(
            timestamps=timestamps,
            states=states,
            inter_times=inter_times,
            event_types=event_types,
            intensity_features=intensity_features,
            origin=origin,
            time_scale_params=time_scale_params,
            state_names=state_cols,
            event_type_names=event_type_names,
            dtypes={"timestamps": str(np_dtype), "states": str(np_dtype)},
        )

    def get_metadata(self, output: StreamOutput) -> AdapterMetadata:
        """Extract metadata from stream output."""
        shapes = {
            "timestamps": output.timestamps.shape,
            "states": output.states.shape,
            "inter_times": output.inter_times.shape,
        }
        if output.event_types is not None:
            shapes["event_types"] = output.event_types.shape
        if output.intensity_features is not None:
            shapes["intensity_features"] = output.intensity_features.shape

        return AdapterMetadata(
            modality=self.modality,
            feature_names=output.state_names,
            shapes=shapes,
            dtypes=output.dtypes,
            temporal_info={
                "origin": output.origin,
                "scale_params": output.time_scale_params,
            },
            extra={
                "n_events": output.n_events,
                "n_state_dims": output.n_state_dims,
                "n_event_types": output.n_event_types,
                "event_type_names": output.event_type_names,
            },
        )

    def serialize(
        self,
        output: StreamOutput,
        path: Path | str,
        fmt: SerializationFormat | str,
    ) -> None:
        """Serialize stream output to disk."""
        path = Path(path)
        fmt = SerializationFormat(fmt) if isinstance(fmt, str) else fmt

        path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == SerializationFormat.NUMPY:
            save_dict: dict[str, Any] = {
                "timestamps": output.timestamps,
                "states": output.states,
                "inter_times": output.inter_times,
                "origin": np.array(output.origin),
                "state_names": np.array(output.state_names),
            }
            if output.event_types is not None:
                save_dict["event_types"] = output.event_types
                save_dict["event_type_names"] = np.array(output.event_type_names)
            if output.intensity_features is not None:
                save_dict["intensity_features"] = output.intensity_features
            np.savez(path, **save_dict)

        elif fmt == SerializationFormat.PYTORCH:
            try:
                import torch

                data = output.to_torch()
                data["origin"] = output.origin
                data["state_names"] = output.state_names
                data["event_type_names"] = output.event_type_names
                data["time_scale_params"] = output.time_scale_params
                torch.save(data, path)
            except ImportError as e:
                raise ImportError("PyTorch required for pytorch format") from e

        elif fmt == SerializationFormat.PARQUET:
            # Store as tabular format
            data_dict = {
                "timestamp": output.timestamps,
                "inter_time": output.inter_times,
            }
            for i, name in enumerate(output.state_names):
                data_dict[name] = output.states[:, i]
            if output.event_types is not None:
                data_dict["event_type"] = output.event_types

            df = pl.DataFrame(data_dict)
            df.write_parquet(path)

            # Write metadata
            meta_path = path.with_suffix(".meta.json")
            meta = self.get_metadata(output)
            meta_path.write_text(json.dumps(meta.to_dict(), indent=2))

        elif fmt == SerializationFormat.JSON:
            # Convert numpy types to Python native types for JSON serialization
            scale_params = {k: float(v) for k, v in output.time_scale_params.items()}
            data = {
                "timestamps": output.timestamps.tolist(),
                "states": output.states.tolist(),
                "inter_times": output.inter_times.tolist(),
                "origin": float(output.origin),
                "time_scale_params": scale_params,
                "state_names": output.state_names,
            }
            if output.event_types is not None:
                data["event_types"] = output.event_types.tolist()
                data["event_type_names"] = output.event_type_names
            path.write_text(json.dumps(data, indent=2))

        else:
            raise ValueError(f"Unsupported format for StreamAdapter: {fmt}")

        logger.info(f"Serialized stream to {path} as {fmt}")

    def deserialize(
        self,
        path: Path | str,
        fmt: SerializationFormat | str,
    ) -> StreamOutput:
        """Deserialize stream output from disk."""
        path = Path(path)
        fmt = SerializationFormat(fmt) if isinstance(fmt, str) else fmt

        if fmt == SerializationFormat.NUMPY:
            loaded = np.load(path, allow_pickle=True)
            event_types = loaded.get("event_types")
            intensity_feats = loaded.get("intensity_features")
            event_type_names = loaded.get("event_type_names")
            return StreamOutput(
                timestamps=loaded["timestamps"],
                states=loaded["states"],
                inter_times=loaded["inter_times"],
                event_types=event_types if event_types is not None else None,
                intensity_features=intensity_feats if intensity_feats is not None else None,
                origin=float(loaded["origin"]),
                state_names=list(loaded["state_names"]),
                event_type_names=(list(event_type_names) if event_type_names is not None else []),
            )

        elif fmt == SerializationFormat.PYTORCH:
            try:
                import torch

                data = torch.load(path, weights_only=False)
                event_types = data.get("event_types")
                intensity_feats = data.get("intensity_features")
                return StreamOutput(
                    timestamps=data["timestamps"].numpy(),
                    states=data["states"].numpy(),
                    inter_times=data["inter_times"].numpy(),
                    event_types=event_types.numpy() if event_types is not None else None,
                    intensity_features=(
                        intensity_feats.numpy() if intensity_feats is not None else None
                    ),
                    origin=data["origin"],
                    time_scale_params=data.get("time_scale_params", {}),
                    state_names=data["state_names"],
                    event_type_names=data.get("event_type_names", []),
                )
            except ImportError as e:
                raise ImportError("PyTorch required for pytorch format") from e

        elif fmt == SerializationFormat.JSON:
            data = json.loads(path.read_text())
            event_types = data.get("event_types")
            return StreamOutput(
                timestamps=np.array(data["timestamps"]),
                states=np.array(data["states"]),
                inter_times=np.array(data["inter_times"]),
                event_types=np.array(event_types) if event_types else None,
                origin=data["origin"],
                time_scale_params=data.get("time_scale_params", {}),
                state_names=data["state_names"],
                event_type_names=data.get("event_type_names", []),
            )

        else:
            raise ValueError(f"Unsupported format for deserialization: {fmt}")
