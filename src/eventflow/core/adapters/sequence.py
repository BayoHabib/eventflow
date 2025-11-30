"""SequenceAdapter: Convert EventFrames to padded sequences for RNN/Transformers."""

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
from eventflow.core.adapters.configs import SequenceAdapterConfig
from eventflow.core.utils import get_logger

if TYPE_CHECKING:
    from eventflow.core.event_frame import EventFrame

logger = get_logger(__name__)


@dataclass
class SequenceOutput:
    """Output from SequenceAdapter conversion.

    Attributes:
        sequences: 3D array of shape (n_sequences, seq_length, n_features)
        masks: Boolean mask of shape (n_sequences, seq_length) indicating valid positions
        sequence_ids: Identifiers for each sequence (e.g., grid cell IDs)
        feature_names: List of feature names
        lengths: Original lengths before padding
        time_encoding: Optional time position encodings
        dtypes: Data type information
    """

    sequences: np.ndarray
    masks: np.ndarray
    sequence_ids: list[Any] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    lengths: np.ndarray = field(default_factory=lambda: np.array([]))
    time_encoding: np.ndarray | None = None
    dtypes: dict[str, str] = field(default_factory=dict)

    @property
    def n_sequences(self) -> int:
        """Number of sequences."""
        return int(self.sequences.shape[0])

    @property
    def max_length(self) -> int:
        """Maximum sequence length."""
        return int(self.sequences.shape[1])

    @property
    def n_features(self) -> int:
        """Number of features per timestep."""
        return int(self.sequences.shape[2])

    def to_torch(self) -> dict[str, Any]:
        """Convert to PyTorch tensors."""
        try:
            import torch

            result: dict[str, Any] = {
                "sequences": torch.from_numpy(self.sequences),
                "masks": torch.from_numpy(self.masks),
                "lengths": torch.from_numpy(self.lengths),
            }
            if self.time_encoding is not None:
                result["time_encoding"] = torch.from_numpy(self.time_encoding)
            return result
        except ImportError as e:
            raise ImportError("PyTorch required for to_torch()") from e


class SequenceAdapter(BaseModalityAdapter[SequenceOutput]):
    """Convert EventFrames to padded sequences for RNN/Transformer models.

    Creates sequences per spatial unit (e.g., grid cell), with proper
    padding and attention masks for variable-length sequences.
    """

    def __init__(
        self,
        config: SequenceAdapterConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize SequenceAdapter.

        Args:
            config: Configuration object or None for defaults
            **kwargs: Configuration parameters (used if config is None)
        """
        if config is None:
            config = SequenceAdapterConfig(**kwargs)
        self.config = config

    @property
    def modality(self) -> str:
        return "sequence"

    def convert(self, event_frame: EventFrame) -> SequenceOutput:
        """Convert EventFrame to sequence format.

        Args:
            event_frame: The EventFrame to convert

        Returns:
            SequenceOutput with padded sequences and masks
        """
        logger.info("Converting EventFrame to sequence format")

        # Get configuration
        spatial_col = self.config.spatial_col
        timestamp_col = self.config.timestamp_col or event_frame.schema.timestamp_col

        if spatial_col is None:
            # Try to infer spatial column
            if "grid_id" in event_frame.lazy_frame.collect_schema().names():
                spatial_col = "grid_id"
            elif "zone_id" in event_frame.lazy_frame.collect_schema().names():
                spatial_col = "zone_id"
            else:
                raise ValueError("spatial_col must be specified or grid_id/zone_id must exist")

        # Collect data
        df = event_frame.collect()

        # Determine feature columns
        if self.config.feature_cols is not None:
            feature_cols = list(self.config.feature_cols)
        else:
            exclude = {spatial_col, timestamp_col}
            if event_frame.schema.lat_col:
                exclude.add(event_frame.schema.lat_col)
            if event_frame.schema.lon_col:
                exclude.add(event_frame.schema.lon_col)
            if event_frame.schema.geometry_col:
                exclude.add(event_frame.schema.geometry_col)

            feature_cols = [
                col
                for col in df.columns
                if col not in exclude
                and df[col].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
            ]

        # Sort by spatial unit and timestamp
        df = df.sort([spatial_col, timestamp_col])

        # Group by spatial unit
        groups = df.group_by(spatial_col, maintain_order=True)

        # Collect sequences
        sequences_list: list[np.ndarray] = []
        lengths_list: list[int] = []
        sequence_ids: list[Any] = []

        for (spatial_id,), group_df in groups:
            seq_data = group_df.select(feature_cols).to_numpy()
            sequences_list.append(seq_data)
            lengths_list.append(len(seq_data))
            sequence_ids.append(spatial_id)

        # Determine max sequence length
        if self.config.sequence_length is not None:
            max_len = self.config.sequence_length
        else:
            max_len = max(lengths_list) if lengths_list else 0

        n_sequences = len(sequences_list)
        n_features = len(feature_cols)

        # Determine dtype
        np_dtype = np.float32 if self.config.dtype == "float32" else np.float64

        # Create padded arrays
        sequences = np.full(
            (n_sequences, max_len, n_features),
            self.config.padding_value,
            dtype=np_dtype,
        )
        masks = np.zeros((n_sequences, max_len), dtype=bool)
        lengths = np.array(lengths_list, dtype=np.int64)

        # Fill in sequences
        for i, (seq, length) in enumerate(zip(sequences_list, lengths_list, strict=True)):
            actual_len = min(length, max_len)
            if self.config.padding_side == "right":
                sequences[i, :actual_len, :] = seq[:actual_len]
                masks[i, :actual_len] = True
            else:  # left padding
                sequences[i, -actual_len:, :] = seq[:actual_len]
                masks[i, -actual_len:] = True

        # Add time encoding if requested
        time_encoding = None
        if self.config.time_encoding == "positional":
            time_encoding = np.arange(max_len, dtype=np_dtype)[np.newaxis, :]
            time_encoding = np.broadcast_to(time_encoding, (n_sequences, max_len)).copy()
        elif self.config.time_encoding == "sinusoidal":
            positions = np.arange(max_len, dtype=np_dtype)
            d_model = n_features
            time_encoding = np.zeros((max_len, d_model), dtype=np_dtype)
            for i in range(0, d_model, 2):
                div_term = np.exp(-np.log(10000.0) * i / d_model)
                time_encoding[:, i] = np.sin(positions * div_term)
                if i + 1 < d_model:
                    time_encoding[:, i + 1] = np.cos(positions * div_term)
            time_encoding = np.broadcast_to(
                time_encoding[np.newaxis, :, :], (n_sequences, max_len, d_model)
            ).copy()

        logger.info(
            f"Created {n_sequences} sequences with max_length={max_len}, "
            f"n_features={n_features}"
        )

        return SequenceOutput(
            sequences=sequences,
            masks=masks,
            sequence_ids=sequence_ids,
            feature_names=feature_cols,
            lengths=lengths,
            time_encoding=time_encoding,
            dtypes={"sequences": str(np_dtype), "masks": "bool"},
        )

    def get_metadata(self, output: SequenceOutput) -> AdapterMetadata:
        """Extract metadata from sequence output."""
        shapes = {
            "sequences": output.sequences.shape,
            "masks": output.masks.shape,
            "lengths": output.lengths.shape,
        }
        if output.time_encoding is not None:
            shapes["time_encoding"] = output.time_encoding.shape

        return AdapterMetadata(
            modality=self.modality,
            feature_names=output.feature_names,
            shapes=shapes,
            dtypes=output.dtypes,
            extra={
                "n_sequences": output.n_sequences,
                "max_length": output.max_length,
                "n_features": output.n_features,
                "sequence_ids": output.sequence_ids,
            },
        )

    def serialize(
        self,
        output: SequenceOutput,
        path: Path | str,
        fmt: SerializationFormat | str,
    ) -> None:
        """Serialize sequence output to disk."""
        path = Path(path)
        fmt = SerializationFormat(fmt) if isinstance(fmt, str) else fmt

        path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == SerializationFormat.NUMPY:
            save_dict: dict[str, np.ndarray] = {
                "sequences": output.sequences,
                "masks": output.masks,
                "lengths": output.lengths,
                "feature_names": np.array(output.feature_names),
                "sequence_ids": np.array(output.sequence_ids),
            }
            if output.time_encoding is not None:
                save_dict["time_encoding"] = output.time_encoding
            np.savez(path, **save_dict)  # type: ignore[arg-type]

        elif fmt == SerializationFormat.PYTORCH:
            try:
                import torch

                data = output.to_torch()
                data["feature_names"] = output.feature_names
                data["sequence_ids"] = output.sequence_ids
                torch.save(data, path)
            except ImportError as e:
                raise ImportError("PyTorch required for pytorch format") from e

        elif fmt == SerializationFormat.ARROW:
            # Flatten for Arrow storage
            flat_data = {
                "sequence_idx": np.repeat(np.arange(output.n_sequences), output.max_length),
                "position": np.tile(np.arange(output.max_length), output.n_sequences),
                "valid": output.masks.flatten(),
            }
            for i, name in enumerate(output.feature_names):
                flat_data[name] = output.sequences[:, :, i].flatten()

            df = pl.DataFrame(flat_data)
            df.write_ipc(path)

            # Write metadata
            meta_path = path.with_suffix(".meta.json")
            meta = self.get_metadata(output)
            meta_path.write_text(json.dumps(meta.to_dict(), indent=2))

        else:
            raise ValueError(f"Unsupported format for SequenceAdapter: {fmt}")

        logger.info(f"Serialized sequences to {path} as {fmt}")

    def deserialize(
        self,
        path: Path | str,
        fmt: SerializationFormat | str,
    ) -> SequenceOutput:
        """Deserialize sequence output from disk."""
        path = Path(path)
        fmt = SerializationFormat(fmt) if isinstance(fmt, str) else fmt

        if fmt == SerializationFormat.NUMPY:
            loaded = np.load(path, allow_pickle=True)
            time_enc = loaded.get("time_encoding")
            return SequenceOutput(
                sequences=loaded["sequences"],
                masks=loaded["masks"],
                lengths=loaded["lengths"],
                feature_names=list(loaded["feature_names"]),
                sequence_ids=list(loaded["sequence_ids"]),
                time_encoding=time_enc if time_enc is not None else None,
            )

        elif fmt == SerializationFormat.PYTORCH:
            try:
                import torch

                data = torch.load(path, weights_only=False)
                time_enc = data.get("time_encoding")
                return SequenceOutput(
                    sequences=data["sequences"].numpy(),
                    masks=data["masks"].numpy(),
                    lengths=data["lengths"].numpy(),
                    feature_names=data["feature_names"],
                    sequence_ids=data["sequence_ids"],
                    time_encoding=time_enc.numpy() if time_enc is not None else None,
                )
            except ImportError as e:
                raise ImportError("PyTorch required for pytorch format") from e

        else:
            raise ValueError(f"Unsupported format for deserialization: {fmt}")
