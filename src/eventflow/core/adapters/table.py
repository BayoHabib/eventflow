"""TableAdapter: Convert EventFrames to tabular DataFrames for GLM/Poisson."""

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
from eventflow.core.adapters.configs import TableAdapterConfig
from eventflow.core.utils import get_logger

if TYPE_CHECKING:
    import pandas as pd  # type: ignore

    from eventflow.core.event_frame import EventFrame

logger = get_logger(__name__)


@dataclass
class TableOutput:
    """Output from TableAdapter conversion.

    Attributes:
        data: The converted DataFrame (Polars)
        feature_names: List of feature column names
        target: Target column name (if specified)
        offset: Offset column name (if specified)
        weights: Weight column name (if specified)
        index_cols: Index column names
        dtypes: Column dtypes mapping
    """

    data: pl.DataFrame
    feature_names: list[str] = field(default_factory=list)
    target: str | None = None
    offset: str | None = None
    weights: str | None = None
    index_cols: list[str] = field(default_factory=list)
    dtypes: dict[str, str] = field(default_factory=dict)

    def to_pandas(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        pdf = self.data.to_pandas()
        if self.index_cols:
            pdf = pdf.set_index(self.index_cols)
        return pdf

    def to_numpy(self, include_target: bool = False) -> np.ndarray:
        """Convert features to numpy array.

        Args:
            include_target: Whether to include target column

        Returns:
            numpy array of shape (n_samples, n_features)
        """
        cols = list(self.feature_names)
        if include_target and self.target:
            cols.append(self.target)
        return self.data.select(cols).to_numpy()

    def get_X_y(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Get feature matrix X and target vector y.

        Returns:
            Tuple of (X, y) where y is None if no target specified
        """
        X = self.to_numpy(include_target=False)
        y = None
        if self.target:
            y = self.data[self.target].to_numpy()
        return X, y


class TableAdapter(BaseModalityAdapter[TableOutput]):
    """Convert EventFrames to tabular DataFrames for GLM/Poisson regression.

    Supports:
    - Offset/exposure columns for count models
    - Observation weights
    - Categorical encoding options
    - Intercept term for GLM
    """

    def __init__(
        self,
        config: TableAdapterConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize TableAdapter.

        Args:
            config: Configuration object or None for defaults
            **kwargs: Configuration parameters (used if config is None)
        """
        if config is None:
            config = TableAdapterConfig(**kwargs)
        self.config = config

    @property
    def modality(self) -> str:
        return "table"

    def convert(self, event_frame: EventFrame) -> TableOutput:
        """Convert EventFrame to tabular format.

        Args:
            event_frame: The EventFrame to convert

        Returns:
            TableOutput with DataFrame and metadata
        """
        logger.info("Converting EventFrame to table format")

        # Check if this is an EventFrame (has EventSchema) vs plain DataFrame/LazyFrame
        has_eventframe_schema = False
        event_schema = None
        is_polars_lazy = isinstance(event_frame, pl.LazyFrame)
        is_polars_dataframe = isinstance(event_frame, pl.DataFrame)

        if not is_polars_lazy and not is_polars_dataframe:
            if hasattr(event_frame, "schema"):
                event_schema = event_frame.schema
                has_eventframe_schema = hasattr(event_schema, "timestamp_col")

        # Collect the lazy frame if needed
        if hasattr(event_frame, "collect"):
            df = event_frame.collect()
        else:
            df = event_frame  # type: ignore[assignment]

        # Determine feature columns
        if self.config.feature_cols is not None:
            feature_cols = list(self.config.feature_cols)
        else:
            # Auto-detect: numeric columns excluding target/offset/weight/index
            exclude_cols = set(self.config.index_cols)
            if self.config.target_col:
                exclude_cols.add(self.config.target_col)
            if self.config.offset_col:
                exclude_cols.add(self.config.offset_col)
            if self.config.weight_col:
                exclude_cols.add(self.config.weight_col)

            # Also exclude timestamp and geometry columns if EventFrame
            if has_eventframe_schema and event_schema:
                exclude_cols.add(event_schema.timestamp_col)
                if event_schema.lat_col:
                    exclude_cols.add(event_schema.lat_col)
                if event_schema.lon_col:
                    exclude_cols.add(event_schema.lon_col)
                if event_schema.geometry_col:
                    exclude_cols.add(event_schema.geometry_col)

            feature_cols = [
                col
                for col in df.columns
                if col not in exclude_cols
                and df[col].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
            ]

        # Build output columns
        output_cols = list(self.config.index_cols)

        # Add intercept if requested
        if self.config.include_intercept:
            df = df.with_columns(pl.lit(1.0).alias("_intercept"))
            feature_cols = ["_intercept"] + feature_cols

        output_cols.extend(feature_cols)

        # Add target, offset, weight columns
        if self.config.target_col and self.config.target_col in df.columns:
            output_cols.append(self.config.target_col)
        if self.config.offset_col and self.config.offset_col in df.columns:
            output_cols.append(self.config.offset_col)
        if self.config.weight_col and self.config.weight_col in df.columns:
            output_cols.append(self.config.weight_col)

        # Select and cast to target dtype
        df = df.select([col for col in output_cols if col in df.columns])

        target_dtype = pl.Float32 if self.config.dtype == "float32" else pl.Float64
        cast_exprs = []
        for col in feature_cols:
            if col in df.columns:
                cast_exprs.append(pl.col(col).cast(target_dtype))

        if cast_exprs:
            df = df.with_columns(cast_exprs)

        # Build dtype mapping
        dtypes = {col: str(df[col].dtype) for col in df.columns}

        logger.info(f"Created table with {len(df)} rows, {len(feature_cols)} features")

        return TableOutput(
            data=df,
            feature_names=feature_cols,
            target=self.config.target_col,
            offset=self.config.offset_col,
            weights=self.config.weight_col,
            index_cols=list(self.config.index_cols),
            dtypes=dtypes,
        )

    def get_metadata(self, output: TableOutput) -> AdapterMetadata:
        """Extract metadata from table output."""
        return AdapterMetadata(
            modality=self.modality,
            feature_names=output.feature_names,
            shapes={"data": (len(output.data), len(output.data.columns))},
            dtypes=output.dtypes,
            extra={
                "target": output.target,
                "offset": output.offset,
                "weights": output.weights,
                "index_cols": output.index_cols,
            },
        )

    def serialize(
        self,
        output: TableOutput,
        path: Path | str,
        fmt: SerializationFormat | str,
    ) -> None:
        """Serialize table output to disk."""
        path = Path(path)
        fmt = SerializationFormat(fmt) if isinstance(fmt, str) else fmt

        path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == SerializationFormat.PARQUET:
            output.data.write_parquet(path)
            # Write metadata sidecar
            meta_path = path.with_suffix(".meta.json")
            meta = self.get_metadata(output)
            meta_path.write_text(json.dumps(meta.to_dict(), indent=2))

        elif fmt == SerializationFormat.ARROW:
            output.data.write_ipc(path)
            meta_path = path.with_suffix(".meta.json")
            meta = self.get_metadata(output)
            meta_path.write_text(json.dumps(meta.to_dict(), indent=2))

        elif fmt == SerializationFormat.NUMPY:
            X, y = output.get_X_y()
            np.savez(
                path,
                X=X,
                y=y if y is not None else np.array([]),
                feature_names=np.array(output.feature_names),
            )

        elif fmt == SerializationFormat.PYTORCH:
            try:
                import torch  # type: ignore

                X, y = output.get_X_y()
                data: dict[str, Any] = {
                    "X": torch.from_numpy(X),
                    "feature_names": output.feature_names,
                }
                if y is not None:
                    data["y"] = torch.from_numpy(y)
                torch.save(data, path)
            except ImportError as e:
                raise ImportError("PyTorch required for pytorch format") from e

        else:
            raise ValueError(f"Unsupported format for TableAdapter: {fmt}")

        logger.info(f"Serialized table to {path} as {fmt}")

    def deserialize(
        self,
        path: Path | str,
        fmt: SerializationFormat | str,
    ) -> TableOutput:
        """Deserialize table output from disk."""
        path = Path(path)
        fmt = SerializationFormat(fmt) if isinstance(fmt, str) else fmt

        if fmt == SerializationFormat.PARQUET:
            df = pl.read_parquet(path)
            meta_path = path.with_suffix(".meta.json")
            if meta_path.exists():
                meta_dict = json.loads(meta_path.read_text())
                meta = AdapterMetadata.from_dict(meta_dict)
                return TableOutput(
                    data=df,
                    feature_names=meta.feature_names,
                    target=meta.extra.get("target"),
                    offset=meta.extra.get("offset"),
                    weights=meta.extra.get("weights"),
                    index_cols=meta.extra.get("index_cols", []),
                    dtypes=meta.dtypes,
                )
            return TableOutput(data=df)

        elif fmt == SerializationFormat.ARROW:
            df = pl.read_ipc(path)
            meta_path = path.with_suffix(".meta.json")
            if meta_path.exists():
                meta_dict = json.loads(meta_path.read_text())
                meta = AdapterMetadata.from_dict(meta_dict)
                return TableOutput(
                    data=df,
                    feature_names=meta.feature_names,
                    target=meta.extra.get("target"),
                    offset=meta.extra.get("offset"),
                    weights=meta.extra.get("weights"),
                    index_cols=meta.extra.get("index_cols", []),
                    dtypes=meta.dtypes,
                )
            return TableOutput(data=df)

        elif fmt == SerializationFormat.NUMPY:
            loaded = np.load(path, allow_pickle=True)
            X = loaded["X"]
            feature_names = list(loaded["feature_names"])
            df = pl.DataFrame({name: X[:, i] for i, name in enumerate(feature_names)})
            return TableOutput(data=df, feature_names=feature_names)

        else:
            raise ValueError(f"Unsupported format for deserialization: {fmt}")
