"""GraphAdapter: Convert EventFrames to graph structures for GNNs."""

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
from eventflow.core.adapters.configs import GraphAdapterConfig
from eventflow.core.utils import get_logger

if TYPE_CHECKING:
    from eventflow.core.event_frame import EventFrame

logger = get_logger(__name__)


@dataclass
class GraphSnapshot:
    """A single temporal snapshot of the graph.

    Attributes:
        timestamp: Timestamp for this snapshot
        node_features: Node feature matrix (n_nodes, n_features)
        edge_index: Edge indices (2, n_edges) in COO format
        edge_features: Edge feature matrix (n_edges, n_edge_features) or None
        node_ids: Original node identifiers
    """

    timestamp: Any
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray | None = None
    node_ids: list[Any] = field(default_factory=list)


@dataclass
class GraphOutput:
    """Output from GraphAdapter conversion.

    Attributes:
        node_features: Node feature matrix (n_nodes, n_features) for static graph
                      or list of matrices for temporal snapshots
        edge_index: Edge indices in COO format (2, n_edges)
        edge_features: Edge feature matrix (n_edges, n_edge_features) or None
        adjacency: Dense adjacency matrix (n_nodes, n_nodes)
        node_ids: Original node identifiers
        feature_names: Names of node features
        edge_feature_names: Names of edge features
        snapshots: List of temporal snapshots (for dynamic graphs)
        dtypes: Data type information
    """

    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray | None = None
    adjacency: np.ndarray | None = None
    node_ids: list[Any] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    edge_feature_names: list[str] = field(default_factory=list)
    snapshots: list[GraphSnapshot] = field(default_factory=list)
    dtypes: dict[str, str] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return int(self.node_features.shape[0])

    @property
    def n_edges(self) -> int:
        """Number of edges."""
        return int(self.edge_index.shape[1])

    @property
    def n_features(self) -> int:
        """Number of node features."""
        return int(self.node_features.shape[1])

    @property
    def n_snapshots(self) -> int:
        """Number of temporal snapshots."""
        return len(self.snapshots)

    def to_torch_geometric(self) -> dict[str, Any]:
        """Convert to PyTorch Geometric format."""
        try:
            import torch  # type: ignore[import-not-found]

            result: dict[str, Any] = {
                "x": torch.from_numpy(self.node_features),
                "edge_index": torch.from_numpy(self.edge_index).long(),
            }
            if self.edge_features is not None:
                result["edge_attr"] = torch.from_numpy(self.edge_features)
            return result
        except ImportError as e:
            raise ImportError("PyTorch required for to_torch_geometric()") from e

    def to_dgl(self) -> Any:
        """Convert to DGL graph format."""
        try:
            import dgl  # type: ignore[import-not-found]
            import torch

            src = self.edge_index[0]
            dst = self.edge_index[1]
            g = dgl.graph((src, dst), num_nodes=self.n_nodes)
            g.ndata["feat"] = torch.from_numpy(self.node_features)
            if self.edge_features is not None:
                g.edata["feat"] = torch.from_numpy(self.edge_features)
            return g
        except ImportError as e:
            raise ImportError("DGL and PyTorch required for to_dgl()") from e


class GraphAdapter(BaseModalityAdapter[GraphOutput]):
    """Convert EventFrames to graph structures for GNN models.

    Creates node/edge arrays and adjacency matrices, with optional
    temporal snapshots for dynamic graph neural networks.
    """

    def __init__(
        self,
        config: GraphAdapterConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize GraphAdapter.

        Args:
            config: Configuration object or None for defaults
            **kwargs: Configuration parameters (used if config is None)
        """
        if config is None:
            config = GraphAdapterConfig(**kwargs)
        self.config = config

    @property
    def modality(self) -> str:
        return "graph"

    def convert(self, event_frame: EventFrame) -> GraphOutput:
        """Convert EventFrame to graph format.

        Args:
            event_frame: The EventFrame to convert

        Returns:
            GraphOutput with node/edge arrays and adjacency matrix
        """
        logger.info("Converting EventFrame to graph format")

        # Get configuration
        node_col = self.config.node_col or "grid_id"
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

        if node_col not in df.columns:
            raise ValueError(f"Node column '{node_col}' not found in EventFrame")

        # Auto-detect feature columns
        if self.config.feature_cols is not None:
            feature_cols = list(self.config.feature_cols)
        else:
            exclude = {node_col, timestamp_col}
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

        # Get unique nodes
        node_ids = df[node_col].unique().sort().to_list()
        node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        n_nodes = len(node_ids)

        # Determine dtype
        np_dtype = np.float32 if self.config.dtype == "float32" else np.float64

        # Aggregate node features
        node_agg = (
            df.group_by(node_col)
            .agg([pl.col(c).mean().alias(c) for c in feature_cols])
            .sort(node_col)
        )

        node_features = np.zeros((n_nodes, len(feature_cols)), dtype=np_dtype)
        for row in node_agg.iter_rows(named=True):
            idx = node_to_idx[row[node_col]]
            for j, feat in enumerate(feature_cols):
                val = row.get(feat)
                if val is not None:
                    node_features[idx, j] = val

        # Build adjacency based on type
        edge_list: list[tuple[int, int]] = []
        edge_weights: list[float] = []

        if self.config.adjacency_type in ("spatial", "both"):
            edge_list, edge_weights = self._build_spatial_edges(
                df, node_col, node_to_idx, event_frame
            )

        # Add self-loops if requested
        if self.config.include_self_loops:
            for i in range(n_nodes):
                edge_list.append((i, i))
                edge_weights.append(1.0)

        # Create edge index
        if edge_list:
            edge_index = np.array(edge_list, dtype=np.int64).T
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)

        # Create adjacency matrix
        adjacency = np.zeros((n_nodes, n_nodes), dtype=np_dtype)
        for (src, dst), w in zip(edge_list, edge_weights, strict=True):
            adjacency[src, dst] = w

        # Normalize adjacency if requested
        if self.config.normalize_adjacency and n_nodes > 0:
            # Symmetric normalization: D^(-1/2) A D^(-1/2)
            degree = adjacency.sum(axis=1)
            degree[degree == 0] = 1  # Avoid division by zero
            d_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
            adjacency = d_inv_sqrt @ adjacency @ d_inv_sqrt

        # Edge features
        edge_features = None
        if edge_weights and self.config.edge_feature_cols:
            edge_features = np.array(edge_weights, dtype=np_dtype).reshape(-1, 1)

        # Build temporal snapshots if interval specified
        snapshots: list[GraphSnapshot] = []
        if self.config.snapshot_interval:
            snapshots = self._build_snapshots(
                df,
                node_col,
                timestamp_col,
                feature_cols,
                node_to_idx,
                node_ids,
                edge_index,
                np_dtype,
            )

        logger.info(
            f"Created graph with {n_nodes} nodes, {len(edge_list)} edges, "
            f"{len(feature_cols)} features"
        )

        return GraphOutput(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            adjacency=adjacency,
            node_ids=node_ids,
            feature_names=feature_cols,
            edge_feature_names=self.config.edge_feature_cols or [],
            snapshots=snapshots,
            dtypes={"node_features": str(np_dtype), "adjacency": str(np_dtype)},
        )

    def _build_spatial_edges(
        self,
        df: pl.DataFrame,
        node_col: str,
        node_to_idx: dict[Any, int],
        event_frame: EventFrame,
    ) -> tuple[list[tuple[int, int]], list[float]]:
        """Build edges based on spatial proximity."""
        edge_list: list[tuple[int, int]] = []
        edge_weights: list[float] = []

        # Handle both EventFrame (has EventSchema) and plain DataFrame (has Polars Schema)
        if hasattr(event_frame, "schema") and hasattr(event_frame.schema, "lat_col"):
            lat_col = event_frame.schema.lat_col
            lon_col = event_frame.schema.lon_col
        else:
            # For plain DataFrames, try common column names or return empty edges
            lat_col = None
            lon_col = None
            for lat_name in ["latitude", "lat", "Latitude", "LAT"]:
                if lat_name in df.columns:
                    lat_col = lat_name
                    break
            for lon_name in ["longitude", "lon", "lng", "Longitude", "LON", "LNG"]:
                if lon_name in df.columns:
                    lon_col = lon_name
                    break

        if lat_col and lon_col and lat_col in df.columns and lon_col in df.columns:
            # Get node centroids
            node_coords = (
                df.group_by(node_col)
                .agg([pl.col(lat_col).mean(), pl.col(lon_col).mean()])
                .sort(node_col)
            )

            coords = node_coords.select([lat_col, lon_col]).to_numpy()
            n_nodes = len(coords)

            threshold = self.config.spatial_threshold
            if threshold is None:
                # Default: connect nodes within 1km (rough estimate)
                threshold = 0.01  # ~1km in degrees

            # Build edges based on distance threshold
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    dist = np.sqrt(
                        (coords[i, 0] - coords[j, 0]) ** 2 + (coords[i, 1] - coords[j, 1]) ** 2
                    )
                    if dist <= threshold:
                        edge_list.append((i, j))
                        edge_list.append((j, i))  # Undirected
                        weight = 1.0 / (dist + 1e-6)
                        edge_weights.extend([weight, weight])
        else:
            # Fallback: grid adjacency (4-connectivity)
            node_ids = list(node_to_idx.keys())
            for nid in node_ids:
                if isinstance(nid, str) and "_" in nid:
                    try:
                        row, col = map(int, nid.split("_"))
                        # Check 4-neighbors
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            neighbor = f"{row + dr}_{col + dc}"
                            if neighbor in node_to_idx:
                                src = node_to_idx[nid]
                                dst = node_to_idx[neighbor]
                                edge_list.append((src, dst))
                                edge_weights.append(1.0)
                    except ValueError:
                        pass

        return edge_list, edge_weights

    def _build_snapshots(
        self,
        df: pl.DataFrame,
        node_col: str,
        timestamp_col: str,
        feature_cols: list[str],
        node_to_idx: dict[Any, int],
        node_ids: list[Any],
        edge_index: np.ndarray,
        np_dtype: type,
    ) -> list[GraphSnapshot]:
        """Build temporal graph snapshots."""
        snapshots: list[GraphSnapshot] = []

        # Group by time interval
        interval = self.config.snapshot_interval or "1d"
        df = df.with_columns(pl.col(timestamp_col).dt.truncate(interval).alias("_snapshot_time"))

        snapshot_times = df["_snapshot_time"].unique().sort().to_list()

        n_nodes = len(node_ids)

        for ts in snapshot_times:
            snapshot_df = df.filter(pl.col("_snapshot_time") == ts)

            # Aggregate features for this snapshot
            node_agg = snapshot_df.group_by(node_col).agg(
                [pl.col(c).mean().alias(c) for c in feature_cols]
            )

            snapshot_features: np.ndarray = np.zeros((n_nodes, len(feature_cols)), dtype=np_dtype)
            for row in node_agg.iter_rows(named=True):
                if row[node_col] in node_to_idx:
                    idx = node_to_idx[row[node_col]]
                    for j, feat in enumerate(feature_cols):
                        val = row.get(feat)
                        if val is not None:
                            snapshot_features[idx, j] = val

            snapshots.append(
                GraphSnapshot(
                    timestamp=ts,
                    node_features=snapshot_features,
                    edge_index=edge_index.copy(),
                    node_ids=node_ids,
                )
            )

        return snapshots

    def get_metadata(self, output: GraphOutput) -> AdapterMetadata:
        """Extract metadata from graph output."""
        shapes = {
            "node_features": output.node_features.shape,
            "edge_index": output.edge_index.shape,
        }
        if output.adjacency is not None:
            shapes["adjacency"] = output.adjacency.shape
        if output.edge_features is not None:
            shapes["edge_features"] = output.edge_features.shape

        return AdapterMetadata(
            modality=self.modality,
            feature_names=output.feature_names,
            shapes=shapes,
            dtypes=output.dtypes,
            extra={
                "n_nodes": output.n_nodes,
                "n_edges": output.n_edges,
                "n_snapshots": output.n_snapshots,
                "edge_feature_names": output.edge_feature_names,
            },
        )

    def serialize(
        self,
        output: GraphOutput,
        path: Path | str,
        fmt: SerializationFormat | str,
    ) -> None:
        """Serialize graph output to disk."""
        path = Path(path)
        fmt = SerializationFormat(fmt) if isinstance(fmt, str) else fmt

        path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == SerializationFormat.NUMPY:
            save_dict: dict[str, Any] = {
                "node_features": output.node_features,
                "edge_index": output.edge_index,
                "node_ids": np.array(output.node_ids, dtype=object),
                "feature_names": np.array(output.feature_names),
            }
            if output.adjacency is not None:
                save_dict["adjacency"] = output.adjacency
            if output.edge_features is not None:
                save_dict["edge_features"] = output.edge_features
            np.savez(path, **save_dict)

        elif fmt == SerializationFormat.PYTORCH:
            try:
                import torch

                data = output.to_torch_geometric()
                data["node_ids"] = output.node_ids
                data["feature_names"] = output.feature_names
                if output.adjacency is not None:
                    data["adjacency"] = torch.from_numpy(output.adjacency)
                torch.save(data, path)
            except ImportError as e:
                raise ImportError("PyTorch required for pytorch format") from e

        elif fmt == SerializationFormat.JSON:
            data = {
                "node_features": output.node_features.tolist(),
                "edge_index": output.edge_index.tolist(),
                "node_ids": output.node_ids,
                "feature_names": output.feature_names,
            }
            if output.adjacency is not None:
                data["adjacency"] = output.adjacency.tolist()
            path.write_text(json.dumps(data, indent=2))

        else:
            raise ValueError(f"Unsupported format for GraphAdapter: {fmt}")

        logger.info(f"Serialized graph to {path} as {fmt}")

    def deserialize(
        self,
        path: Path | str,
        fmt: SerializationFormat | str,
    ) -> GraphOutput:
        """Deserialize graph output from disk."""
        path = Path(path)
        fmt = SerializationFormat(fmt) if isinstance(fmt, str) else fmt

        if fmt == SerializationFormat.NUMPY:
            loaded = np.load(path, allow_pickle=True)
            adjacency = loaded.get("adjacency")
            edge_features = loaded.get("edge_features")
            return GraphOutput(
                node_features=loaded["node_features"],
                edge_index=loaded["edge_index"],
                adjacency=adjacency if adjacency is not None else None,
                edge_features=edge_features if edge_features is not None else None,
                node_ids=list(loaded["node_ids"]),
                feature_names=list(loaded["feature_names"]),
            )

        elif fmt == SerializationFormat.PYTORCH:
            try:
                import torch

                data = torch.load(path, weights_only=False)
                adjacency = data.get("adjacency")
                edge_attr = data.get("edge_attr")
                return GraphOutput(
                    node_features=data["x"].numpy(),
                    edge_index=data["edge_index"].numpy(),
                    adjacency=adjacency.numpy() if adjacency is not None else None,
                    edge_features=edge_attr.numpy() if edge_attr is not None else None,
                    node_ids=data["node_ids"],
                    feature_names=data["feature_names"],
                )
            except ImportError as e:
                raise ImportError("PyTorch required for pytorch format") from e

        elif fmt == SerializationFormat.JSON:
            data = json.loads(path.read_text())
            adjacency = data.get("adjacency")
            return GraphOutput(
                node_features=np.array(data["node_features"]),
                edge_index=np.array(data["edge_index"]),
                adjacency=np.array(adjacency) if adjacency else None,
                node_ids=data["node_ids"],
                feature_names=data["feature_names"],
            )

        else:
            raise ValueError(f"Unsupported format for deserialization: {fmt}")
