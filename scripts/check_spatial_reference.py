"""Optional numerical oracle: pip install esda; python scripts/check_spatial_reference.py.

PySAL is a development oracle, not an Shikong runtime dependency. The regular
test suite also checks hand-computed fixtures without this optional installation.
"""

from datetime import datetime

import esda
import libpysal
import numpy as np
import polars as pl

from shikong.core import EventFrame, EventMetadata, EventSchema
from shikong.core.steps.spatial import GetisOrdStep, LocalMoranStep
from shikong.core.weights import SpatialWeights


def check() -> None:
    """Compare the same graph, aggregation and variance conventions."""
    for values, neighbors in [
        ([1.0, 2.0, 4.0, 8.0], {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}),
        (
            [2.0, 3.0, 3.2, 5.0, 8.0, 7.0],
            {0: [1, 3], 1: [0, 3, 4], 2: [4], 3: [0, 1], 4: [1, 2, 5], 5: [4]},
        ),
    ]:
        n = len(values)
        events = EventFrame(
            pl.DataFrame(
                {
                    "t": [datetime(2024, 1, 1)] * n,
                    "x": [0.0] * n,
                    "y": [0.0] * n,
                    "grid_id": range(n),
                    "value": values,
                }
            ).lazy(),
            EventSchema(timestamp_col="t", lon_col="x", lat_col="y"),
            EventMetadata(dataset_name="reference", crs="EPSG:3857"),
        )
        weights = SpatialWeights.from_edges(
            list(range(n)), [(i, j) for i, row in neighbors.items() for j in row]
        )
        actual_i = (
            LocalMoranStep("value", permutations=0, spatial_weights=weights)
            .run(events)
            .collect()["value_local_moran_i"]
            .to_numpy()
        )
        reference_i = esda.Moran_Local(
            np.array(values), libpysal.weights.W(neighbors), permutations=0
        ).Is
        # PySAL divides by n-1. Shikong uses the n convention (spdep mlvar=True).
        np.testing.assert_allclose(actual_i, reference_i * n / (n - 1), rtol=1e-12)
        actual_g = (
            GetisOrdStep("value", spatial_weights=weights)
            .run(events)
            .collect()["value_gi_star"]
            .to_numpy()
        )
        reference_g = esda.G_Local(
            np.array(values),
            libpysal.weights.W(neighbors),
            transform="B",
            star=True,
            permutations=0,
        ).Zs
        np.testing.assert_allclose(actual_g, reference_g, rtol=1e-12)
    print(
        f"Spatial reference parity passed (esda {esda.__version__}, libpysal {libpysal.__version__})"
    )


if __name__ == "__main__":
    check()
