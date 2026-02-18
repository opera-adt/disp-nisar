#!/usr/bin/env python
"""Create algorithm parameter overrides for DISP-NISAR based on DISP-S1 overrides.

For each NISAR frame, finds the spatially intersecting S1 frames that have
algorithm parameter overrides, then assigns the override from the S1 frame
with the largest intersection area.

Usage:
    python create_algorithm_parameters_overrides.py \
        --s1-overrides /path/to/opera-disp-s1-algorithm-parameters-overrides.json \
        --s1-frames /path/to/frame-geometries-simple-0.16.0.geojson \
        --nisar-frames /path/to/opera-nisar-disp-frames.gpkg \
        --output opera-disp-nisar-algorithm-parameters-overrides.json
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import geopandas as gpd


def load_s1_override_frames(
    s1_overrides_path: Path,
    s1_frames_path: Path,
) -> gpd.GeoDataFrame:
    """Load S1 frames that have overrides and attach the override parameters.

    Parameters
    ----------
    s1_overrides_path : Path
        Path to the DISP-S1 algorithm parameters overrides JSON file.
    s1_frames_path : Path
        Path to the S1 frame geometries GeoJSON file.

    Returns
    -------
    gpd.GeoDataFrame
        S1 frames filtered to those with overrides, with an 'overrides'
        column containing the override dict for each frame.
    """
    with open(s1_overrides_path) as f:
        s1_overrides = json.load(f)

    # Read the S1 frame DB with fid as index (1-based frame IDs)
    s1_gdf = gpd.read_file(
        s1_frames_path, engine="pyogrio", fid_as_index=True
    )
    s1_gdf.index.name = "frame_id"

    # Filter to only frames that have overrides
    override_ids = [int(k) for k in s1_overrides.keys()]
    s1_gdf = s1_gdf.loc[s1_gdf.index.intersection(override_ids)].copy()

    # Attach overrides as a column
    s1_gdf["overrides"] = s1_gdf.index.map(
        lambda fid: s1_overrides[str(fid)]
    )

    click.echo(
        f"Loaded {len(s1_gdf)} S1 frames with overrides "
        f"(out of {len(s1_overrides)} in JSON)."
    )
    return s1_gdf


def map_overrides_to_nisar(
    s1_gdf: gpd.GeoDataFrame,
    nisar_gdf: gpd.GeoDataFrame,
) -> dict[str, dict]:
    """Map S1 overrides to NISAR frames via spatial intersection.

    For each NISAR frame, computes the intersection area with all S1 override
    frames and assigns the override from the S1 frame with the largest overlap.

    Parameters
    ----------
    s1_gdf : gpd.GeoDataFrame
        S1 frames with override parameters (from `load_s1_override_frames`).
        Index is the S1 frame_id.
    nisar_gdf : gpd.GeoDataFrame
        NISAR frame geometries with 'frame_idx' column.

    Returns
    -------
    dict
        Mapping of NISAR frame_idx (as string) to override parameters.
    """
    # Reset S1 index so frame_id becomes a column for sjoin
    s1_reset = s1_gdf.reset_index()
    s1_reset = s1_reset.rename(columns={"frame_id": "s1_frame_id"})

    # Use an equal-area CRS for area calculations
    ea_crs = "EPSG:6933"
    s1_ea = s1_reset.to_crs(ea_crs)
    nisar_ea = nisar_gdf.to_crs(ea_crs)

    # sjoin to find candidate intersections
    joined = gpd.sjoin(nisar_ea, s1_ea, how="inner", predicate="intersects")

    nisar_overrides: dict[str, dict] = {}
    n_matched = 0

    for nisar_idx, group in joined.groupby(joined.index):
        nisar_geom = nisar_ea.loc[nisar_idx, "geometry"]
        best_area = 0.0
        best_override = None

        for _, row in group.iterrows():
            s1_row_idx = row["index_right"]
            s1_geom = s1_ea.loc[s1_row_idx, "geometry"]
            intersection_area = nisar_geom.intersection(s1_geom).area

            if intersection_area > best_area:
                best_area = intersection_area
                best_override = row["overrides"]

        if best_override is not None:
            frame_idx = str(nisar_gdf.loc[nisar_idx, "frame_idx"])
            nisar_overrides[frame_idx] = best_override
            n_matched += 1

    click.echo(
        f"Mapped overrides to {n_matched} NISAR frames "
        f"(out of {len(nisar_gdf)} total)."
    )
    return nisar_overrides


@click.command(context_settings={"show_default": True})
@click.option(
    "--s1-overrides",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the DISP-S1 algorithm parameters overrides JSON.",
)
@click.option(
    "--s1-frames",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the S1 frame geometries GeoJSON.",
)
@click.option(
    "--nisar-frames",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the NISAR DISP frames GeoPackage.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False),
    default="opera-disp-nisar-algorithm-parameters-overrides.json",
    help="Output JSON filename.",
)
def main(s1_overrides: str, s1_frames: str, nisar_frames: str, output: str):
    """Create DISP-NISAR algorithm parameter overrides from DISP-S1.

    Finds spatial intersections between S1 frames (that have overrides) and
    NISAR frames, then assigns the S1 override to each NISAR frame based on
    the largest intersection area.
    """
    # Load S1 override frames
    s1_gdf = load_s1_override_frames(
        s1_overrides_path=Path(s1_overrides),
        s1_frames_path=Path(s1_frames),
    )

    # Load NISAR frames
    nisar_gdf = gpd.read_file(nisar_frames)
    click.echo(f"Loaded {len(nisar_gdf)} NISAR frames.")

    # Map overrides
    nisar_overrides = map_overrides_to_nisar(s1_gdf, nisar_gdf)

    # Adjust half_window: NISAR uses equal x and y, set to max of S1 values
    for frame_idx, params in nisar_overrides.items():
        hw = params.get("phase_linking", {}).get("half_window")
        if hw is not None:
            max_val = max(hw["x"], hw["y"])
            hw["x"] = max_val
            hw["y"] = max_val

    # Sort by frame_idx for readability
    nisar_overrides = dict(
        sorted(nisar_overrides.items(), key=lambda x: int(x[0]))
    )

    # Write output
    with open(output, "w") as f:
        json.dump(nisar_overrides, f, indent=2)
    click.echo(f"Written {len(nisar_overrides)} NISAR overrides to {output}")


if __name__ == "__main__":
    main()
