from __future__ import annotations

from pathlib import Path

import click
import geopandas as gpd
import h5py
from dolphin import Bbox
from pyproj import Transformer
from shapely.geometry import box


def _load_gpkg(gpkg_path: Path) -> gpd.GeoDataFrame:
    """Load the NISAR frames GeoPackage indexed by frame_idx."""
    gdf = gpd.read_file(gpkg_path)
    gdf = gdf.set_index("frame_idx")
    return gdf


def _get_nisar_bbox(file_path: Path) -> Bbox:
    """Read the bounding box from a NISAR GSLC .h5 file and return as lon/lat."""
    with h5py.File(file_path, "r") as f:
        grid_path = "/science/LSAR/GSLC/grids/frequencyA"
        x_coords = f[f"{grid_path}/xCoordinates"][:]
        y_coords = f[f"{grid_path}/yCoordinates"][:]

        epsg_code = f[f"{grid_path}/projection"][()]
        if isinstance(epsg_code, bytes):
            epsg_code = int(epsg_code.decode())
        else:
            epsg_code = int(epsg_code)

    transformer = Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)
    lon_min, lat_min = transformer.transform(x_coords.min(), y_coords.min())
    lon_max, lat_max = transformer.transform(x_coords.max(), y_coords.max())
    return Bbox(lon_min, lat_min, lon_max, lat_max)


def _find_intersecting_frames(
    gdf: gpd.GeoDataFrame, query_bbox: Bbox
) -> gpd.GeoDataFrame:
    """Return GeoDataFrame of frames whose geometry intersects the query bbox."""
    query_geom = box(
        query_bbox.left, query_bbox.bottom, query_bbox.right, query_bbox.top
    )
    return gdf[gdf.geometry.intersects(query_geom)]


def _print_bbox_results(query_bbox: Bbox, gdf_match: gpd.GeoDataFrame) -> None:
    """Print the table of intersecting frames."""
    click.echo(
        "Searching for frames intersecting bbox:"
        f" {query_bbox.left:.6f} {query_bbox.bottom:.6f}"
        f" {query_bbox.right:.6f} {query_bbox.top:.6f}"
    )
    click.echo(f"{'Frame':>8s}  {'EPSG':>6s}")
    click.echo("-" * 16)
    for idx in sorted(gdf_match.index):
        epsg = int(gdf_match.loc[idx, "epsg"])
        click.echo(f"{idx:>8d}  {epsg:>6d}")


def _get_frame_latlon_bounds(row) -> Bbox:
    """Get the lat/lon bounding box from a gpkg row's geometry."""
    minx, miny, maxx, maxy = row.geometry.bounds
    return Bbox(minx, miny, maxx, maxy)


def _plot_frames(
    gdf_match: gpd.GeoDataFrame,
    query_bbox: Bbox | None = None,
) -> None:
    """Plot actual frame polygons on a map with annotations."""
    import cartopy.crs as ccrs
    import cartopy.io.shapereader as shpreader
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from shapely.ops import unary_union

    geometry_union = unary_union(gdf_match.geometry.values)
    minx, miny, maxx, maxy = geometry_union.bounds

    if query_bbox is not None:
        minx = min(minx, query_bbox.left)
        miny = min(miny, query_bbox.bottom)
        maxx = max(maxx, query_bbox.right)
        maxy = max(maxy, query_bbox.top)

    pad = 1.5
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(
        [minx - pad, maxx + pad, miny - pad, maxy + pad],
        crs=ccrs.PlateCarree(),
    )

    # Background: states, coastlines
    states_shp = shpreader.natural_earth(
        resolution="10m", category="cultural", name="admin_1_states_provinces_lakes"
    )
    ax.add_geometries(
        shpreader.Reader(states_shp).geometries(),
        ccrs.PlateCarree(),
        facecolor="lightgray",
        edgecolor="white",
        linewidth=1,
        zorder=1,
    )
    gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.4)
    gl.top_labels = gl.right_labels = False

    # Colors by pass direction
    colors = {"Ascending": "blue", "Descending": "green"}
    default_color = "gray"

    # Plot each frame polygon
    for idx, row in gdf_match.iterrows():
        geom = row.geometry
        direction = row.get("passDirection", "")
        color = colors.get(direction, default_color)

        polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for poly in polys:
            xs, ys = poly.exterior.xy
            ax.fill(
                xs,
                ys,
                alpha=0.15,
                facecolor=color,
                edgecolor=color,
                linewidth=1.5,
                transform=ccrs.PlateCarree(),
                zorder=3,
            )

        centroid = geom.centroid
        ax.text(
            centroid.x,
            centroid.y,
            str(idx),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="white",
            bbox={"boxstyle": "round,pad=0.2", "fc": color, "ec": "none", "alpha": 0.8},
            transform=ccrs.PlateCarree(),
            zorder=4,
        )

    # Legend for pass directions
    present_dirs = (
        set(gdf_match["passDirection"].values)
        if "passDirection" in gdf_match.columns
        else set()
    )
    legend_handles = [
        mpatches.Patch(facecolor=c, edgecolor=c, alpha=0.4, label=d)
        for d, c in colors.items()
        if d in present_dirs
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

    # Plot query bbox if given
    if query_bbox is not None:
        w, s, e, n = (
            query_bbox.left,
            query_bbox.bottom,
            query_bbox.right,
            query_bbox.top,
        )
        rect = mpatches.Rectangle(
            (w, s),
            e - w,
            n - s,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            linestyle="--",
            transform=ccrs.PlateCarree(),
            zorder=5,
        )
        ax.add_patch(rect)
        ax.text(
            (w + e) / 2,
            n,
            "query",
            fontsize=9,
            color="red",
            ha="center",
            va="bottom",
            transform=ccrs.PlateCarree(),
            zorder=5,
        )

    ax.set_title(f"NISAR Frames ({len(gdf_match)} found)")
    plt.tight_layout()
    plt.show()


@click.command(name="search")
@click.option(
    "-g",
    "--gpkg",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the NISAR frames GeoPackage file.",
)
@click.option(
    "-b",
    "--bbox",
    type=float,
    nargs=4,
    default=None,
    help="Bounding box as W S E N in decimal degrees (lon/lat).",
)
@click.option(
    "-n",
    "--frame-number",
    type=int,
    default=None,
    help="Frame number to look up.",
)
@click.option(
    "-i",
    "--input-h5",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="NISAR GSLC .h5 file to read bounding box from.",
)
@click.option(
    "--plot",
    is_flag=True,
    default=False,
    help="Plot the matching frame polygons on a map.",
)
def search(
    gpkg: Path,
    bbox: tuple[float, ...] | None,
    frame_number: int | None,
    input_h5: Path | None,
    plot: bool,
) -> None:
    """Search NISAR frames by bounding box, frame number, or .h5 file.

    If --bbox is given, prints frame numbers and EPSG codes for intersecting frames.
    If --frame-number is given, prints the EPSG code and bounding box in lat/lon.
    If --input-h5 is given, reads the bbox from the file and finds intersecting frames.
    Add --plot to visualize the frame polygons on a map.
    """
    provided = sum(x is not None for x in [bbox, frame_number, input_h5])
    if provided == 0:
        raise click.UsageError("Must provide --bbox, --frame-number, or --input-h5.")
    if provided > 1:
        raise click.UsageError(
            "Provide only one of --bbox, --frame-number, or --input-h5."
        )

    gdf = _load_gpkg(gpkg)

    if input_h5 is not None:
        query_bbox = _get_nisar_bbox(input_h5)
        click.echo(f"Read bbox from: {input_h5}")
        gdf_match = _find_intersecting_frames(gdf, query_bbox)
        _print_bbox_results(query_bbox, gdf_match)
        if plot:
            _plot_frames(gdf_match, query_bbox)
    elif bbox is not None:
        query_bbox = Bbox(*bbox)
        gdf_match = _find_intersecting_frames(gdf, query_bbox)
        _print_bbox_results(query_bbox, gdf_match)
        if plot:
            _plot_frames(gdf_match, query_bbox)
    else:
        if frame_number not in gdf.index:
            raise click.UsageError(f"Frame {frame_number} not found in {gpkg}")
        row = gdf.loc[frame_number]
        epsg = int(row["epsg"])
        latlon_bounds = _get_frame_latlon_bounds(row)
        click.echo(f"Frame:  {frame_number}")
        click.echo(f"EPSG:   {epsg}")
        click.echo(
            f"Bounds (WSEN): {latlon_bounds.left:.6f} {latlon_bounds.bottom:.6f}"
            f" {latlon_bounds.right:.6f} {latlon_bounds.top:.6f}"
        )
        if plot:
            _plot_frames(gdf.loc[[frame_number]])
