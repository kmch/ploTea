"""
Read a decimated window of a (possibly huge) DEM for a single map panel.

rasterio-based, and deliberately *not* part of the installed ``plotea`` package --
plotea itself takes no raster-IO dependency. Keep it as a standalone helper: the
caller passes any DEM path and a lon/lat extent, and gets back a small,
resolution-appropriate array ready for ``plotea.hillshade`` / ``plotea.plot_raster``.

The point is per-panel resampling: read a wide view from a coarse copy and a zoom
from native tiles, both decimated to the same on-screen pixel budget, so no
oversized array is ever materialised.

Example
-------
    from dem_window import read_dem
    dem, dx, dy = read_dem('/data/merit_europe_coarse.tif', (-10, 35, 35, 72), max_px=1600)
    shade = plotea.hillshade(dem, dx=dx, dy=dy)
    plotea.plot_raster(shade, extent=(-10, 35, 35, 72), ax=ax, cmap='gray')

"""
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds


def read_dem(path, extent, max_px: int = 1600):
    """
    Read ``path`` over ``extent`` (lon0, lon1, lat0, lat1), decimated to ~``max_px`` across.

    Parameters
    ----------
    path : str or Path
        A DEM readable by rasterio (GeoTIFF, VRT, ...), assumed lon/lat (EPSG:4326).
    extent : sequence of float
        ``(lon_min, lon_max, lat_min, lat_max)`` in degrees.
    max_px : int
        Target size of the longer output side; the read is decimated to it.

    Returns
    -------
    dem : numpy.ma.MaskedArray
        Elevation over the window, nodata masked.
    dx, dy : float
        Pixel spacing in metres (dx uses the mid-latitude cosine), for ``hillshade``.

    """
    lon0, lon1, lat0, lat1 = extent
    with rasterio.open(path) as ds:
        win = from_bounds(lon0, lat0, lon1, lat1, ds.transform)
        scale = min(1.0, max_px / max(win.width, win.height))
        out = (max(1, round(win.height * scale)), max(1, round(win.width * scale)))
        dem = ds.read(1, window=win, out_shape=out, resampling=Resampling.average, masked=True, boundless=True)
    mlat = 0.5 * (lat0 + lat1)
    dx = (lon1 - lon0) / out[1] * 111320.0 * np.cos(np.radians(mlat))
    dy = (lat1 - lat0) / out[0] * 110540.0
    return dem, dx, dy
