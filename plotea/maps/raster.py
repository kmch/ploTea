"""
Raster drawing on a map: a generic array display and a shaded-relief background.

plotea does no raster IO -- it takes no rioxarray/rasterio dependency. You pass in
a 2D array plus its lon/lat ``extent`` (read and resampled however you like, e.g.
a decimated window of a large DEM), and these draw it on a ``LonLatAxes``: because
that axes assumes lon/lat, ``imshow`` needs no explicit ``transform``.

"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource

from plotea.log import get_logger

_log = get_logger(__name__)


def plot_raster(data, extent, ax=None, origin: str = 'upper', **kwargs):
    """
    Draw a 2D array over its lon/lat ``extent`` on a map axes.

    Parameters
    ----------
    data : array-like
        The 2D array (or RGBA) to display; a masked array shows nodata as transparent.
    extent : sequence of float
        ``(lon_min, lon_max, lat_min, lat_max)`` in degrees -- the array's bounds.
    ax : LonLatAxes, optional
        Map axes to draw on; the current axes if None.
    origin : str
        'upper' (row 0 at the top, the usual raster convention) or 'lower'.
    **kwargs
        Passed to ``imshow`` (e.g. ``cmap``, ``alpha``, ``zorder``, ``vmin``).

    Returns
    -------
    matplotlib.image.AxesImage

    Notes
    -----
    On a ``LonLatAxes`` the lon/lat ``extent`` needs no ``transform``; on a stock
    projected ``GeoAxes`` pass ``transform=ccrs.PlateCarree()``.

    Examples
    --------
    >>> plot_raster(dem, extent=(-10, 35, 35, 72), ax=ax, cmap='gray')

    """
    if ax is None:
        ax = plt.gca()
    return ax.imshow(data, extent=extent, origin=origin, **kwargs)


def hillshade(dem, azdeg: float = 315.0, altdeg: float = 45.0, vert_exag: float = 1.0, dx: float = 1.0, dy: float = 1.0):
    """
    Return shaded-relief intensity in ``[0, 1]`` from an elevation array.

    Parameters
    ----------
    dem : array-like
        Elevation grid. A masked array keeps its mask (nodata -> transparent when
        drawn), and its masked cells are filled flat before shading so the coastline
        gets no false cliff.
    azdeg : float
        Illumination azimuth in degrees (315 = light from the north-west).
    altdeg : float
        Illumination altitude above the horizon, in degrees.
    vert_exag : float
        Vertical exaggeration; raise it to bring out relief in flat terrain.
    dx, dy : float
        Pixel spacing in the elevation's own units (e.g. metres). Passing the true
        spacing keeps the shading physically sensible across resolutions -- with the
        default ``1`` a coarse grid over-saturates, since a cell's rise is compared
        to a one-unit run.

    Returns
    -------
    numpy.ndarray or numpy.ma.MaskedArray
        Intensity in ``[0, 1]``; draw it with ``plot_raster(..., cmap='gray')``.

    Examples
    --------
    >>> shade = hillshade(dem, vert_exag=1.5, dx=2000, dy=2000)   # ~2 km pixels
    >>> plot_raster(shade, extent=ext, ax=ax, cmap='gray', vmin=0, vmax=1, zorder=0.5)

    """
    arr = np.ma.asarray(dem).astype(float)
    mask = np.ma.getmaskarray(arr)
    filled = arr.filled(np.ma.median(arr)) if mask.any() else np.asarray(arr)
    intensity = LightSource(azdeg=azdeg, altdeg=altdeg).hillshade(filled, vert_exag=vert_exag, dx=dx, dy=dy)
    return np.ma.array(intensity, mask=mask)
