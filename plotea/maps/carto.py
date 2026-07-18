"""
The cartopy backend: create map axes and draw the basemap. The only file importing cartopy.

Notes
-----
This increment draws the basemap onto a stock cartopy ``GeoAxes``. The
``LonLatAxes`` subclass that makes bare ``gdf.plot(ax=ax)`` land correctly on any
projection is added in the next increment; swapping the axes class is internal to
``new_axes`` and invisible to ``BaseMap``.

``resolution='50m'`` is deliberate, not cartopy's ``'110m'`` default: the 50m
land/ocean/coastline/border layers are cached locally and render offline, whereas
the 110m ocean and coastline may be absent and would trigger a download.

Natural Earth caching
---------------------
The land, ocean, coastline and border features are all Natural Earth vector data,
which cartopy does not bundle -- it downloads each shapefile the first time it is
needed and reuses it thereafter. The cache lives under
``~/.local/share/cartopy/shapefiles/natural_earth/``, split by theme:
``physical/`` holds ``ne_<res>_land``, ``ne_<res>_ocean`` and ``ne_<res>_coastline``;
``cultural/`` holds the country borders as ``ne_<res>_admin_0_boundary_lines_land``.
So the first ever draw at a given resolution reaches the network; every draw after
that is offline. To warm the cache up front -- e.g. on a fresh machine or in CI,
where nothing is downloaded yet -- run ``cartopy_feature_download physical cultural``,
which fetches all resolutions of both themes in one go.

"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from plotea.log import get_logger
from plotea.maps.basemap_styles import BASEMAP_PLAIN, BasemapStyle

_log = get_logger(__name__)


def new_axes(fig, crs: ccrs.CRS):
    """
    Add and return a ``cartopy`` GeoAxes for ``crs`` on ``fig``.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to add the axes to.
    crs : cartopy.crs.CRS
        The projection the map is drawn in.

    Returns
    -------
    cartopy.mpl.geoaxes.GeoAxes

    Notes
    -----
    Uses ``add_subplot`` rather than ``add_axes`` with a tight rectangle: the
    default subplot margins reserve room around the map so the gridline labels,
    which cartopy draws just outside the map boundary, are not clipped at the
    figure edge -- and without relying on ``bbox_inches='tight'`` at save time.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from plotea.maps.crs import equal_earth
    >>> ax = new_axes(plt.figure(), equal_earth())

    """
    return fig.add_subplot(1, 1, 1, projection=crs)


def draw_basemap(ax, extent=None, style: BasemapStyle = BASEMAP_PLAIN, land: bool = True, ocean: bool = True, coastline: bool = True, borders: bool = True, graticules: bool = True, resolution: str = '50m') -> None:
    """
    Draw land, ocean, coastlines, country borders and graticules onto a map axes.

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxes
        Axes to draw into.
    extent : list or None
        [lon_min, lon_max, lat_min, lat_max] in degrees, or None for the whole world.
    style : BasemapStyle
        Fill colours and line widths. Defaults to ``BASEMAP_PLAIN``.
    land, ocean, coastline, borders, graticules : bool
        Layer toggles. ``borders=True`` is what puts country outlines on the map.
    resolution : str
        Natural Earth resolution: '50m' (default, cached/offline), '110m' or '10m'.

    Notes
    -----
    ``borders`` uses ``cfeature.BORDERS``, which IS Natural Earth's
    ``admin_0_boundary_lines_land`` -- country outlines are built into cartopy, no
    download script needed. The extent is set with ``crs=ccrs.PlateCarree()``
    because the values are degrees; passing the projected axes CRS instead would
    read them as metres and place the view in the wrong spot.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from plotea.maps.crs import equal_earth, europe_laea
    >>> ax = new_axes(plt.figure(), equal_earth())
    >>> draw_basemap(ax)                             # whole world
    >>> ax = new_axes(plt.figure(), europe_laea())
    >>> draw_basemap(ax, extent=[-10, 35, 35, 72])   # Europe

    """
    if extent is None:
        ax.set_global()
    else:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    if land:
        ax.add_feature(cfeature.LAND.with_scale(resolution), facecolor=style.land, zorder=0)
    if ocean:
        ax.add_feature(cfeature.OCEAN.with_scale(resolution), facecolor=style.ocean, zorder=0)
    if coastline:
        ax.add_feature(cfeature.COASTLINE.with_scale(resolution), edgecolor=style.coastline, linewidth=style.coastline_width, zorder=1)
    if borders:
        ax.add_feature(cfeature.BORDERS.with_scale(resolution), edgecolor=style.border, linewidth=style.border_width, zorder=1)
    if graticules:
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=style.graticule_width, color=style.graticule, alpha=0.6, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
