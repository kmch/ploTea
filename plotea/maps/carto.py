"""
The cartopy backend: create map axes and draw the basemap. The only file importing cartopy.

Notes
-----
A coordinate reference system (CRS) can be *geographic* or *projected*, and the
difference decides what the numbers you plot actually mean.

A **geographic** CRS (e.g. EPSG:4326) stores coordinates as **longitude and latitude 
in degrees**. A point is ``(17.65, 47.79)``.
On the other hand, a **projected** CRS (e.g. Lambert Azimuthal Equal-Area, or LAEA for short) stores coordinates 
as **x/y in metres**. The *same* point becomes ``(572366, -438535)`` -- about 572 km east and 439 km 
south of the LAEA centre at (10 E, 52 N).

Note, the degree ticks you see on a map are a separate graticule overlay drawn on top by ``ax.gridlines`` purely for the reader -- the coordinate *values* a
projected axes works in are still in metres, e.g. ``ax.get_xlim()`` on a projected map returns metres.

Consequence for plotting: a cartopy ``GeoAxes`` treats untransformed data as being
in its own projected (metre) coordinates. So ``ax.scatter(lon, lat)`` with lon/lat
*degrees* and no ``transform`` is read as "x metres, y metres" -- every point
collapses into an invisible speck near the projection origin. No error, no
warning: a silently wrong map. The fix is ``transform=ccrs.PlateCarree()``, which
tells cartopy "these numbers are lon/lat degrees; you reproject them to metres".

Note, ``PlateCarree`` is cartopy's stand-in for raw lon/lat because in PlateCarree
x = lon and y = lat literally.

``LonLatAxes`` flips that default: on it, untransformed data is *assumed* to be
lon/lat, so bare ``ax.scatter(lon, lat)`` and ``gdf.plot(ax=ax)`` land correctly on
any projection without a per-call ``transform=``. See the class for the mechanism
and for why three separate hooks are irreducible. An explicit ``transform=`` always
wins, so the fast idiom ``gdf.to_crs(ax.projection).plot(ax=ax,
transform=ax.projection)`` is unaffected.

Notes
-----
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
import functools

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.ticker as mticker
from cartopy.mpl.feature_artist import FeatureArtist
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.gridliner import Gridliner
from matplotlib.image import AxesImage
from shapely.geometry import box as _box

from plotea.log import get_logger
from plotea.maps.styles import BASEMAP_PLAIN, BasemapStyle

_log = get_logger(__name__)

# The lon/lat CRS that untransformed data is assumed to be in (see LonLatAxes).
LONLAT = ccrs.PlateCarree()

# Artists hook 1 must NOT stamp the lon/lat transform on:
# - FeatureArtist / Gridliner reproject from their own .crs; stamping double-transforms
#   and silently corrupts the basemap.
# - AxesImage from imshow: cartopy's decorated imshow already warps the array into the
#   axes projection and needs transData; stamping PlateCarree instead reads the projected
#   metres as degrees and the image lands off-map (invisible). imshow is handled by hook 2.
_SKIP = (FeatureArtist, Gridliner, AxesImage)

# The GeoAxes methods cartopy wraps to default transform -> self.projection (metres).
# Verified against cartopy 0.25.0; a change here is silent misplacement, so the
# canary test in tests/test_transform.py pins this exact set.
_TRANSFORM_METHODS = ('imshow', 'contour', 'contourf', 'scatter', 'annotate', 'hexbin', 'pcolormesh', 'pcolor', 'quiver', 'barbs', 'streamplot')


class MapAxes(GeoAxes):
    """
    A cartopy GeoAxes that keeps a map's aspect fixed at 'equal'.

    Notes
    -----
    A projected map must render with equal x/y scaling or the projection is
    distorted. GeoPandas' ``plot`` defaults to ``aspect='auto'``, which for
    lon/lat data calls ``set_aspect(1 / cos(lat))`` -- roughly 1.7 at European
    latitudes -- and that silently squashes a map drawn on the axes. This subclass
    ignores such external attempts and holds the aspect at 'equal', so a bare
    ``gdf.plot(ax=ax)`` no longer needs ``aspect=None``. Verified: cartopy itself
    does not set a numeric aspect via ``set_extent``, so forcing 'equal' never
    fights the backend.

    Examples
    --------
    >>> ax = new_axes(plt.figure(), equal_earth())
    >>> type(ax).__name__
    'MapAxes'

    """

    def set_aspect(self, aspect, *args, **kwargs):
        """
        Force 'equal', ignoring the requested aspect so overlays cannot squash the map.

        Examples
        --------
        >>> ax.set_aspect(1.7)      # no-op; the map stays 'equal'

        """
        return super().set_aspect('equal', *args, **kwargs)


class LonLatAxes(MapAxes):
    """
    A map axes on which untransformed data is assumed to be lon/lat degrees.

    On a stock cartopy ``GeoAxes`` a plotting call with no ``transform`` is read in
    the axes' own projected (metre) coordinates, so ``ax.scatter(lon, lat)`` with
    degrees is silently misplaced. ``LonLatAxes`` flips that default to
    ``PlateCarree`` (raw lon/lat), so ``ax.scatter(lon, lat)`` and bare
    ``gdf.plot(ax=ax)`` land correctly on any projection. See the module docstring
    for the degrees-vs-metres explanation.

    Notes
    -----
    Three disjoint hooks are needed because there is no single choke point:

    1. ``_set_artist_props`` -- the collection/patch/line path (geopandas lines and
       polygons via ``add_collection``, a ``Rectangle`` via ``add_patch``). These
       artists are not decorated by cartopy; matplotlib would stamp ``transData``
       (metres), so we stamp lon/lat instead when no transform is set.
    2. cartopy's decorated methods (``_TRANSFORM_METHODS``) -- points via
       ``scatter`` and rasters via ``pcolormesh`` reach the axes already
       transform-set, so hook 1 is blind to them; we inject the default before
       cartopy reads it.
    3. ``text`` -- matplotlib's ``text`` sets ``transform=transData`` *explicitly*,
       so ``is_transform_set()`` is already True and hook 1 cannot see it.

    ``_SKIP`` protects cartopy's own ``FeatureArtist`` and ``Gridliner`` (the
    basemap and graticule), which reproject from their own CRS internally.
    An explicit ``transform=`` always wins; set the class attribute
    ``data_crs = None`` (via a subclass) to fall back to stock cartopy behaviour.

    Examples
    --------
    >>> ax = new_axes(plt.figure(), laea_eu())
    >>> type(ax).__name__
    'LonLatAxes'
    >>> _ = ax.scatter([17.65], [47.79])            # lon/lat, no transform needed

    """

    data_crs = LONLAT

    def _set_artist_props(self, a):
        """
        Stamp the lon/lat transform on an untransformed non-cartopy artist (hook 1).

        Examples
        --------
        >>> # called by add_collection / add_patch / add_line, not directly

        """
        if self.data_crs is not None and not isinstance(a, _SKIP) and not a.is_transform_set():
            a.set_transform(self.data_crs)
        super()._set_artist_props(a)

    def text(self, *args, **kwargs):
        """
        Default ``text`` to lon/lat coordinates (hook 3).

        Examples
        --------
        >>> _ = ax.text(17.65, 47.79, 'Bratislava')     # placed by lon/lat

        """
        if self.data_crs is not None and 'transform' not in kwargs:
            kwargs['transform'] = self.data_crs
        return super().text(*args, **kwargs)


def _make_lonlat_method(name):
    """
    Build a ``LonLatAxes`` override of a cartopy-decorated method that defaults its transform to lon/lat (hook 2).

    Examples
    --------
    >>> LonLatAxes.scatter = _make_lonlat_method('scatter')

    """
    parent = getattr(GeoAxes, name)

    @functools.wraps(parent)
    def method(self, *args, **kwargs):
        if self.data_crs is not None and kwargs.get('transform', None) is None:
            kwargs['transform'] = self.data_crs
        return parent(self, *args, **kwargs)

    return method


for _name in _TRANSFORM_METHODS:
    setattr(LonLatAxes, _name, _make_lonlat_method(_name))


class _MapProjection:
    """
    Adapter so ``add_subplot(projection=...)`` builds a ``LonLatAxes`` for a given CRS.

    Notes
    -----
    matplotlib calls ``_as_mpl_axes`` on any non-string projection to learn which
    axes class and kwargs to use; this returns ``LonLatAxes`` instead of the stock
    ``GeoAxes`` cartopy's CRS would give.

    Examples
    --------
    >>> ax = plt.figure().add_subplot(projection=_MapProjection(equal_earth()))

    """

    def __init__(self, crs: ccrs.CRS) -> None:
        self.crs = crs

    def _as_mpl_axes(self):
        """
        Return ``(LonLatAxes, kwargs)`` for matplotlib's projection machinery.

        Examples
        --------
        >>> _MapProjection(equal_earth())._as_mpl_axes()[0].__name__
        'LonLatAxes'

        """
        return LonLatAxes, {'projection': self.crs}


def new_axes(fig, crs: ccrs.CRS, spec=None, rect=None):
    """
    Add and return a ``LonLatAxes`` for ``crs`` on ``fig``: a subplot, a GridSpec cell, or an explicit rectangle.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to add the axes to.
    crs : cartopy.crs.CRS
        The projection the map is drawn in.
    spec : matplotlib.gridspec.SubplotSpec, optional
        A GridSpec cell (e.g. ``fig.add_gridspec(1, 2)[0, 1]``) to place the axes
        in. When None (and ``rect`` is None) the axes fills the figure as a subplot.
    rect : sequence of float, optional
        An explicit ``[left, bottom, width, height]`` in figure fractions. Used by
        the mosaic to place equal-aspect panels exactly, so they align without the
        drift a GridSpec cell allows. Takes precedence over ``spec``.

    Returns
    -------
    LonLatAxes
        A GeoAxes subclass whose aspect is locked to 'equal' and on which
        untransformed data is assumed to be lon/lat.

    Notes
    -----
    Uses ``add_subplot`` rather than ``add_axes`` with a tight rectangle: the
    default subplot margins reserve room around the map so the gridline labels,
    which cartopy draws just outside the map boundary, are not clipped at the
    figure edge -- and without relying on ``bbox_inches='tight'`` at save time.
    Passing ``spec`` is what lets several maps -- or maps and plain panels -- share
    one figure, e.g. a Europe overview beside a country zoom.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from plotea.maps.crs import equal_earth, laea_eu
    >>> ax = new_axes(plt.figure(), equal_earth())
    >>> fig = plt.figure(); gs = fig.add_gridspec(1, 2)
    >>> ax_left = new_axes(fig, equal_earth(), spec=gs[0, 0])
    >>> ax_right = new_axes(fig, laea_eu(), spec=gs[0, 1])

    """
    proj = _MapProjection(crs)
    if rect is not None:
        return fig.add_axes(rect, projection=proj)
    if spec is None:
        return fig.add_subplot(1, 1, 1, projection=proj)
    return fig.add_subplot(spec, projection=proj)


def draw_basemap(ax, extent=None, style: BasemapStyle = BASEMAP_PLAIN, land: bool = True, ocean: bool = True, coastline: bool = True, borders: bool = True, graticules: bool = True, graticule_labels: bool = True, graticule_step=None, graticule_inward: bool = False, resolution: str = '50m') -> None:
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
    >>> from plotea.maps.crs import equal_earth, laea_eu
    >>> ax = new_axes(plt.figure(), equal_earth())
    >>> draw_basemap(ax)                             # whole world
    >>> ax = new_axes(plt.figure(), laea_eu())
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
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=graticule_labels, linewidth=style.graticule_width, color=style.graticule, alpha=0.6, linestyle='--')
        if graticule_step is not None:
            gl.xlocator = mticker.MultipleLocator(graticule_step)
            gl.ylocator = mticker.MultipleLocator(graticule_step)
        if graticule_labels:
            gl.top_labels = False
            gl.right_labels = False
            if graticule_inward:
                # Negative padding pulls the labels inside the frame, so on abutting
                # mosaic panels they do not stick out and collide with the neighbour.
                gl.xpadding = -12
                gl.ypadding = -12
                gl.xlabel_style = {'va': 'top', 'color': style.graticule}
                gl.ylabel_style = {'ha': 'left', 'color': style.graticule}


def countries_shapefile(resolution: str = '50m') -> str:
    """
    Return the path to the Natural Earth ``admin_0_countries`` shapefile (country polygons).

    Fetched and cached by cartopy exactly like the basemap layers, so the first
    call at a given resolution reaches the network and every call after is offline.
    This is a *different* file from the border lines ``draw_basemap`` uses
    (``admin_0_boundary_lines_land``): those are outlines, this carries the filled
    country polygons and their ISO attributes.

    Parameters
    ----------
    resolution : str
        Natural Earth resolution: '50m' (default), '110m' or '10m'.

    Returns
    -------
    str
        Filesystem path to the ``.shp``; read it with ``geopandas.read_file``.

    Examples
    --------
    >>> import geopandas as gpd
    >>> gdf = gpd.read_file(countries_shapefile('50m'))

    """
    return shpreader.natural_earth(resolution=resolution, category='cultural', name='admin_0_countries')


def projected_aspect(extent, crs: ccrs.CRS) -> float:
    """
    Return the width/height ratio of a lon/lat ``extent`` once projected into ``crs``.

    This is what a mosaic needs to size panels: a map axes is locked to equal
    scaling, so its on-screen height is set by this ratio, not by the cell it sits
    in. The lon/lat box is projected with ``project_geometry`` -- the exact call
    cartopy's ``set_extent`` uses -- so the aspect matches the rendered axes to
    machine precision and equal-aspect panels tile without shrinking out of line.

    Parameters
    ----------
    extent : sequence of float
        ``(lon_min, lon_max, lat_min, lat_max)`` in degrees (the ``Bbox.extent`` order).
    crs : cartopy.crs.CRS
        The projection the map is drawn in.

    Returns
    -------
    float
        Projected width divided by projected height.

    Examples
    --------
    >>> from plotea.maps.crs import laea_eu
    >>> projected_aspect((-4.762, 9.556, 41.384, 51.097), laea_eu())   # France, ~1.0

    """
    lon0, lon1, lat0, lat1 = extent
    minx, miny, maxx, maxy = crs.project_geometry(_box(lon0, lat0, lon1, lat1), ccrs.PlateCarree()).bounds
    return float((maxx - minx) / (maxy - miny))


def visible_bbox(ax):
    """
    Return the lon/lat ``Bbox`` actually shown in a projected map axes.

    A map panel is a rectangle in *projected* (metre) space, which back-projects to a
    *curved* lon/lat region reaching well beyond the lon/lat view box -- e.g. an
    Equal-Earth/LAEA Europe panel shows land far east and, in its corners, Iceland
    and Greenland. Reading rivers, a DEM, etc. to this box (instead of the view box)
    lets them fill the panel; the axes then clips them exactly at the frame, so
    nothing looks cut off mid-map.

    Parameters
    ----------
    ax : cartopy GeoAxes
        A map axes whose extent has been set (its projected x/y limits are read).

    Returns
    -------
    Bbox
        The lon/lat bounding box of the visible rectangle (EPSG:4326).

    Notes
    -----
    Computed with ``project_geometry`` (the same densified projection ``set_extent``
    uses), so the box tightly bounds the true curved footprint.

    Examples
    --------
    >>> HydroRivers(path, bbox=visible_bbox(ax)).plot(ax=ax)   # rivers fill the panel
    >>> dem, dx, dy = read_dem(src, visible_bbox(ax).extent)   # DEM covers it, no white corners

    """
    from plotea.maps.vector import Bbox
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    lonmin, latmin, lonmax, latmax = ccrs.PlateCarree().project_geometry(_box(x0, y0, x1, y1), ax.projection).bounds
    return Bbox([lonmin, latmin, lonmax, latmax], target_crs=4326)
