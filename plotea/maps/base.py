"""
``BaseMap`` -- a basemap specification. ``BaseMap()`` is the whole world with country outlines.

Notes
-----
``BaseMap`` never owns the figure. ``plot(ax=...)`` draws into any axes, and
``axes_crs()`` hands the CRS to any matplotlib axes factory (GridSpec,
``add_axes``, ``subplots(subplot_kw=)``, insets) -- so several maps, or maps and
plain panels, can share one figure.

There is deliberately no ``BaseMap.add()``: you chain with ``gdf.plot(ax=ax)``.

"""
import matplotlib.pyplot as plt

from plotea.log import get_logger
from plotea.maps import carto
from plotea.maps.basemap_styles import BasemapStyle
from plotea.maps.crs import Crs
from plotea.maps.vector import Bbox

_log = get_logger(__name__)

DEFAULT_FIGSIZE = (8, 5)


class BaseMap:
    """
    A basemap specification. ``BaseMap()`` is the whole world with country outlines in Equal Earth.

    Parameters
    ----------
    bbox : str or list or GeoDataFrame or GeoSeries or Bbox, optional
        The view. A key of ``ROIS`` (e.g. 'europe'), a ``[minx, miny, maxx, maxy]``
        box, a geometry, or a ``Bbox``. None (or 'world') is the whole world.
    crs : cartopy CRS, optional
        The map projection. Defaults to Equal Earth; pass any cartopy CRS to
        override (e.g. ``crs=europe_laea()``).
    style : BasemapStyle, optional
        Fill colours and line widths. Defaults to ``BASEMAP_PLAIN``.
    land, ocean, coastline, borders, graticules : bool
        Layer toggles. ``borders=True`` is what makes "the whole world with country
        outlines" a directly assertable default.
    resolution : str
        Natural Earth resolution: '50m' (default, offline), '110m' or '10m'.

    Notes
    -----
    ``plot`` returns a ``LonLatAxes`` (a cartopy ``GeoAxes`` subclass) on which
    untransformed data is assumed to be lon/lat degrees, so bare
    ``ax.scatter(lon, lat)`` and ``gdf.plot(ax=ax)`` land correctly on any
    projection without a per-call ``transform=``. An explicit ``transform=`` always
    wins. See ``plotea.maps.carto`` for the degrees-vs-metres explanation.

    Examples
    --------
    >>> import plotea
    >>> bm = plotea.BaseMap()
    >>> fig, ax = bm.plot()
    >>> fig, ax = plotea.BaseMap(bbox='europe').plot()
    >>> fig, ax = plotea.BaseMap(bbox='europe', crs=plotea.europe_laea()).plot()

    """

    def __init__(self, bbox=None, crs=None, style=None, resolution='50m', \
        land=True, ocean=True, coastline=True, borders=True, graticules=True) -> None:
        """
        Build a map specification. Draws nothing until ``plot`` is called.

        Examples
        --------
        >>> bm = BaseMap(bbox='europe', resolution='10m')
        >>> bm = BaseMap(bbox=[-10, 35, 35, 72])

        """
        self.bbox = Bbox.from_any(bbox)
        self.extent = self.bbox.extent
        self.crs = Crs.from_any(crs)
        self.style = BasemapStyle.from_any(style)
        self.land = land
        self.ocean = ocean
        self.coastline = coastline
        self.borders = borders
        self.graticules = graticules
        self.resolution = resolution

    def plot(self, ax=None, figsize=None):
        """
        Draw the basemap and return ``(fig, ax)``. Creates a figure and axes if ``ax`` is None.

        Parameters
        ----------
        ax : cartopy GeoAxes, optional
            Existing map axes to draw into. Created if None.
        figsize : tuple, optional
            Figure size in inches. When None and a new figure is created, a
            projection-aware size is derived from the view's aspect ratio.

        Returns
        -------
        fig, ax

        Examples
        --------
        >>> fig, ax = BaseMap().plot()
        >>> fig, ax = BaseMap().plot(bbox='europe')
        >>> fig, ax = BaseMap().plot(figsize=(6, 3))

        """
        if ax is None:
            # figsize = DEFAULT_FIGSIZE if figsize is None else figsize
            fig = plt.figure(figsize=figsize)
            ax = carto.new_axes(fig, self.crs)
        else:
            fig = ax.figure

        carto.draw_basemap(ax, extent=self.extent, resolution=self.resolution, style=self.style,\
             land=self.land, ocean=self.ocean, coastline=self.coastline, borders=self.borders, graticules=self.graticules)
        where = 'whole world' if self.extent is None else f'extent {self.extent}'
        _log.info('%s, %s, resolution %s', where, type(self.crs).__name__, self.resolution)
        return fig, ax
