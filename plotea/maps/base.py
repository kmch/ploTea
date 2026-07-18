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
from plotea.maps.basemap_styles import BASEMAP_PLAIN
from plotea.maps.crs import equal_earth
from plotea.maps.vector import resolve_bbox

_log = get_logger(__name__)

# Default figure width in inches; height follows from the map's aspect ratio, and
# both are capped so a wide world map stays comfortable on screen. Override per
# call with plot(figsize=...).
_FIGURE_WIDTH_IN = 7.0
_FIGURE_MAX_HEIGHT_IN = 6.0


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
    This increment returns a stock cartopy ``GeoAxes``. The ``LonLatAxes`` subclass
    that lets bare ``gdf.plot(ax=ax)`` land correctly on any projection arrives in
    the next increment and does not change this API.

    Examples
    --------
    >>> import plotea
    >>> bm = plotea.BaseMap()
    >>> fig, ax = bm.plot()
    >>> fig, ax = plotea.BaseMap(bbox='europe').plot()
    >>> fig, ax = plotea.BaseMap(bbox='europe', crs=plotea.europe_laea()).plot()

    """

    def __init__(self, bbox=None, crs=None, style=None, land: bool = True, ocean: bool = True, coastline: bool = True, borders: bool = True, graticules: bool = True, resolution: str = '50m') -> None:
        """
        Build a map specification. Draws nothing until ``plot`` is called.

        Examples
        --------
        >>> bm = BaseMap(bbox='europe', resolution='10m')
        >>> bm = BaseMap(bbox=[-10, 35, 35, 72])

        """
        self.bbox = resolve_bbox(bbox)
        self.crs = crs if crs is not None else equal_earth()
        self.style = style if style is not None else BASEMAP_PLAIN
        self.land = land
        self.ocean = ocean
        self.coastline = coastline
        self.borders = borders
        self.graticules = graticules
        self.resolution = resolution

    def axes_crs(self):
        """
        Return the CRS to pass as ``projection=`` to any matplotlib axes factory.

        Returns
        -------
        cartopy.crs.CRS

        Notes
        -----
        The composition path: lay maps into a GridSpec, ``add_axes``,
        ``subplots(subplot_kw=)`` or an inset alongside plain (non-map) panels.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> crs = BaseMap().axes_crs()
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(1, 1, 1, projection=crs)
        >>> BaseMap(bbox='europe').plot(ax=ax)

        """
        return self.crs

    def _figsize(self, bbox):
        """
        Derive a screen-friendly figsize from the drawn box's aspect ratio.

        Parameters
        ----------
        bbox : tuple or None
            The ``(xmin, xmax, ymin, ymax)`` box to be drawn, or ``None`` for the
            whole world.

        Notes
        -----
        A GeoAxes enforces the data aspect, so a fixed square figsize wastes half
        the frame on a wide world map. This measures the aspect from a throwaway
        axes, then sizes the figure to match, capped so it stays comfortable.

        Examples
        --------
        >>> w, h = BaseMap()._figsize(None)

        """
        import cartopy.crs as ccrs

        fig = plt.figure()
        try:
            ax = carto.new_axes(fig, self.crs)
            if bbox is None:
                ax.set_global()
            else:
                ax.set_extent(bbox, crs=ccrs.PlateCarree())
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            aspect = abs(x1 - x0) / abs(y1 - y0)
        finally:
            plt.close(fig)

        width = _FIGURE_WIDTH_IN
        height = width / aspect
        if height > _FIGURE_MAX_HEIGHT_IN:
            height = _FIGURE_MAX_HEIGHT_IN
            width = height * aspect
        return (round(width, 2), round(height, 2))

    def plot(self, ax=None, figsize=None, bbox=None):
        """
        Draw the basemap and return ``(fig, ax)``. Creates a figure and axes if ``ax`` is None.

        Parameters
        ----------
        ax : cartopy GeoAxes, optional
            Existing map axes to draw into. Created if None.
        figsize : tuple, optional
            Figure size in inches. When None and a new figure is created, a
            projection-aware size is derived from the view's aspect ratio.
        bbox : str or list or GeoDataFrame or GeoSeries or Bbox, optional
            Override the view for this draw only -- a name, a
            [minx, miny, maxx, maxy] box, a geometry, or a ``Bbox``. When None, the
            map's own bbox is used. Only the view changes; the projection is not.

        Returns
        -------
        fig, ax

        Examples
        --------
        >>> fig, ax = BaseMap().plot()
        >>> fig, ax = BaseMap().plot(bbox='europe')
        >>> fig, ax = BaseMap().plot(figsize=(6, 3))

        """
        view = self.bbox if bbox is None else resolve_bbox(bbox)
        draw_bbox = None if view is None else view.extent

        if ax is None:
            if figsize is None:
                figsize = self._figsize(draw_bbox)
            fig = plt.figure(figsize=figsize)
            ax = carto.new_axes(fig, self.crs)
        else:
            fig = ax.figure

        carto.draw_basemap(ax, extent=draw_bbox, style=self.style, land=self.land, ocean=self.ocean, coastline=self.coastline, borders=self.borders, graticules=self.graticules, resolution=self.resolution)
        where = 'whole world' if draw_bbox is None else f'extent {draw_bbox}'
        _log.info('%s, %s, resolution %s', where, type(self.crs).__name__, self.resolution)
        return fig, ax
