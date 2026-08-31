"""
``BaseMap`` -- the entry point for a map. Start here.

A ``BaseMap`` is a *specification*, not a drawing: constructing one only records
your choices (the view, the projection, the style, which layers to show) and draws
nothing. Cartopy runs only when you call ``plot()``, which creates the axes and
paints the layers. That two-phase split -- resolve, then draw -- is why one
``BaseMap`` can be drawn into several different figures.

How the map subpackage fits together (each lower module knows nothing of the ones
above it)::

    BaseMap (this module)   the spec object + the .plot() entry point
       | builds through
    carto                   the cartopy backend: the axes classes + draw_basemap
       | normalises its inputs with
    crs      vector    styles            registry
    (Crs)    (Bbox)    (BasemapStyle)    (ROIS: named boxes)

The recurring idiom is *resolve at the edge*: every loose argument you pass
(``'eu'``, ``'laea_eu'``, ``[-10, 35, 35, 72]``) is normalised exactly
once, in ``__init__``, by that type's ``from_any`` classmethod -- ``Bbox.from_any``,
``Crs.from_any``, ``BasemapStyle.from_any``. After construction the object holds
only strict, resolved values, so the drawing code downstream never has to reason
about a loose string again.

``BaseMap`` never owns the figure. ``plot(ax=...)`` draws into any axes you hand
it; ``plot()`` otherwise makes its own via ``carto.new_axes``. Either way the axes
is a ``LonLatAxes`` (see ``plotea.maps.carto``), so several maps -- or maps and
plain panels -- can share one figure through GridSpec, ``add_axes`` or insets.
There is deliberately no ``BaseMap.add()``: you overlay your data with the native
``gdf.plot(ax=ax)`` / ``ax.scatter(...)`` calls, which land correctly because the
axes assumes lon/lat.

"""
import matplotlib.pyplot as plt

from plotea.log import get_logger
from plotea.maps import carto
from plotea.maps.styles import BasemapStyle
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
        The view. A key of ``ROIS`` (e.g. 'eu'), a ``[minx, miny, maxx, maxy]``
        box, a geometry, or a ``Bbox``. None (or 'world') is the whole world.
    crs : cartopy CRS, optional
        The map projection. Defaults to Equal Earth; pass any cartopy CRS to
        override (e.g. ``crs=laea_eu()``).
    style : BasemapStyle, optional
        Fill colours and line widths. Defaults to ``BASEMAP_STYLE_DEFAULT``.
    land, ocean, coastline, borders, graticules : bool
        Layer toggles. ``borders=True`` is what makes "the whole world with country
        outlines" a directly assertable default.
    resolution : str
        Natural Earth resolution: '50m' (default, offline), '110m' or '10m'.

    Notes
    -----
    Two phases. Construction resolves each argument through the matching
    ``from_any`` and stores the result, so ``__init__`` is pure bookkeeping and
    draws nothing; ``plot`` is where the axes are created and the layers painted.

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
    >>> fig, ax = plotea.BaseMap(bbox='eu').plot()
    >>> fig, ax = plotea.BaseMap(bbox='eu', crs=plotea.laea_eu()).plot()

    """

    def __init__(self, bbox=None, crs=None, style=None, resolution='50m', \
        land=True, ocean=True, coastline=True, borders=True, graticules=True, graticule_labels=True, graticule_step=None, graticule_inward=False) -> None:
        """
        Build a map specification. Draws nothing until ``plot`` is called.

        Notes
        -----
        Each loose argument is normalised once here and then stored: ``bbox`` via
        ``Bbox.from_any`` (the whole world becomes a ``Bbox.world()`` whose
        ``extent`` is None), ``crs`` via ``Crs.from_any``, and ``style`` via
        ``BasemapStyle.from_any``. The layer toggles and ``resolution`` are kept
        as given. Nothing is drawn.

        Examples
        --------
        >>> bm = BaseMap(bbox='eu', resolution='10m')
        >>> bm = BaseMap(bbox=[-10, 35, 35, 72])

        """
        # The three basic specs
        self.bbox = Bbox.from_any(bbox)
        self.crs = Crs.from_any(crs)
        self.resolution = resolution # of the coastline and 
        self.style = BasemapStyle.from_any(style)
        
        # Derived attributes
        self.extent = self.bbox.extent
        
        # Boolean 
        self.land = land
        self.ocean = ocean
        self.coastline = coastline
        self.borders = borders
        self.graticules = graticules
        self.graticule_labels = graticule_labels
        self.graticule_step = graticule_step
        self.graticule_inward = graticule_inward


    @classmethod
    def from_any(cls, basemap, **defaults):
        """
        Coerce whatever a plotting method was handed into a ``BaseMap``, or into None.

        The counterpart of ``Bbox.from_any`` and ``Crs.from_any`` for the map itself, so
        that ``basemap=`` means one thing everywhere it appears -- on ``Raster``, ``Swath``
        and ``HealpixCells`` alike.

        =============  ==========================================================
        ``basemap``    result
        =============  ==========================================================
        ``False``      ``None`` -- draw the data with no map under it
        ``None``       a ``BaseMap`` built from ``defaults``
        ``True``       the same
        ``dict``       built from ``defaults``, with these entries overriding them
        ``BaseMap``    returned unchanged
        =============  ==========================================================

        ``None`` and ``False`` differ deliberately: ``None`` means "unspecified, use the
        default", which is what a ``plot_map`` wants, and ``False`` means "none at all",
        which is what a bare ``plot`` onto an existing axes wants. A method chooses between
        them by which one it takes as its own parameter default.

        Parameters
        ----------
        basemap : bool or dict or BaseMap or None
        **defaults
            Constructor arguments for the map to build when one is not supplied whole. A
            caller's dict entries win over these.

        Returns
        -------
        BaseMap or None

        Examples
        --------
        >>> BaseMap.from_any(None, bbox='eu')
        >>> BaseMap.from_any({'graticule_step': 5}, bbox='eu')
        >>> BaseMap.from_any(False) is None
        True

        """
        if isinstance(basemap, cls):
            return basemap
        if basemap is False:
            return None
        if isinstance(basemap, dict):
            return cls(**{**defaults, **basemap})
        if basemap is None or basemap is True:
            return cls(**defaults)
        raise TypeError(f'basemap is a BaseMap, a dict of its options, True/None for the '
                        f'default one or False for none; got {type(basemap).__name__}')

    def plot(self, ax=None, fig=None, spec=None, figsize=None):
        """
        Draw the basemap and return ``(fig, ax)``. Creates a figure and axes if ``ax`` is None.

        Parameters
        ----------
        ax : cartopy GeoAxes, optional
            Existing map axes to draw into. Created if None.
        fig : matplotlib.figure.Figure, optional
            Figure to add the axes to (with ``spec``) when ``ax`` is None. Created
            if None.
        spec : matplotlib.gridspec.SubplotSpec, optional
            A GridSpec cell (e.g. ``fig.add_gridspec(1, 2)[0, 1]``) to place the map
            in -- how you put a Europe overview beside a country zoom in one figure.
        figsize : tuple, optional
            Figure size in inches. When None and a new figure is created, a
            projection-aware size is derived from the view's aspect ratio.

        Returns
        -------
        fig, ax

        Notes
        -----
        Two paths. With ``ax`` given, the basemap is drawn into it (this is how
        several maps share one figure); otherwise a ``LonLatAxes`` is created via
        ``carto.new_axes`` -- in ``spec`` if given, else filling a fresh figure.
        Either way the layers are painted by ``carto.draw_basemap``, and the
        resolved ``extent`` -- None for the whole world -- selects ``set_global``
        over ``set_extent``.

        Examples
        --------
        >>> fig, ax = BaseMap().plot()
        >>> fig, ax = BaseMap(bbox='eu').plot()
        >>> fig, ax = BaseMap().plot(figsize=(6, 3))
        >>> f = plt.figure(); gs = f.add_gridspec(1, 2)
        >>> _, ax_left = BaseMap(bbox='eu').plot(fig=f, spec=gs[0, 0])

        """
        if ax is None:
            if fig is None:
                fig = plt.figure(figsize=figsize)
            ax = carto.new_axes(fig, self.crs, spec=spec)
        else:
            fig = ax.figure

        carto.draw_basemap(ax, extent=self.extent, resolution=self.resolution, style=self.style,\
             land=self.land, ocean=self.ocean, coastline=self.coastline, borders=self.borders, graticules=self.graticules, graticule_labels=self.graticule_labels, graticule_step=self.graticule_step, graticule_inward=self.graticule_inward)
        where = 'whole world' if self.extent is None else f'extent {self.extent}'
        # DEBUG: one basemap is plumbing, and a layout draws eight of them per figure.
        _log.debug(f'{where}, {type(ax.projection).__name__}, resolution {self.resolution}')
        return fig, ax
