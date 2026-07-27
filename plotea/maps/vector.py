"""
Vector data and bounding boxes: the ``Vector`` wrapper, the ``Bbox`` view, and the name resolver.

Notes
-----
A ``Bbox`` is a padded bounding box derived from geometry; in a map it is what
sets the axes extent (its ``extent`` property is the tuple cartopy's
``set_extent`` wants, or None for the whole world). ``Bbox.from_any`` turns a name
(a key of ``ROIS``), a raw box, a geometry, an existing ``Bbox``, or None/'world'
into a ``Bbox`` -- the whole world being a ``Bbox.world()`` in the unbounded state,
not a ``None``.

"""
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import box as shapely_box

from plotea.log import get_logger
from plotea.maps.registry import ROIS

_log = get_logger(__name__)


class Vector:
    """
    A vector dataset that is neither necessarily a file nor a GeoDataFrame.

    Can be initialised with a file path, a GeoDataFrame, both, or neither.
    Data is loaded from disk lazily -- only when first accessed via ``.data``.

    Parameters
    ----------
    path : str or Path, optional
    data : geopandas.GeoDataFrame, optional
    """

    def __init__(self, path=None, data=None):
        self.path = Path(path) if path is not None else None
        self._data = data

    def __repr__(self):
        return f"Vector(path={self.path}, data={'loaded' if self._data is not None else 'not loaded'})"

    def info(self):
        print(f"path    : {self.path}")
        print(f"has data: {self._data is not None}")
        if self.data is not None:
            print(f"rows    : {len(self.data)}")
            print(f"crs     : {self.crs}")
            print(f"columns : {list(self.data.columns)}")
            print(f"bounds  : {self.data.total_bounds}")

    @property
    def data(self) -> gpd.GeoDataFrame | None:
        if self._data is None and self.path is not None:
            self._data = gpd.read_file(self.path)
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def crs(self):
        return self.data.crs if self.data is not None else None

    def clip(self, mask_geometry, keep_geom_type=True) -> 'Vector':
        """Clip to *mask_geometry*. Returns a new Vector; self is not modified."""
        if self.data is None:
            raise ValueError("No data to clip.")
        clipped = gpd.clip(self.data, mask_geometry, keep_geom_type=keep_geom_type).reset_index(drop=True)
        return Vector(data=clipped)

    def to_crs(self, crs) -> 'Vector':
        """Return a new Vector reprojected to *crs*."""
        if self.data is None:
            raise ValueError("No data to reproject.")
        return Vector(data=self.data.to_crs(crs))

    def plot(self, ax=None, **kwargs):
        """Plot the vector geometry. Drawing is delegated to plot.plot_vector."""
        if self.data is None:
            raise ValueError("No data to plot.")
        from .plot import plot_vector
        return plot_vector(self.data, ax=ax, **kwargs)

    def save(self, out_file) -> Path:
        """Write to *out_file* (format inferred from extension). Returns the path."""
        if self.data is None:
            raise ValueError("No data to save.")
        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        self.data.to_file(out_file)
        _log.info(f"Saved vector to {out_file}")
        return out_file


class Bbox:
    """
    A padded bounding box derived from a polygon geometry or raw bounds.

    Parameters
    ----------
    geometry : GeoDataFrame or GeoSeries or list
        Input geometry, or a ``[minx, miny, maxx, maxy]`` bounds sequence, to
        derive the box from.
    target_crs : str or int or CRS, optional
        Reproject the geometry to this CRS before computing bounds. When None, the
        geometry's own CRS is kept (or, for a raw bounds sequence, left unset).
    pad : float
        Fractional padding added to each side (0.1 = 10% of width/height).

    Examples
    --------
    >>> bbox = Bbox([-10, 35, 35, 72], target_crs=4326, pad=0.1)
    >>> bbox.extent          # (xmin, xmax, ymin, ymax) in lon/lat degrees
    >>> bbox.plot()

    """

    def __init__(self, geometry=None, target_crs=None, pad: float = 0.0) -> None:
        """
        Build the box, or leave it empty when ``geometry`` is None.

        Examples
        --------
        >>> bbox = Bbox([-10, 35, 35, 72], target_crs=4326)

        """
        self.bbox = None
        self.is_world = False
        if geometry is not None:
            self._build(geometry, target_crs=target_crs, pad=pad)

    @classmethod
    def from_any(cls, bbox) -> 'Bbox':
        """
        Coerce a name, a raw box, a geometry, an existing ``Bbox`` or None/'world' into a ``Bbox``.

        This is the single entry point ``BaseMap`` uses to normalise its ``bbox``
        argument; it always returns a ``Bbox`` (the whole world is a ``Bbox`` in the
        unbounded state, not ``None``).

        Parameters
        ----------
        bbox : str or list or GeoDataFrame or GeoSeries or Bbox or None
            A key of ``ROIS``; the name 'world' or None for the whole world; a
            ``[minx, miny, maxx, maxy]`` box; a geometry; or an existing ``Bbox``
            (returned unchanged).

        Returns
        -------
        Bbox

        Examples
        --------
        >>> Bbox.from_any('europe').extent
        (-10.0, 35.0, 35.0, 72.0)
        >>> Bbox.from_any(None).is_world
        True
        >>> Bbox.from_any('world').is_world
        True

        """
        if bbox is None:
            return cls.world()
        if isinstance(bbox, Bbox):
            return bbox
        if isinstance(bbox, str):
            return cls.world() if bbox == 'world' else cls.from_name(bbox)
        return cls(bbox)

    @classmethod
    def world(cls) -> 'Bbox':
        """
        The whole-world view: an unbounded ``Bbox`` whose ``extent`` is None.

        Notes
        -----
        A ``None`` extent is the signal ``BaseMap``/cartopy read as "draw the whole
        globe" (``set_global``), so the world is a first-class ``Bbox`` state rather
        than a ``None`` special case threaded through the call sites.

        Examples
        --------
        >>> Bbox.world().extent is None
        True

        """
        box = cls()
        box.is_world = True
        return box

    @classmethod
    def from_name(cls, name: str, pad: float = 0.0) -> 'Bbox':
        """
        Build a ``Bbox`` from a named region of interest in ``ROIS``.

        Parameters
        ----------
        name : str
            A key of ``ROIS`` (e.g. 'europe', 'pl').
        pad : float
            Fractional padding added to each side.

        Returns
        -------
        Bbox

        Raises
        ------
        KeyError
            If ``name`` is not a known region.

        Examples
        --------
        >>> Bbox.from_name('europe').extent
        (-10.0, 35.0, 35.0, 72.0)

        """
        try:
            bounds = ROIS[name]
        except KeyError:
            known = ', '.join(sorted(ROIS))
            raise KeyError(f'unknown bbox {name!r}; known bboxes: {known}') from None
        return cls(list(bounds), target_crs=4326, pad=pad)

    def _build(self, geometry, target_crs, pad: float) -> None:
        """
        Derive, reproject and pad the box, storing it as a one-box GeoSeries.

        Examples
        --------
        >>> b = Bbox(); b._build([0, 0, 10, 10], target_crs=4326, pad=0.1)

        """
        if isinstance(geometry, (list, tuple, np.ndarray)) and len(geometry) == 4:
            bounds = [float(x) for x in geometry]
            crs = target_crs
        else:
            geoseries = geometry.geometry if isinstance(geometry, gpd.GeoDataFrame) else geometry
            if target_crs is not None:
                geoseries = geoseries.to_crs(target_crs)
            crs = geoseries.crs
            bounds = list(geoseries.total_bounds)  # [minx, miny, maxx, maxy]

        w, h = bounds[2] - bounds[0], bounds[3] - bounds[1]
        padded = [bounds[0] - w * pad, bounds[1] - h * pad, bounds[2] + w * pad, bounds[3] + h * pad]
        self.bbox = gpd.GeoSeries([shapely_box(*padded)], crs=crs)

    @property
    def extent(self):
        """
        The box as ``(xmin, xmax, ymin, ymax)`` in lon/lat degrees, or None for the whole world.

        Notes
        -----
        None -- the whole-world signal ``cartopy``'s ``set_global`` wants -- is
        returned for a ``Bbox.world()``. Otherwise the box is reprojected to lon/lat
        when it carries a CRS (a box with no CRS is assumed lon/lat already), and the
        order matches ``set_extent`` (x first, then y), not GeoPandas' ``total_bounds``
        (which interleaves them).

        Examples
        --------
        >>> Bbox([-10, 35, 35, 72], target_crs=4326).extent
        (-10.0, 35.0, 35.0, 72.0)
        >>> Bbox.world().extent is None
        True

        """
        if self.is_world:
            return None
        if self.bbox is None:
            raise ValueError('Bbox not built yet.')
        gs = self.bbox
        if gs.crs is not None:
            gs = gs.to_crs('EPSG:4326')
        minx, miny, maxx, maxy = gs.total_bounds
        return (float(minx), float(maxx), float(miny), float(maxy))

    def plot(self, ax=None, edgecolor='red', linestyle='-', facecolor='none', zoom_in: bool = False, zoom_pad: float = 0.05, **kwargs):
        """
        Draw the box outline, optionally zooming the axes to it.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. Defaults to the current axes.
        edgecolor : str
            Outline colour.
        linestyle : str
            Outline style, e.g. '-', '--', '-.', ':'.
        facecolor : str
            Fill colour; 'none' leaves the box unfilled.
        zoom_in : bool
            Zoom the axes to the box.
        zoom_pad : float
            Fraction of width/height added around the box when zooming.
        **kwargs
            Passed to the underlying GeoSeries plot. On a projected map axes pass
            ``transform=ccrs.PlateCarree()`` here until the lon/lat axes lands.

        Returns
        -------
        matplotlib.axes.Axes

        Examples
        --------
        >>> ax = Bbox([-10, 35, 35, 72], target_crs=4326).plot(zoom_in=True)

        """
        if self.bbox is None:
            raise ValueError('Bbox not built yet.')
        if ax is None:
            ax = plt.gca()
        self.bbox.plot(ax=ax, edgecolor=edgecolor, linestyle=linestyle, facecolor=facecolor, **kwargs)

        if zoom_in:
            minx, miny, maxx, maxy = self.bbox.total_bounds
            w, h = maxx - minx, maxy - miny
            ax.set_xlim(minx - w * zoom_pad, maxx + w * zoom_pad)
            ax.set_ylim(miny - h * zoom_pad, maxy + h * zoom_pad)
        return ax


class Basins(Vector):
    pass
