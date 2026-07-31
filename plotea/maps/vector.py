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
import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

class Dataframe:
    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    def plot_xy(self, column: str, xcol='lon', ycol='lat', ax=None, cmap='viridis',
                vmin=None, vmax=None, cbar=True, marker='o', marker_size=10,
                edgecolor='none', label=None, title=None, zorder=None):
        if ax is None:
            ax = plt.gca()
        if label is None:
            label = column
        df = self.df
        scatter = ax.scatter(df[xcol], df[ycol], c=df[column],
                             cmap=cmap, marker=marker, s=marker_size,
                             vmin=vmin, vmax=vmax, edgecolor=edgecolor, zorder=zorder)
        if cbar:
            plt.colorbar(scatter, ax=ax, label=label)
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.set_title(title)
        return ax

class Geodataframe(Dataframe):
    def __init__(self, gdf):
        super().__init__(df=gdf)
        self.gdf = gdf

    def plot(self, column: str, ax=None, **kwargs):
        """
        Plot using geometry coordinates extracted from GeoDataFrame.
        
        """
        df = self.df.copy()
        df["longitude"] = self.gdf.geometry.x
        df["latitude"]  = self.gdf.geometry.y
        kwargs['xcol'] = 'longitude'
        kwargs['ycol'] = 'latitude'
        self.df = df
        return super().plot_xy(column=column, ax=ax, **kwargs)



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

    Notes
    -----
    In the ``BaseMap`` flow a ``Bbox`` is the *view*: its ``extent`` -- in lon/lat
    degrees, ordered the way cartopy's ``set_extent`` wants -- is what crops the
    map, or None for the whole world. You rarely build one directly; ``BaseMap``
    calls ``Bbox.from_any``, which accepts a name (a key of ``ROIS``), a raw box, a
    geometry, an existing ``Bbox`` or None/'world', and always returns a ``Bbox`` --
    the whole world being a ``Bbox.world()`` in the unbounded state.

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
        >>> Bbox.from_any('eu').extent
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
            A key of ``ROIS`` (e.g. 'eu', 'pl').
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
        >>> Bbox.from_name('eu').extent
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
            Passed to the underlying GeoSeries plot. On a plotea ``LonLatAxes`` the
            box (lon/lat degrees) needs no ``transform``; it lands correctly as-is.

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


class HydroBasins(Basins):
    """
    HydroBASINS drainage basins at a given Pfafstetter level (1 coarse ... 12 fine).

    A ``Vector`` over the HydroBASINS shapefiles: coarser levels give fewer, larger
    basins (useful e.g. as conservative spatial cross-validation groups), finer
    levels give many small sub-basins. Loaded lazily like any ``Vector`` -- draw
    with ``.plot(ax=ax)``.

    Parameters
    ----------
    level : int
        Pfafstetter level, 1-12.
    base_dir : str or Path, optional
        Directory holding ``hybas_{region}_lev{NN}_v1c.shp``. Defaults to the
        ``HYDROBASINS_DIR`` environment variable; a consumer package can set that
        from its own config so ``HydroBasins(level)`` works with no path.
    region : str
        HydroBASINS regional code in the filename (e.g. 'eu', 'na', 'as').

    Examples
    --------
    >>> hb = HydroBasins(level=2)                     # HYDROBASINS_DIR must be set
    >>> hb = HydroBasins(level=6, base_dir='~/data/hybas_eu_lev01-12_v1c')
    >>> fig, ax = plotea.BaseMap(bbox='eu').plot()
    >>> hb.data.plot(ax=ax, facecolor='none', edgecolor='b')

    """

    id_col = 'HYBAS_ID'

    def __init__(self, level, base_dir=None, region: str = 'eu') -> None:
        """
        Resolve the shapefile for ``level`` and ``region``; the geometry loads lazily.

        Examples
        --------
        >>> hb = HydroBasins(level=2)

        """
        if not 1 <= level <= 12:
            raise ValueError(f'HydroBASINS level must be 1-12, got {level}.')
        self.level = level
        self.region = region
        base = base_dir if base_dir is not None else os.environ.get('HYDROBASINS_DIR')
        if not base:
            raise ValueError('HydroBasins needs base_dir, or the HYDROBASINS_DIR environment variable.')
        self.base_dir = Path(base).expanduser()
        super().__init__(path=self._file_path())

    def __repr__(self):
        """
        Show the level and resolved path.

        Examples
        --------
        >>> repr(HydroBasins(level=2))

        """
        return f'HydroBasins(level={self.level}, path={self.path})'

    def _file_path(self) -> Path:
        """
        Return the ``hybas_{region}_lev{NN}_v1c.shp`` path, raising if it is missing.

        Examples
        --------
        >>> HydroBasins(level=2)._file_path()

        """
        path = self.base_dir / f'hybas_{self.region}_lev{self.level:02d}_v1c.shp'
        if not path.exists():
            raise FileNotFoundError(f'HydroBASINS shapefile not found: {path}')
        return path


class Country(Vector):
    """
    A country polygon (Natural Earth ``admin_0``) plus the ROI box for zooming to it.

    ``Country('fr')`` carries two things a country panel needs: ``.bbox``, the
    region-of-interest box from ``ROIS`` (the *view* extent, e.g. mainland France),
    and ``.data``, the country's polygon -- lazily loaded from Natural Earth and
    clipped to that ROI, so overseas territories and any globe-spanning geometry are
    dropped before any downstream drawing, clipping or bounds computation.

    Parameters
    ----------
    code : str
        A country code that is both a key of ``ROIS`` (lower-case, e.g. 'fr') and,
        upper-cased, the Natural Earth ``ISO_A2_EH`` code (e.g. 'FR').
    resolution : str
        Natural Earth resolution: '50m' (default), '110m' or '10m'.
    clip : bool
        Clip the polygon to the ROI box before storing it. True by default -- this
        is what removes overseas territories.
    pad : float
        Fractional padding on the ROI box used *for clipping* (not for the view), so
        the mainland outline is not shaved at the box edges while distant territories
        are still cut.

    Notes
    -----
    ``ISO_A2_EH``, not ``ISO_A2``: Natural Earth codes France and Norway as
    ``ISO_A2 == '-99'`` (a sovereignty quirk); the ``_EH`` variant gives the
    expected 'FR'/'NO'. ``.data`` is a one-row (single country) ``GeoDataFrame`` in
    EPSG:4326; on a plotea ``LonLatAxes`` it draws with a bare ``.plot(ax=ax)``.

    Examples
    --------
    >>> fr = Country('fr')
    >>> fr.bbox.extent                       # mainland view extent, from ROIS
    (-4.762, 9.556, 41.384, 51.097)
    >>> ax = fr.data.plot(facecolor='none', edgecolor='k')   # mainland outline

    """

    def __init__(self, code: str, resolution: str = '50m', clip: bool = True, pad: float = 0.15) -> None:
        """
        Resolve the code to its ROI box; the polygon loads lazily on first ``.data`` access.

        Examples
        --------
        >>> fr = Country('fr')

        """
        super().__init__()
        self.roi = code.lower()
        self.iso = code.upper()
        self.resolution = resolution
        self._clip = clip
        self._pad = pad
        self.bbox = Bbox.from_name(self.roi)

    @property
    def data(self) -> gpd.GeoDataFrame:
        """
        The country polygon, loaded from Natural Earth and clipped to the ROI, cached after first access.

        Examples
        --------
        >>> Country('fr').data.shape[0]
        1

        """
        if self._data is None:
            self._data = self._load()
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    def _load(self) -> gpd.GeoDataFrame:
        """
        Read Natural Earth ``admin_0_countries``, select this country by ``ISO_A2_EH``, and clip to the ROI.

        Examples
        --------
        >>> gdf = Country('fr')._load()

        """
        from plotea.maps import carto  # cartopy access stays in carto
        gdf = gpd.read_file(carto.countries_shapefile(self.resolution))
        sel = gdf[gdf['ISO_A2_EH'] == self.iso]
        if sel.empty:
            raise KeyError(f'no country with ISO_A2_EH == {self.iso!r} in Natural Earth admin_0_countries')
        _log.info('%s: %d feature(s) selected', self.iso, len(sel))
        if self._clip:
            clip_box = Bbox.from_name(self.roi, pad=self._pad).bbox
            sel = gpd.clip(sel, clip_box)
        return sel.dissolve().reset_index(drop=True)


def _read_bbox(bbox):
    """
    Turn a ``Bbox.from_any`` input into the ``(minx, miny, maxx, maxy)`` tuple GeoPandas reads with.

    Examples
    --------
    >>> _read_bbox('fr')
    (-4.762, 41.384, 9.556, 51.097)

    """
    if bbox is None:
        return None
    xmin, xmax, ymin, ymax = Bbox.from_any(bbox).extent
    return (xmin, ymin, xmax, ymax)


class Streams(Vector):
    """
    A stream / river line network -- a ``Vector`` whose ``plot`` draws lines.

    Notes
    -----
    A thin base over ``Vector`` that only fixes line-appropriate plot defaults
    (a blue hairline); subclasses such as ``HydroRivers`` add dataset knowledge.

    Examples
    --------
    >>> Streams(data=rivers_gdf).plot(ax=ax, linewidth=0.6)

    """

    _PLOT = dict(color='#4b8fbf', linewidth=0.4)

    def plot(self, ax=None, width_by=None, width_range=(0.15, 1.2), **kwargs):
        """
        Draw the streams as lines; optionally taper the line width by river size.

        Parameters
        ----------
        ax : matplotlib axes, optional
        width_by : str, optional
            A column (e.g. 'UPLAND_SKM' or 'ORD_STRA') to scale line width by, on a
            log scale between ``width_range``. This is what makes a dense, connected
            network read naturally -- headwaters as hairlines, main stems bold --
            rather than a sparse threshold that leaves branches dangling.
        width_range : tuple
            ``(min, max)`` line widths in points for the smallest and largest rivers.
        **kwargs
            Override the blue-hairline defaults (``color``, ``linewidth``, ...).

        Examples
        --------
        >>> streams.plot(ax=ax, width_by='UPLAND_SKM')
        >>> streams.plot(ax=ax, color='steelblue', linewidth=0.8)

        """
        if self.data is None:
            raise ValueError('No data to plot.')
        data = self.data
        if width_by is not None and width_by in data:
            v = np.log10(np.asarray(data[width_by], dtype=float).clip(1.0))
            lo, hi = float(v.min()), float(v.max())
            kwargs['linewidth'] = width_range[0] if hi <= lo else np.interp(v, (lo, hi), width_range)
        # Reproject to the axes CRS once and draw in it, so cartopy skips its slow
        # per-vertex reprojection of every reach (a dense network: minutes -> seconds).
        proj = getattr(ax, 'projection', None)
        if proj is not None and data.crs is not None and 'transform' not in kwargs:
            data = data.to_crs(proj)
            kwargs['transform'] = proj
        return data.plot(ax=ax, **{**self._PLOT, **kwargs})


class HydroRivers(Streams):
    """
    The HydroRIVERS network, read filtered to the major rivers of a view.

    Only reaches with upstream catchment area >= ``min_upland`` (km2) within
    ``bbox`` are read, via an attribute + spatial filter at read time, so the
    ~1e6-feature file loads in a moment. Upstream area is chosen over discharge
    because it grows monotonically downstream, so the kept network stays *connected*
    to the sea -- a discharge threshold instead severs low-flow main stems in arid
    regions and leaves tributaries dangling. The schema also carries ``DIS_AV_CMS``
    (mean discharge), ``ORD_FLOW`` (1-10, lower = larger) and ``ORD_STRA``.

    Parameters
    ----------
    path : str or Path
        The HydroRIVERS ``.gdb`` (or any HydroRIVERS-schema file).
    min_upland : float or None
        Keep reaches with ``UPLAND_SKM >= min_upland``; None keeps all (slow).
    bbox : str or Bbox or geometry, optional
        Anything ``Bbox.from_any`` accepts (e.g. 'fr'); only rivers intersecting it
        are read.

    Examples
    --------
    >>> HydroRivers(path, min_upland=5000, bbox='fr').plot(ax=ax)

    """

    def __init__(self, path, min_upland: float = 1000.0, bbox=None) -> None:
        """
        Store the read filters; the network loads lazily on first ``.data`` access.

        Examples
        --------
        >>> rivers = HydroRivers(path, min_upland=20000, bbox='eu')

        """
        super().__init__(path=path)
        self.min_upland = min_upland
        self._bbox = bbox

    @property
    def data(self) -> gpd.GeoDataFrame:
        """
        The filtered river lines, read once and cached.

        Examples
        --------
        >>> HydroRivers(path, bbox='fr').data.crs
        <Geographic 2D CRS: EPSG:4326>

        """
        if self._data is None:
            where = None if self.min_upland is None else f'UPLAND_SKM >= {self.min_upland}'
            self._data = gpd.read_file(self.path, where=where, bbox=_read_bbox(self._bbox))
            _log.info('%d river reaches (upland >= %s km2)', len(self._data), self.min_upland)
        return self._data

    @data.setter
    def data(self, value):
        self._data = value


class Lakes(Vector):
    """
    Lake polygons -- a ``Vector`` whose ``plot`` fills the water bodies.

    Examples
    --------
    >>> Lakes(data=lakes_gdf).plot(ax=ax)

    """

    _PLOT = dict(facecolor='#cfe1f2', edgecolor='none')

    def plot(self, ax=None, **kwargs):
        """
        Draw the lakes as filled polygons; keyword args override the pale-blue defaults.

        Examples
        --------
        >>> lakes.plot(ax=ax, facecolor='#d6e6f2')

        """
        if self.data is None:
            raise ValueError('No data to plot.')
        return self.data.plot(ax=ax, **{**self._PLOT, **kwargs})


class HydroLakes(Lakes):
    """
    The HydroLAKES polygons, read filtered to the larger lakes of a view.

    Only lakes with ``Lake_area >= min_area`` (km2) within ``bbox`` are read, so the
    global ~1.4e6-feature file loads quickly.

    Parameters
    ----------
    path : str or Path
        The HydroLAKES ``.gdb`` (or any HydroLAKES-schema file).
    min_area : float or None
        Keep lakes with ``Lake_area >= min_area`` km2; None keeps all (slow).
    bbox : str or Bbox or geometry, optional
        Anything ``Bbox.from_any`` accepts; only lakes intersecting it are read.

    Examples
    --------
    >>> HydroLakes(path, min_area=50, bbox='fr').plot(ax=ax)

    """

    def __init__(self, path, min_area: float = 10.0, bbox=None) -> None:
        """
        Store the read filters; the lakes load lazily on first ``.data`` access.

        Examples
        --------
        >>> lakes = HydroLakes(path, min_area=100, bbox='eu')

        """
        super().__init__(path=path)
        self.min_area = min_area
        self._bbox = bbox

    @property
    def data(self) -> gpd.GeoDataFrame:
        """
        The filtered lake polygons, read once and cached.

        Examples
        --------
        >>> HydroLakes(path, bbox='fr').data.crs
        <Geographic 2D CRS: EPSG:4326>

        """
        if self._data is None:
            where = None if self.min_area is None else f'Lake_area >= {self.min_area}'
            self._data = gpd.read_file(self.path, where=where, bbox=_read_bbox(self._bbox))
            _log.info('%d lakes (area >= %s km2)', len(self._data), self.min_area)
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
