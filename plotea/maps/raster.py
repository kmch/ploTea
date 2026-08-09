"""
Raster on a map: an array display, a shaded-relief background, and a ``Raster`` class.

The array helpers (``plot_raster``, ``hillshade``) do no IO -- you pass a 2D array plus its
lon/lat ``extent``, drawn on a ``LonLatAxes`` (which assumes lon/lat, so ``imshow`` needs no
``transform``). The ``Raster`` class *does* read files, lazily importing ``rasterio`` (an
*optional* plotea dependency): ``import plotea`` works without it; only ``Raster.read`` needs it.

"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LightSource

from plotea.log import get_logger

__all__ = ['plot_raster', 'hillshade', 'Raster', 'RasterAnalyzer']

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


class Raster:
    """
    A raster file that reads a decimated window and plots itself on a Europe basemap.

    Bundles a file ``path`` with its plotting config -- ``cmap`` and the decimation
    ``resampling`` (use ``'nearest'`` for a class raster, so codes are not averaged into
    nonsense; ``'average'`` for a continuous one). ``read`` lazily imports ``rasterio``, an
    *optional* plotea dependency: ``import plotea`` works without it; only reading needs it.

    Parameters
    ----------
    path : str or Path
        The raster file (GeoTIFF, VRT, ...) in lon/lat (EPSG:4326).
    cmap : str or Colormap or DiscreteCmap
        Colormap; a ``DiscreteCmap`` (class raster) draws with its norm + colorbar labels.
    resampling : str
        rasterio resampling name for the decimated read ('average', 'nearest', ...).
    label : str
        Colorbar label (defaults to the file stem); ``unit`` is appended as ``(unit)``.
    unit : str
        Physical unit of the (scaled) values, shown on the colorbar as ``label (unit)``.
    scale : float
        Multiply read values by this to reach physical ``unit`` (e.g. ``0.01`` for a raster
        stored x100), so the colour scale and colorbar are in real units.
    robust : bool
        Robust (2-98%) colour limits for a continuous cmap.
    cbar_rect : tuple
        Inset colorbar position ``(x, y, w, h)`` in axes fraction.

    Examples
    --------
    >>> Raster('/data/merit_elv.vrt', cmap='terrain', scale=0.01, unit='m').plot_map()
    >>> Raster('/data/lulc.vrt', cmap=esa_worldcover(), resampling='nearest').plot_map()

    """

    def __init__(self, path, cmap='viridis', resampling='average', label='', unit='', scale=1.0, robust=True, cbar_rect=(0.84, 0.52, 0.03, 0.4)):
        """
        Store the path and plotting/reading config.

        Examples
        --------
        >>> Raster('/data/merit_elv.vrt', cmap='terrain', scale=0.01, unit='m')

        """
        self.path = Path(path).expanduser()
        self.cmap = cmap
        self.resampling = resampling
        self.label = label
        self.unit = unit
        self.scale = scale
        self.robust = robust
        self.cbar_rect = cbar_rect

    @property
    def name(self) -> str:
        """
        The file stem (default colorbar label / plot title).

        Examples
        --------
        >>> Raster('/data/merit_twi.vrt').name
        'merit_twi'

        """
        return self.path.stem

    def read(self, extent=None, max_px: int = 2000):
        """
        Read a decimated window over ``extent`` (lon0, lon1, lat0, lat1); whole raster if None.

        Lazily imports ``rasterio``; decimates to ~``max_px`` on the longer side using
        ``self.resampling``.

        Parameters
        ----------
        extent : sequence of float, optional
            ``(lon_min, lon_max, lat_min, lat_max)``; the raster's own bounds if None.
        max_px : int
            Target size of the longer output side.

        Returns
        -------
        data : numpy.ma.MaskedArray
        extent : tuple
            The extent actually read (echoed for plotting).

        Examples
        --------
        >>> data, extent = Raster('/data/merit_twi.vrt').read((-10, 30, 35, 72))

        """
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.windows import from_bounds

        with rasterio.open(self.path) as ds:
            if extent is None:
                b = ds.bounds
                extent = (b.left, b.right, b.bottom, b.top)
            lon0, lon1, lat0, lat1 = extent
            win = from_bounds(lon0, lat0, lon1, lat1, ds.transform)
            scale = min(1.0, max_px / max(win.width, win.height))
            out = (max(1, round(win.height * scale)), max(1, round(win.width * scale)))
            data = ds.read(1, window=win, out_shape=out, resampling=Resampling[self.resampling], masked=True, boundless=True)
            if ds.nodata is not None:                               # external .ovr overviews may drop the nodata -> mask it explicitly
                data = np.ma.masked_equal(data, ds.nodata)
        if self.scale != 1.0:                                       # -> physical units for the colour scale
            data = data * self.scale
        return data, extent

    def plot_map(self, bbox='eu', cmap=None, max_px=2000, vmin=None, vmax=None, label=None, title=None, figsize=(8, 8)):
        """
        Plot the raster over ``bbox`` (default Europe) with an inset colorbar (fig01 overview style).

        Reads a decimated window, then draws it over Europe: coloured land, ocean showing
        through nodata, black graticules and white borders. A ``DiscreteCmap`` for ``cmap``
        draws with its norm and class colorbar labels.

        Parameters
        ----------
        bbox : str or list or Bbox
            View extent (default 'eu').
        cmap : str or Colormap or DiscreteCmap, optional
            Override ``self.cmap`` for a one-off.
        max_px : int
            Decimation target for the read.
        vmin, vmax : float, optional
            Colour limits for a continuous cmap (ignored for a ``DiscreteCmap``).
        label, title : str, optional
        figsize : tuple

        Returns
        -------
        fig, ax

        Examples
        --------
        >>> Raster('/data/merit_elv.vrt', cmap='terrain').plot_map()

        """
        from dataclasses import replace

        from plotea import BASEMAP_GREY, BaseMap, laea_eu
        from plotea.maps.cmaps import DiscreteCmap
        from plotea.maps.vector import Bbox

        cmap = cmap if cmap is not None else self.cmap
        extent = Bbox.from_any(bbox).extent
        data, extent = self.read(extent, max_px)
        scheme = cmap if isinstance(cmap, DiscreteCmap) else None
        kwargs = {'cmap': scheme.cmap if scheme else cmap, 'zorder': 0.5}
        if scheme is not None:
            kwargs['norm'] = scheme.norm
        else:
            if self.robust and vmin is None and vmax is None and data.count():
                vmin, vmax = (float(v) for v in np.nanpercentile(data.compressed(), [2, 98]))
            kwargs['vmin'], kwargs['vmax'] = vmin, vmax
        style = replace(BASEMAP_GREY, land='#d9d9d9', graticule='black', border='white')
        # Grey land + blue ocean under the raster (so land outside it still shows), coastline
        # off -- the raster's own nodata edge is the coast, no coarse line over the data.
        fig, ax = BaseMap(bbox=bbox, crs=laea_eu(), style=style, coastline=False, graticule_step=10).plot(figsize=figsize)
        im = plot_raster(data, extent=extent, ax=ax, **kwargs)
        # Horizontal colorbar in the top-left corner -- over the NW-Atlantic / Iceland, off the data.
        wide = 0.46 if scheme is not None else 0.30            # class rasters get a wider bar for their labels
        cax = ax.inset_axes([0.04, 0.90, wide, 0.02])
        cb  = fig.colorbar(im, cax=cax, orientation='horizontal')
        cb.ax.xaxis.set_ticks_position('bottom')
        cb.ax.xaxis.set_label_position('top')
        if scheme is not None:
            cb.set_ticks(scheme.values)
            cb.set_ticklabels(scheme.labels)
            cb.ax.tick_params(labelsize=5, rotation=90)
        else:
            cb.ax.tick_params(labelsize=7)
        base = label if label is not None else (self.label or self.name)
        cb.set_label(f'{base} ({self.unit})' if self.unit else base, fontsize=8)
        if title is not None:
            ax.set_title(title)
        return fig, ax


class RasterAnalyzer:
    """
    Report structure and value ranges of rasters as a pandas DataFrame.

    ``describe`` returns one row per raster (generic, no assumptions about naming);
    ``summary`` groups those rows via ``group_by`` and aggregates per group.

    Examples
    --------
    >>> RasterAnalyzer.describe(['a_regunit_43.tif', 'a_regunit_44.tif'])
    >>> RasterAnalyzer.summary(PATH.accum_rasters, group_by='dir')

    """

    REGUNIT_SUFFIX = r'_regunit_\d+.*'   # strip this from a tile filename to get its base name

    # gdalinfo-style column order for ``full`` (name/dir prepended, val_mean kept if present)
    FULL_COLUMNS = ['name', 'dir', 'size_gb', 'driver', 'width', 'height', 'n_bands', 'dtype',
                    'nodata', 'val_min', 'val_max', 'val_mean', 'compression', 'block_x', 'block_y',
                    'crs', 'res_x', 'res_y', 'x_min', 'y_min', 'x_max', 'y_max']

    @classmethod
    def describe(cls, rasters, stats: bool = True, approx: bool = True, full: bool = False) -> pd.DataFrame:
        """
        Return a DataFrame with one row per raster: name, dir, dtype, nodata, dims, size (+ value range).

        Parameters
        ----------
        rasters : str or Path or sequence
            A directory (searched recursively for ``*.tif``), a single raster, or a
            list of paths.
        stats : bool
            Also compute the value range (``val_min`` / ``val_max`` / ``val_mean``).
            Skip it (``False``) for a fast structure-only check of huge rasters.
        approx : bool
            When ``stats``, use approximate statistics (overviews / subsampling) --
            much faster on large rasters.
        full : bool
            Also report the full gdalinfo-style set of columns: ``driver``, ``n_bands``,
            ``compression``, block size (``block_x`` / ``block_y``), ``crs``, pixel size
            (``res_x`` / ``res_y``) and bounds (``x_min`` / ``y_min`` / ``x_max`` / ``y_max``),
            ordered as in ``FULL_COLUMNS``.

        Returns
        -------
        pandas.DataFrame

        Examples
        --------
        >>> RasterAnalyzer.describe(PATH.accum_rasters / 'merit_elv', stats=False)
        >>> RasterAnalyzer.describe('/data/dem.tif', full=True)

        """
        rows = []
        for path in cls._paths(rasters):
            with rasterio.open(path) as ds:
                row = {'name': path.name,
                       'dir': path.parent.name,
                       'dtype': ds.dtypes[0],
                       'nodata': ds.nodata,
                       'width': ds.width,
                       'height': ds.height,
                       'size_gb': round(path.stat().st_size / 1e9, 4)}
                if stats:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')          # rasterio statistics() deprecation
                            st = ds.statistics(1, approx=approx)
                        row['val_min'], row['val_max'], row['val_mean'] = st.min, st.max, round(st.mean, 4)
                    except Exception as exc:                       # unreadable / all-nodata band
                        row['val_min'] = row['val_max'] = row['val_mean'] = float('nan')
                        _log.warning('stats failed for %s: %s', path.name, exc)
                if full:
                    block_y, block_x = ds.block_shapes[0]          # (rows, cols) per gdalinfo Block=WxH
                    res_x, res_y = ds.res
                    b = ds.bounds
                    row.update(driver=ds.driver, n_bands=ds.count,
                               compression=(ds.compression.name if ds.compression else None),
                               block_x=block_x, block_y=block_y,
                               crs=(ds.crs.to_string() if ds.crs else None),
                               res_x=res_x, res_y=res_y,
                               x_min=b.left, y_min=b.bottom, x_max=b.right, y_max=b.top)
            rows.append(row)
        _log.info('described %d rasters', len(rows))
        df = pd.DataFrame(rows)
        if full:
            df = df[[c for c in cls.FULL_COLUMNS if c in df.columns]]
        return df

    @classmethod
    def summary(cls, rasters, group_by=REGUNIT_SUFFIX, stats: bool = True, approx: bool = True) -> pd.DataFrame:
        """
        Group the rasters and aggregate: file count, dtype, nodata, value range, total size.

        Parameters
        ----------
        rasters : str or Path or sequence
            As for ``describe``.
        group_by : str or callable
            How to form each row's group key from its filename: a regex *stripped* from
            the name (default ``_regunit_N...`` -> group tiles by base name); the literal
            ``'dir'`` to group by parent-directory name; or a callable ``name -> key``.
        stats, approx : bool
            As for ``describe``.

        Returns
        -------
        pandas.DataFrame
            Indexed by the group key, with ``n_files``, ``dtype``, ``nodata``,
            ``size_gb_total`` (+ ``val_min`` / ``val_max`` when ``stats``).

        Examples
        --------
        >>> RasterAnalyzer.summary(PATH.accum_rasters, group_by='dir')
        >>> RasterAnalyzer.summary(tiles, group_by=lambda n: n.split('_')[0])

        """
        df = cls.describe(rasters, stats=stats, approx=approx)
        df['group'] = cls._group_key(df, group_by)
        agg = {'n_files': ('name', 'count'), 'dtype': ('dtype', 'first'),
               'nodata': ('nodata', 'first'), 'size_gb_total': ('size_gb', 'sum')}
        if stats:
            agg.update(val_min=('val_min', 'min'), val_max=('val_max', 'max'))
        return df.groupby('group').agg(**agg).round(4)

    @staticmethod
    def _group_key(df, group_by) -> pd.Series:
        """
        Derive a group key per row: a callable on the name, ``'dir'``, or a regex to strip.

        Examples
        --------
        >>> RasterAnalyzer._group_key(df, 'dir')

        """
        if callable(group_by):
            return df['name'].map(group_by)
        if group_by == 'dir':
            return df['dir']
        return df['name'].str.replace(group_by, '', regex=True)

    @staticmethod
    def _paths(rasters) -> list:
        """
        Normalise a directory, a single path, or a list into a sorted list of raster paths.

        Examples
        --------
        >>> RasterAnalyzer._paths(PATH.accum_rasters)

        """
        if isinstance(rasters, (str, Path)):
            p = Path(rasters).expanduser()
            return sorted(p.rglob('*.tif')) if p.is_dir() else [p]
        return [Path(r).expanduser() for r in rasters]
