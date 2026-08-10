"""
Raster class and related utilities.

"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LightSource

from plotea.log import get_logger

__all__ = ['Raster', 'RasterAnalyzer', 'RasterClipper', 'RasterPlotter']

_log = get_logger(__name__)

class Raster:
    """
    A raster: a file, an in-memory array, or neither yet -- plus how to read and draw it.

    ``Raster()`` takes no arguments, so an empty one can be filled in later; with a ``path``
    the data is read lazily, on first access, and never on construction. Two ways in:
    ``data`` gives the whole thing as an xarray DataArray (for analysis), ``read_window`` a
    decimated window (for plotting a continent without loading a continent).

    Parameters
    ----------
    path : str or Path, optional
        The raster file (GeoTIFF, VRT, ...) in lon/lat (EPSG:4326).
    data : xarray.DataArray, optional
        In-memory data, e.g. the result of an operation on another raster.
    cmap : str or Colormap or DiscreteCmap
        Colormap; a ``DiscreteCmap`` (class raster) draws with its norm + colorbar labels.
    resampling : str
        rasterio resampling name for the decimated read ('average', 'nearest', ...). Use
        'nearest' for a class raster, so codes are not averaged into nonsense.
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

    Notes
    -----
    ``rasterio``, ``rioxarray`` and ``xarray`` are imported inside the methods that need
    them: they are optional dependencies, so ``import plotea`` works without them and only
    reading a file pays for them.

    Examples
    --------
    >>> Raster()                                                   # fill in later
    >>> Raster('/data/merit_elv.vrt', cmap='terrain', scale=0.01, unit='m').plot_map()
    >>> data, extent = Raster('/data/merit_twi.vrt').read_window('eu', max_px=1000)

    """

    def __init__(self, path=None, data=None, cmap='viridis', resampling='average', label='', unit='', scale=1.0, robust=True, cbar_rect=(0.84, 0.52, 0.03, 0.4)):
        self.path       = Path(path).expanduser() if path is not None else None
        self._data      = data
        self.cmap       = cmap
        self.resampling = resampling
        self.label      = label
        self.unit       = unit
        self.scale      = scale
        self.robust     = robust
        self.cbar_rect  = cbar_rect

    def __repr__(self):
        return f'Raster(path={self.path}, data={"loaded" if self._data is not None else "not loaded"})'

    @property
    def bounds(self):
        """
        ``(minx, miny, maxx, maxy)``, or None when there is no data yet.

        Examples
        --------
        >>> Raster('/data/merit_twi.vrt').bounds

        """
        return self.data.rio.bounds() if self.data is not None else None

    @property
    def crs(self):
        """
        The raster's CRS, or None when there is no data yet.

        Examples
        --------
        >>> Raster('/data/merit_twi.vrt').crs

        """
        return self.data.rio.crs if self.data is not None else None

    @property
    def data(self):
        """
        The whole raster as an xarray DataArray, read from ``path`` on first access.

        Examples
        --------
        >>> Raster('/data/merit_twi.vrt').data

        """
        if self._data is None and self.path is not None:
            import rioxarray
            self._data = rioxarray.open_rasterio(self.path, masked=True)
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def name(self) -> str:
        """
        The file stem -- the default colorbar label and plot title.

        Examples
        --------
        >>> Raster('/data/merit_twi.vrt').name
        'merit_twi'

        """
        return self.path.stem if self.path is not None else ''

    @property
    def resolution(self):
        """
        Pixel size ``(x, y)``, or None when there is no data yet.

        Examples
        --------
        >>> Raster('/data/merit_twi.vrt').resolution

        """
        return self.data.rio.resolution() if self.data is not None else None

    @property
    def transform(self):
        """
        The affine transform, or None when there is no data yet.

        Examples
        --------
        >>> Raster('/data/merit_twi.vrt').transform

        """
        return self.data.rio.transform() if self.data is not None else None

    def align_to(self, reference, out_file, resampling=None) -> "Raster":
        """
        Reproject and snap *src* to exactly match *reference*'s CRS, resolution,
        and extent, then write to *out_file*.

        Uses rioxarray.reproject_match() under the hood.

        Parameters
        ----------
        reference : Raster
        out_file : str or Path
        resampling : rasterio.enums.Resampling
            Default bilinear (suitable for continuous data like bdod).
            Use Resampling.nearest for categorical data.

        Returns
        -------
        Raster
            New Raster aligned to *reference* and saved to *out_file*.
        """
        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        from rasterio.enums import Resampling
        resampling = resampling if resampling is not None else Resampling.bilinear
        aligned = self.data.rio.reproject_match(reference.data, resampling=resampling)
        aligned.rio.to_raster(out_file, driver='GTiff')
        _log.info('Aligned %s -> %s', self.path, out_file)
        return Raster(path=out_file)

    def clip(self, mode, out_file, **kwargs) -> 'Raster':
        """
        Clip to a region and write it, returning the new Raster. See ``RasterClipper``.

        Examples
        --------
        >>> Raster('/data/merit_elv.vrt').clip('roi', out, roi='parnu')

        """
        if self.path is None:
            raise ValueError('clip needs a file: build the Raster with a path.')
        return RasterClipper.clip(self.path, mode, out_file, **kwargs)

    def get_dx_dy(self, extent, shape):
        """
        Pixel spacing ``(dx, dy)`` in metres for a window of ``shape`` over ``extent``.

        Shading needs the spacing in metres to get slopes right; a lon/lat raster's own
        degrees would make a coarse grid look like a cliff face. dx uses the cosine of the
        mid-latitude, so a window over Lapland is not treated as one over Iberia.

        Examples
        --------
        >>> Raster(dem_path).get_dx_dy((10, 15, 45, 50), (200, 200))

        """
        lon0, lon1, lat0, lat1 = extent
        mid = 0.5 * (lat0 + lat1)
        return ((lon1 - lon0) / shape[1] * 111320.0 * np.cos(np.radians(mid)),
                (lat1 - lat0) / shape[0] * 110540.0)

    def info(self):
        """
        Log the path, shape, dtype, CRS, bounds and resolution. Reads the data if needed.

        Examples
        --------
        >>> Raster('/data/merit_twi.vrt').info()

        """
        _log.info('%s: has data=%s', self.path, self._data is not None)
        if self.data is not None:
            _log.info('shape=%s dtype=%s crs=%s bounds=%s resolution=%s',
                      self.data.shape, self.data.dtype, self.crs, self.bounds, self.resolution)
        return self

    def mask_to_streams(self, stream_raster) -> "Raster":
        """
        Mask a raster to stream pixels only.

        Stream pixels are defined as cells where the stream raster is non-zero
        and not NaN. All other pixels are set to NaN.

        Parameters
        ----------
        stream_raster : Raster
            Stream network raster (e.g. from reproduce_hy90m).
            Non-zero, non-NaN cells are treated as stream pixels.

        Returns
        -------
        Raster
            New Raster with data only at stream pixels.
        """
        streams = stream_raster.data
        stream_mask = (streams.values != 0) & ~np.isnan(streams.values.astype(float))
        masked_data = self.data.where(stream_mask)
        return Raster(data=masked_data)
    def plot(self, ax=None, bbox=None, max_px=2000, hillshade=False, cmap=None, vmin=None, vmax=None, resampling=None, vert_exag=2.5, **kwargs):
        """
        Draw the raster on an axes: its values, or shaded relief made from them.

        The plotting entry point -- reads a decimated window and hands it to
        ``RasterPlotter``. ``plot_map`` builds on this, adding a basemap and a colorbar.

        Parameters
        ----------
        ax : LonLatAxes, optional
            Axes to draw on; the current axes if None.
        bbox : str or list or Bbox, optional
            Window to read and draw; the whole raster if None -- which for a continental
            VRT means reading every pixel, so pass one.
        max_px : int
            Decimation target for the read.
        hillshade : bool
            Draw shaded relief from the values instead of the values themselves -- grey,
            0 to 1, with the pixel spacing in metres worked out from the window. For a DEM
            this is the terrain backdrop; on anything else it is meaningless.
        cmap : str or Colormap or DiscreteCmap, optional
            Overrides ``self.cmap``; a ``DiscreteCmap`` draws with its own norm.
        vmin, vmax : float, optional
            Colour limits; ``self.robust`` picks the 2-98% range when neither is given.
        resampling : str, optional
            Overrides ``self.resampling`` for the read behind this plot.
        vert_exag : float
            Vertical exaggeration, when ``hillshade``.
        **kwargs
            Passed to ``RasterPlotter.imshow`` (``alpha``, ``zorder``, ...).

        Returns
        -------
        matplotlib.image.AxesImage

        Examples
        --------
        >>> Raster(dem_path).plot(ax=ax, bbox='fr', hillshade=True)
        >>> Raster(lulc_path, cmap=esa_worldcover()).plot(ax=ax, bbox='pl')

        """
        from plotea.maps.cmaps import DiscreteCmap

        # extent= would otherwise land in **kwargs, leaving bbox None: the whole raster gets
        # read (minutes, for a continental VRT) before imshow rejects the duplicate keyword.
        if 'extent' in kwargs:
            raise TypeError('plot() takes bbox=, not extent= -- a region name, a '
                            '[minx, miny, maxx, maxy] box, or a Bbox.')
        data, extent = self.read_window(bbox, max_px, resampling=resampling)

        if hillshade:
            dx, dy = self.get_dx_dy(extent, data.shape)
            shade  = RasterPlotter.hillshaded(data, vert_exag=vert_exag, dx=dx, dy=dy)
            kwargs = {'cmap': 'gray', 'vmin': 0, 'vmax': 1, **kwargs}
            return RasterPlotter.imshow(shade, extent=extent, ax=ax, **kwargs)

        cmap   = cmap if cmap is not None else self.cmap
        scheme = cmap if isinstance(cmap, DiscreteCmap) else None
        if scheme is not None:
            kwargs = {'cmap': scheme.cmap, 'norm': scheme.norm, **kwargs}
        else:
            if self.robust and vmin is None and vmax is None and data.count():
                vmin, vmax = (float(v) for v in np.nanpercentile(data.compressed(), [2, 98]))
            kwargs = {'cmap': cmap, 'vmin': vmin, 'vmax': vmax, **kwargs}
        return RasterPlotter.imshow(data, extent=extent, ax=ax, **kwargs)

    def plot_map(self, bbox='eu', cmap=None, max_px=2000, hillshade=False, vmin=None, vmax=None, resampling=None, label=None, title=None, figsize=(8, 8)):
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

        from plotea import BASEMAP_STYLE_DEFAULT, BaseMap, laea_eu
        from plotea.maps.cmaps import DiscreteCmap

        cmap   = cmap if cmap is not None else self.cmap
        scheme = cmap if isinstance(cmap, DiscreteCmap) else None
        style  = replace(BASEMAP_STYLE_DEFAULT, land='#d9d9d9', graticule='black', border='white')
        # Grey land + blue ocean under the raster (so land outside it still shows), coastline
        # off -- the raster's own nodata edge is the coast, no coarse line over the data.
        fig, ax = BaseMap(bbox=bbox, crs=laea_eu(), style=style, coastline=False, graticule_step=10).plot(figsize=figsize)
        im = self.plot(ax=ax, bbox=bbox, max_px=max_px, hillshade=hillshade, cmap=cmap,
                       vmin=vmin, vmax=vmax, resampling=resampling, zorder=0.5)
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

    def read_window(self, bbox=None, max_px: int = 2000, resampling=None):
        """
        Read one decimated window: a masked array plus the extent it covers.

        Decimates to about ``max_px`` on the longer side, so a continent-sized raster can be
        drawn without reading it at full resolution. Returns rather than stores: the window
        depends on every argument here, so caching it on the instance would leave ``data``
        and the properties derived from it meaning whatever was last asked for.

        Parameters
        ----------
        bbox : str or list or Bbox, optional
            The window: a region name, a ``[minx, miny, maxx, maxy]`` box, or a ``Bbox``.
            The whole raster if None. Note this is a bbox, not an extent -- pass
            ``Bbox.from_any(...)`` rather than someone's ``.extent`` tuple, whose order is
            ``(xmin, xmax, ymin, ymax)``.
        max_px : int
            Target size of the longer output side.
        resampling : str, optional
            Overrides ``self.resampling`` for this read -- 'average' for continuous values,
            'nearest' for a class raster, whose codes averaging would invent classes for.

        Returns
        -------
        data : numpy.ma.MaskedArray
        extent : tuple
            The extent actually read (echoed for plotting).

        Examples
        --------
        >>> data, extent = Raster('/data/merit_twi.vrt').read_window('eu', max_px=800)

        """
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.windows import from_bounds

        from plotea.maps.vector import Bbox

        if self.path is None:
            raise ValueError('read_window needs a file: build the Raster with a path.')
        extent = Bbox.from_any(bbox).extent if bbox is not None else None
        with rasterio.open(self.path) as ds:
            if extent is None:
                b = ds.bounds
                extent = (b.left, b.right, b.bottom, b.top)
            lon0, lon1, lat0, lat1 = extent
            win = from_bounds(lon0, lat0, lon1, lat1, ds.transform)
            scale = min(1.0, max_px / max(win.width, win.height))
            out = (max(1, round(win.height * scale)), max(1, round(win.width * scale)))
            how  = resampling if resampling is not None else self.resampling
            data = ds.read(1, window=win, out_shape=out, resampling=Resampling[how], masked=True, boundless=True)
            if ds.nodata is not None:                               # external .ovr overviews may drop the nodata -> mask it explicitly
                data = np.ma.masked_equal(data, ds.nodata)
        if self.scale != 1.0:                                       # -> physical units for the colour scale
            data = data * self.scale
        return data, extent

    def reproject(self, dst_crs: int, out_file=None) -> 'Raster':
        """
        Write a copy reprojected to ``dst_crs`` (an EPSG code) and return it.

        Examples
        --------
        >>> Raster('/data/merit_elv.vrt').reproject(3035)

        """
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.warp import calculate_default_transform, reproject

        if self.path is None:
            raise ValueError('reproject needs a file: build the Raster with a path.')
        target   = f'EPSG:{dst_crs}'
        out_file = Path(out_file) if out_file is not None else self.path.parent / f'{self.path.stem}_reproj_{dst_crs}.tif'
        with rasterio.open(self.path) as src:
            transform, width, height = calculate_default_transform(src.crs, target, src.width, src.height, *src.bounds)
            meta = src.meta.copy()
            meta.update(crs=target, transform=transform, width=width, height=height)
            with rasterio.open(out_file, 'w', **meta) as dst:
                for band in range(1, src.count + 1):
                    reproject(source=rasterio.band(src, band), destination=rasterio.band(dst, band),
                              src_transform=src.transform, src_crs=src.crs,
                              dst_transform=transform, dst_crs=target, resampling=Resampling.nearest)
        _log.info('Reprojected %s to %s -> %s', self.path, target, out_file)
        return Raster(path=out_file)

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

class RasterClipper:
    """
    Clip a raster to a region, by any of several ways of naming that region.

    ``clip`` is the entry point; the mode picks which of the ``MODES`` builds the shapes
    to cut to. They live together because they share the snapping and writing helpers --
    a clip that does not snap to the source grid gives rasters that no longer align.

    Examples
    --------
    >>> RasterClipper.clip(src, 'roi', out, roi='parnu', snap=True)
    >>> RasterClipper.clip(src, 'bbox', out, bbox=(24.5, 58.0, 26.0, 59.0))

    """

    @staticmethod
    def _snap_bbox(minx, miny, maxx, maxy, transform):
        """Expand a bbox outward to align with the raster's pixel grid.

        Ensures that two rasters with the same pixel grid clipped to the same
        nominal bbox always produce identical shapes — preventing the off-by-one
        pixel mismatches that arise when floating-point bbox edges land mid-pixel.
        """
        res_x = transform.a        # pixel width  (positive)
        res_y = abs(transform.e)   # pixel height (positive magnitude)
        ox    = transform.c        # x of left edge of grid
        oy    = transform.f        # y of top  edge of grid
        minx  = ox + math.floor((minx - ox) / res_x) * res_x
        maxx  = ox + math.ceil( (maxx - ox) / res_x) * res_x
        maxy  = oy - math.floor((oy - maxy) / res_y) * res_y
        miny  = oy - math.ceil( (oy - miny) / res_y) * res_y
        return minx, miny, maxx, maxy
    @staticmethod
    def _pad_bbox(minx, miny, maxx, maxy, pad):
        """Apply fractional padding to a bounding box."""
        if pad == 0.0:
            return minx, miny, maxx, maxy
        w, h = maxx - minx, maxy - miny
        return minx - w * pad, miny - h * pad, maxx + w * pad, maxy + h * pad
    @staticmethod
    def _bbox_to_shapes(minx, miny, maxx, maxy, crs):
        """Convert a bbox to a GeoSeries suitable for rasterio.mask."""
        import geopandas as gpd
        from shapely.geometry import box as shapely_box
        return gpd.GeoSeries([shapely_box(minx, miny, maxx, maxy)], crs=crs)
    @staticmethod
    def _write_clip(ds, shapes, out_file):
        """Run rasterio.mask and write the clipped raster."""
        import rasterio
        from rasterio.mask import mask
        out_image, out_transform = mask(dataset=ds, shapes=shapes, crop=True)
        meta = ds.meta.copy()
        meta.update(
            driver='GTiff',
            height=out_image.shape[1],
            width=out_image.shape[2],
            transform=out_transform,
        )
        with rasterio.open(out_file, 'w', **meta) as dst:
            dst.write(out_image)
    @staticmethod
    def _clip_to_roi(ds, **kwargs):
        """Resolve clip shapes for a named ROI.

        Required kwargs: roi (str).
        Optional kwargs: pad (float), snap (bool).
        """
        from plotea.maps.registry import ROIS
        roi = kwargs['roi']
        pad = kwargs.get('pad', 0.0)
        snap = kwargs.get('snap', False)
        if roi not in ROIS:
            raise ValueError(f"Unknown ROI '{roi}'. Available: {list(ROIS.keys())}")
        minx, miny, maxx, maxy = RasterClipper._pad_bbox(*ROIS[roi], pad)
        if snap:
            minx, miny, maxx, maxy = RasterClipper._snap_bbox(minx, miny, maxx, maxy, ds.transform)
        return RasterClipper._bbox_to_shapes(minx, miny, maxx, maxy, ds.crs)
    @staticmethod
    def _clip_to_bbox(ds, **kwargs):
        """Resolve clip shapes for an explicit bounding box.

        Required kwargs: bbox (minx, miny, maxx, maxy).
        Optional kwargs: pad (float), snap (bool).
        """
        bbox = kwargs['bbox']
        pad = kwargs.get('pad', 0.0)
        snap = kwargs.get('snap', False)
        minx, miny, maxx, maxy = RasterClipper._pad_bbox(*bbox, pad)
        if snap:
            minx, miny, maxx, maxy = RasterClipper._snap_bbox(minx, miny, maxx, maxy, ds.transform)
        return RasterClipper._bbox_to_shapes(minx, miny, maxx, maxy, ds.crs)
    @staticmethod
    def _clip_to_poi(ds, **kwargs):
        """Resolve clip shapes for a point-of-interest square window.

        Required kwargs: poi (lon, lat), half_width (float).
        Optional kwargs: pad (float), snap (bool).
        """
        poi = kwargs['poi']
        if 'half_width' not in kwargs:
            raise ValueError("poi mode requires half_width")
        half_width = kwargs['half_width']
        pad = kwargs.get('pad', 0.0)
        snap = kwargs.get('snap', False)
        cx, cy = poi
        minx, maxx = cx - half_width, cx + half_width
        miny, maxy = cy - half_width, cy + half_width
        minx, miny, maxx, maxy = RasterClipper._pad_bbox(minx, miny, maxx, maxy, pad)
        if snap:
            minx, miny, maxx, maxy = RasterClipper._snap_bbox(minx, miny, maxx, maxy, ds.transform)
        return RasterClipper._bbox_to_shapes(minx, miny, maxx, maxy, ds.crs)
    @staticmethod
    def _clip_to_fraction(ds, **kwargs):
        """Resolve clip shapes for a centred fraction of the raster extent.

        Required kwargs: fraction (float in (0, 1]).
        Optional kwargs: pad (float), snap (bool).
        """
        fraction = kwargs['fraction']
        pad = kwargs.get('pad', 0.0)
        snap = kwargs.get('snap', False)
        if not 0 < fraction <= 1:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        b  = ds.bounds
        cx = (b.left + b.right) / 2
        cy = (b.bottom + b.top) / 2
        hw = (b.right - b.left) * fraction / 2
        hh = (b.top - b.bottom) * fraction / 2
        minx, maxx = cx - hw, cx + hw
        miny, maxy = cy - hh, cy + hh
        minx, miny, maxx, maxy = RasterClipper._pad_bbox(minx, miny, maxx, maxy, pad)
        if snap:
            minx, miny, maxx, maxy = RasterClipper._snap_bbox(minx, miny, maxx, maxy, ds.transform)
        return RasterClipper._bbox_to_shapes(minx, miny, maxx, maxy, ds.crs)
    @staticmethod
    def _clip_to_geometry(ds, **kwargs):
        """Resolve clip shapes for a vector geometry (exact shape or bbox).

        Required kwargs: geometry (GeoDataFrame or GeoSeries).
        Optional kwargs: geometry_bbox (bool), pad (float), snap (bool).
        """
        import geopandas as gpd
        geometry = kwargs['geometry']
        geometry_bbox = kwargs.get('geometry_bbox', False)
        pad = kwargs.get('pad', 0.0)
        snap = kwargs.get('snap', False)
        shapes = geometry.to_crs(ds.crs).geometry \
            if isinstance(geometry, gpd.GeoDataFrame) else geometry
        if geometry_bbox:
            minx, miny, maxx, maxy = RasterClipper._pad_bbox(*shapes.total_bounds, pad)
            if snap:
                minx, miny, maxx, maxy = RasterClipper._snap_bbox(minx, miny, maxx, maxy, ds.transform)
            shapes = RasterClipper._bbox_to_shapes(minx, miny, maxx, maxy, ds.crs)
        return shapes


    MODES = {
        'roi':      _clip_to_roi,
        'bbox':     _clip_to_bbox,
        'poi':      _clip_to_poi,
        'fraction': _clip_to_fraction,
        'geometry': _clip_to_geometry,
    }
    @staticmethod
    def clip(src, mode, out_file, **kwargs) -> Raster:
        """
        Clip a raster and write the result to *out_file*.

        Parameters
        ----------
        src : Raster or Path or str
            Source raster.
        mode : str
            Clip mode — one of 'roi', 'bbox', 'poi', 'fraction', 'geometry'.
        out_file : Path or str
            Output file path.

        Keyword arguments (vary by mode)
        ---------------------------------
        roi mode:
            roi (str) — named ROI from ``plotea.maps.registry.ROIS``.
        bbox mode:
            bbox (tuple) — (minx, miny, maxx, maxy) in raster CRS.
        poi mode:
            poi (tuple) — (lon, lat) centre of square clip window.
            half_width (float) — half-width of clip window (required).
        fraction mode:
            fraction (float) — centred fraction of raster extent, in (0, 1].
        geometry mode:
            geometry (GeoDataFrame) — vector geometry to clip to.
            geometry_bbox (bool) — clip to bbox of geometry instead of exact shape.

        Common kwargs (all modes):
            pad (float) — fractional padding on each bbox side (default 0).
            snap (bool) — snap bbox to pixel grid (default False).

        Returns
        -------
        Raster
            New Raster pointing at *out_file*.

        Examples
        --------
        >>> clip(src, 'roi', out, roi='parnu', snap=True)
        >>> clip(src, 'bbox', out, bbox=(24.5, 58.0, 26.0, 59.0))
        >>> clip(src, 'poi', out, poi=(25.0, 58.5), half_width=0.5)
        >>> clip(src, 'fraction', out, fraction=0.1)
        >>> clip(src, 'geometry', out, geometry=gdf, geometry_bbox=True)
        """
        if mode not in RasterClipper.MODES:
            raise ValueError(f"Unknown clip mode '{mode}'. Available: {list(RasterClipper.MODES)}")

        import rasterio
        src_path = src.path if isinstance(src, Raster) else Path(src)
        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(src_path) as ds:
            shapes = RasterClipper.MODES[mode].__func__(ds, **kwargs)
            RasterClipper._write_clip(ds, shapes, out_file)

        _log.info(f"Clipped {src_path} → {out_file}")
        return Raster(path=out_file)


    # Various utility functions ──────────────────────────────────────────────────

class RasterPlotter:
    """
    Draw raster arrays on a map: the array itself, or shaded relief made from it.

    Neither method does any IO -- you pass a 2D array plus its lon/lat ``extent``, so the
    same pair works for a file read through ``Raster``, a DEM window, or anything else
    already in memory. Drawn on a ``LonLatAxes``, which assumes lon/lat, so ``imshow``
    needs no ``transform``; on a stock projected ``GeoAxes`` pass
    ``transform=ccrs.PlateCarree()``.

    Examples
    --------
    >>> RasterPlotter.imshow(dem, extent=(-10, 35, 35, 72), ax=ax, cmap='terrain')
    >>> shade = RasterPlotter.hillshaded(dem, vert_exag=2.5, dx=dx, dy=dy)

    """

    @staticmethod
    def imshow(data, extent, ax=None, origin: str = 'upper', **kwargs):
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
        >>> RasterPlotter.imshow(dem, extent=(-10, 35, 35, 72), ax=ax, cmap='gray')

        """
        if ax is None:
            ax = plt.gca()
        return ax.imshow(data, extent=extent, origin=origin, **kwargs)


    @staticmethod
    def hillshaded(dem, azdeg: float = 315.0, altdeg: float = 45.0, vert_exag: float = 1.0, dx: float = 1.0, dy: float = 1.0):
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
            Intensity in ``[0, 1]``; draw it with ``imshow(..., cmap='gray')``.

        Examples
        --------
        >>> shade = RasterPlotter.hillshaded(dem, vert_exag=1.5, dx=2000, dy=2000)   # ~2 km pixels
        >>> RasterPlotter.imshow(shade, extent=ext, ax=ax, cmap='gray', vmin=0, vmax=1, zorder=0.5)

        """
        arr = np.ma.asarray(dem).astype(float)
        mask = np.ma.getmaskarray(arr)
        filled = arr.filled(np.ma.median(arr)) if mask.any() else np.asarray(arr)
        intensity = LightSource(azdeg=azdeg, altdeg=altdeg).hillshade(filled, vert_exag=vert_exag, dx=dx, dy=dy)
        return np.ma.array(intensity, mask=mask)

