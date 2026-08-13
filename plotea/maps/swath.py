"""
``Swath`` -- values on a curvilinear sensor grid, drawn from their own lon/lat arrays.

A satellite swath is not a raster. Its rows and columns are detector and scan indices,
not degrees, so the ground positions bend across the image and no ``[minx, maxx, miny,
maxy]`` extent describes it. ``Raster`` therefore cannot draw one: ``imshow`` places
pixels on a regular grid. A swath instead carries a longitude and a latitude for every
pixel, and is drawn as a quadrilateral mesh from those two arrays.

"""
import matplotlib.pyplot as plt
import numpy as np

from plotea.log import get_logger

__all__ = ['Swath', 'SwathPlotter']

_log = get_logger(__name__)


class Swath:
    """
    Values on a curvilinear grid, plus the per-pixel lon/lat that place them on the ground.

    Construction is pure bookkeeping; nothing is drawn and nothing is decimated until
    ``plot`` is called. ``plot`` draws into an axes you give it, ``plot_map`` builds a
    basemap and a colorbar around it.

    Parameters
    ----------
    values : array-like
        2D array of the quantity to draw.
    lon, lat : array-like
        2D arrays of the same shape, in degrees.
    cmap : str or Colormap
        Default colormap.
    label : str
        What the quantity is, used as the colorbar label.
    unit : str
        Appended to ``label`` in parentheses when given.
    robust : bool
        Scale colours to the 2nd-98th percentile rather than to the extremes, so a
        handful of saturated or near-zero pixels do not flatten everything else.

    Notes
    -----
    A full-resolution swath is large -- a 300 m instrument delivers around twenty million
    pixels per granule -- and drawing every quadrilateral is slow and pointless on a page
    a few inches across. ``plot`` therefore strides the arrays down to ``max_px`` by
    default. Striding, not averaging: it keeps the exact measured values, so the picture
    stays a picture of the data rather than of a resampling.

    Examples
    --------
    >>> Swath(radiance, lon, lat, label='Radiance', unit='mW m-2 sr-1 nm-1').plot_map()
    >>> Swath.from_dataarray(ds['s3_radiance'], ds['longitude'], ds['latitude']).plot(ax=ax)

    """

    def __init__(self, values=None, lon=None, lat=None, cmap='viridis', label='', unit='', robust=True):
        self.values = None if values is None else np.asarray(values)
        self.lon    = None if lon is None else np.asarray(lon)
        self.lat    = None if lat is None else np.asarray(lat)
        self.cmap   = cmap
        self.label  = label
        self.unit   = unit
        self.robust = robust
        if self.values is not None and self.lon is not None and self.lon.shape != self.values.shape:
            raise ValueError(f'lon/lat shape {self.lon.shape} does not match values {self.values.shape}')

    def __repr__(self):
        shape = 'empty' if self.values is None else 'x'.join(str(n) for n in self.values.shape)
        return f'Swath({shape}, label={self.label!r})'

    @property
    def extent(self):
        """
        The ``[lon_min, lon_max, lat_min, lat_max]`` the swath covers, ignoring gaps.

        Note the ordering: a *view* extent, as matplotlib and cartopy want it, not a
        ``(minx, miny, maxx, maxy)`` bounding box.

        Examples
        --------
        >>> Swath(values, lon, lat).extent

        """
        return [float(np.nanmin(self.lon)), float(np.nanmax(self.lon)),
                float(np.nanmin(self.lat)), float(np.nanmax(self.lat))]

    @property
    def shape(self):
        """
        Shape of the value array, or None when the swath is empty.

        Examples
        --------
        >>> Swath(values, lon, lat).shape
        (2400, 3000)

        """
        return None if self.values is None else self.values.shape

    @classmethod
    def from_dataarray(cls, values, lon, lat, **kwargs):
        """
        Build from xarray objects, taking the label and unit from the data's own attributes.

        Any explicitly passed ``label``/``unit`` wins over the attributes.

        Examples
        --------
        >>> Swath.from_dataarray(ds['s3_radiance'], ds['longitude'], ds['latitude'])

        """
        attrs  = getattr(values, 'attrs', {})
        kwargs = {'label': attrs.get('long_name', getattr(values, 'name', '') or ''),
                  'unit':  attrs.get('units', ''),
                  **kwargs}
        return cls(np.asarray(values), np.asarray(lon), np.asarray(lat), **kwargs)

    def decimate(self, max_px=400000):
        """
        Return a strided copy holding at most ``max_px`` pixels -- what makes a full granule drawable.

        The stride is the same in both directions, so the picture is not stretched.

        Examples
        --------
        >>> Swath(values, lon, lat).decimate(max_px=100000).shape

        """
        if self.values is None or self.values.size <= max_px:
            return self
        step = int(np.ceil(np.sqrt(self.values.size / max_px)))
        cut  = (slice(None, None, step), slice(None, None, step))
        _log.debug(f'{self.values.shape} -> stride {step}')
        return Swath(self.values[cut], self.lon[cut], self.lat[cut],
                     cmap=self.cmap, label=self.label, unit=self.unit, robust=self.robust)

    def plot(self, ax=None, cmap=None, vmin=None, vmax=None, max_px=400000, **kwargs):
        """
        Draw the swath as a quadrilateral mesh on an axes, and return the artist.

        Parameters
        ----------
        ax : matplotlib axes, optional
            Drawn into the current axes when None. On a ``LonLatAxes`` no ``transform``
            is needed; on a stock projected ``GeoAxes`` pass ``transform=ccrs.PlateCarree()``.
        max_px : int
            Decimate above this many pixels. Pass None to draw every one.

        Returns
        -------
        matplotlib.collections.QuadMesh

        Examples
        --------
        >>> Swath(values, lon, lat).plot(ax=ax)

        """
        swath = self if max_px is None else self.decimate(max_px)
        cmap  = cmap if cmap is not None else self.cmap
        if self.robust and vmin is None and vmax is None and np.isfinite(swath.values).any():
            vmin, vmax = (float(v) for v in np.nanpercentile(swath.values, [2, 98]))
        if ax is None:
            ax = plt.gca()
        kwargs = {'cmap': cmap, 'vmin': vmin, 'vmax': vmax, 'shading': 'auto', **kwargs}
        return SwathPlotter.pcolormesh(swath.values, swath.lon, swath.lat, ax=ax, **kwargs)

    def plot_map(self, bbox=None, ax=None, fig=None, figsize=(8, 8), cmap=None, vmin=None, vmax=None, max_px=400000, title=None, colorbar=True, **kwargs):
        """
        Draw the swath over a basemap with a colorbar, and return ``(fig, ax)``.

        The view defaults to the swath's own extent, padded slightly, so the granule
        fills the frame without being clipped by it.

        Examples
        --------
        >>> fig, ax = Swath(values, lon, lat, label='Radiance').plot_map()

        """
        from plotea.maps.base import BaseMap

        if bbox is None:
            lon_min, lon_max, lat_min, lat_max = self.extent
            pad  = 0.05 * max(lon_max - lon_min, lat_max - lat_min)
            bbox = [lon_min - pad, lat_min - pad, lon_max + pad, lat_max + pad]

        fig, ax = BaseMap(bbox=bbox).plot(ax=ax, fig=fig, figsize=figsize)
        mesh    = self.plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, max_px=max_px, **kwargs)
        if colorbar:
            from plotea.maps.colorbar import Colorbar

            base = f'{self.label} ({self.unit})' if self.unit else self.label
            Colorbar.attach(mesh, ax, label=base)
        if title is not None:
            ax.set_title(title)
        return fig, ax


class SwathPlotter:
    """
    Draw curvilinear arrays on a map. No IO, no decimation, no colour decisions.

    You pass the value array and its lon/lat, so the same call works for anything
    already in memory. Drawn on a ``LonLatAxes``, which assumes lon/lat, so no
    ``transform`` is needed; on a stock projected ``GeoAxes`` pass
    ``transform=ccrs.PlateCarree()``.

    Examples
    --------
    >>> SwathPlotter.pcolormesh(values, lon, lat, ax=ax)

    """

    @staticmethod
    def pcolormesh(values, lon, lat, ax=None, **kwargs):
        """
        ``pcolormesh`` of a curvilinear grid, with non-finite coordinates masked out.

        A swath's lon/lat can hold NaN where the instrument saw nothing, and
        ``pcolormesh`` raises on non-finite coordinates rather than skipping those
        quadrilaterals, so they are masked here instead.

        Examples
        --------
        >>> SwathPlotter.pcolormesh(values, lon, lat, ax=ax, cmap='viridis')

        """
        if ax is None:
            ax = plt.gca()
        bad = ~np.isfinite(lon) | ~np.isfinite(lat)
        if bad.any():
            lon    = np.ma.masked_array(lon, bad)
            lat    = np.ma.masked_array(lat, bad)
            values = np.ma.masked_array(values, bad | ~np.isfinite(values))
        return ax.pcolormesh(lon, lat, values, **kwargs)
