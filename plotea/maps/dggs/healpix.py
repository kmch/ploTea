"""
``HealpixCells`` -- HEALPix cells drawn as filled quadrilaterals, not as points.

Drawing a cell as a dot at its centre is the wrong picture whenever the question is about
coverage: which ground a set of cells occupies, whether they tile without gaps, how a
granule falls across them. That needs the cell's actual boundary, which is what this
draws.

At fine levels there are far too many cells to outline individually, and a scatter of
centres is then both faster and honest -- so this module is for the coarse end, roughly
levels 3 to 10.

"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from plotea.log import get_logger

__all__ = ['HealpixCells']

_log = get_logger(__name__)

# Above this many cells, outlining each one stops being a picture: the edges alone cover
# the page. Roughly where a level-10 granule lands, and the point to switch to plot_raster.
OUTLINE_LIMIT = 20000

# Below this share of the cells tiling the data's extent, resampling onto a regular grid
# finds almost nothing and the image comes out blank. A bijective re-indexing sits many
# orders of magnitude under it; a properly resampled map sits near 1.
SPARSE_LIMIT = 0.05


class HealpixCells:
    """
    A set of HEALPix cells, optionally carrying a value each, drawn as filled quadrilaterals.

    Construction is pure bookkeeping; the cell corners are only looked up when ``plot`` is
    called.

    Parameters
    ----------
    cell_ids : array-like
        Cell identifiers, NESTED ordering.
    level : int
        HEALPix level, so ``nside = 2**level``.
    values : array-like, optional
        One value per cell, used for the fill colour. Cells are drawn unfilled with only
        their edges when this is None -- the plain "which cells are these" picture.
    cmap : str or Colormap
    label : str
        What the values are, used as the colorbar label.
    ellipsoid : str
        Passed through when locating the corners; must match how the ids were derived.
    edgecolor : str or None
    linewidth : float

    Notes
    -----
    A HEALPix cell's edges are not great circles, but at these levels the curvature within
    one cell is far below a line width, so joining the four corners is exact enough.

    Examples
    --------
    >>> HealpixCells(cell_ids, level=7).plot_map()
    >>> HealpixCells(cell_ids, level=7, values=counts, label='Pixels per cell').plot_map()

    """

    def __init__(self, cell_ids=None, level=None, values=None, cmap='viridis', label='', ellipsoid='WGS84', edgecolor='0.4', linewidth=0.3):
        self.cell_ids  = None if cell_ids is None else np.asarray(cell_ids, dtype=np.uint64)
        self.level     = level
        self.values    = None if values is None else np.asarray(values)
        self.cmap      = cmap
        self.label     = label
        self.ellipsoid = ellipsoid
        self.edgecolor = edgecolor
        self.linewidth = linewidth
        if self.values is not None and self.values.size != self.cell_ids.size:
            raise ValueError(f'{self.values.size} values for {self.cell_ids.size} cells')

    def __repr__(self):
        return f'HealpixCells(n={0 if self.cell_ids is None else self.cell_ids.size}, level={self.level})'

    @property
    def corners(self):
        """
        The four ``(lon, lat)`` corners of every cell, as an ``(n, 4, 2)`` array.

        Longitude is wrapped to [-180, 180), and cells straddling the antimeridian are
        dropped rather than drawn: their corners would span the full width of the map and
        paint a stripe across it. See :meth:`plot`, which reports how many went.

        Examples
        --------
        >>> HealpixCells(cell_ids, level=7).corners.shape

        """
        import healpix_geo.nested

        lon, lat = healpix_geo.nested.vertices(self.cell_ids, self.level, ellipsoid=self.ellipsoid)
        lon      = ((np.asarray(lon) + 180.0) % 360.0) - 180.0
        return np.stack([lon, np.asarray(lat)], axis=-1)

    @property
    def extent(self):
        """
        ``[lon_min, lon_max, lat_min, lat_max]`` covered by the cells.

        Examples
        --------
        >>> HealpixCells(cell_ids, level=7).extent

        """
        corners = self.corners
        return [float(corners[..., 0].min()), float(corners[..., 0].max()),
                float(corners[..., 1].min()), float(corners[..., 1].max())]

    @property
    def fill_fraction(self):
        """
        Share of the cells tiling the data's own extent that are actually present.

        Near 1 for a dense map, and vanishingly small for a bijective re-indexing, where
        every source pixel gets its own cell at a level far finer than the sampling. That
        distinction decides how the data can be drawn at all: see :meth:`plot_raster`.

        Examples
        --------
        >>> HealpixCells(cell_ids, level=29, values=v).fill_fraction

        """
        lon_min, lon_max, lat_min, lat_max = self.extent
        # Solid angle of the lon/lat box, over the mean solid angle of one cell.
        radians   = np.pi / 180.0
        solid     = (lon_max - lon_min) * radians * (np.sin(lat_max * radians) - np.sin(lat_min * radians))
        per_cell  = 4.0 * np.pi / (12.0 * 4.0 ** self.level)
        covering  = max(solid / per_cell, 1.0)
        return float(self.cell_ids.size / covering)

    def plot_points(self, ax=None, cmap=None, vmin=None, vmax=None, size=1, **kwargs):
        """
        Scatter the cell centres -- the only honest picture of data too sparse to raster or outline.

        A bijective re-indexing puts each source pixel in its own cell at a very fine
        level, so the cells neither tile anything (nothing to raster) nor have a drawable
        size (nothing to outline). Their centres, though, are exactly the original sample
        positions.

        Examples
        --------
        >>> HealpixCells(cell_ids, level=29, values=v).plot_points(ax=ax)

        """
        import healpix_geo.nested

        lon, lat = healpix_geo.nested.healpix_to_lonlat(self.cell_ids, self.level, ellipsoid=self.ellipsoid)
        lon      = ((np.asarray(lon) + 180.0) % 360.0) - 180.0
        if ax is None:
            ax = plt.gca()
        return ax.scatter(lon, lat, c=self.values, s=size, cmap=cmap or self.cmap,
                          vmin=vmin, vmax=vmax, **kwargs)

    @classmethod
    def from_lonlat(cls, lon, lat, level, ellipsoid='WGS84', **kwargs):
        """
        Cells that a set of points falls into, valued by how many points landed in each.

        The coverage picture: hand it a swath's geolocation and it shows which cells the
        swath touches and how densely.

        Examples
        --------
        >>> HealpixCells.from_lonlat(lon, lat, level=7, label='Pixels per cell')

        """
        import healpix_geo.nested

        lon   = np.asarray(lon).ravel()
        lat   = np.asarray(lat).ravel()
        good  = np.isfinite(lon) & np.isfinite(lat)
        ids   = healpix_geo.nested.lonlat_to_healpix(lon[good], lat[good], level, ellipsoid=ellipsoid)
        unique, counts = np.unique(ids, return_counts=True)
        _log.info(f'{int(good.sum()):,} points -> {unique.size:,} cells at level {level}')
        return cls(unique, level, values=counts, ellipsoid=ellipsoid, **kwargs)

    def plot(self, ax=None, cmap=None, vmin=None, vmax=None, **kwargs):
        """
        Draw the cells on an axes and return the collection.

        Returns
        -------
        matplotlib.collections.PolyCollection

        Examples
        --------
        >>> HealpixCells(cell_ids, level=7).plot(ax=ax)

        """
        if self.cell_ids.size > OUTLINE_LIMIT:
            _log.warning(f'{self.cell_ids.size:,} cells is more outlines than a page can show; '
                         f'plot_raster resamples instead and is what fine levels want')
        corners = self.corners
        # A cell spanning most of the map in longitude has been split by the wrap, not
        # genuinely stretched, so drawing it would streak the figure.
        span    = corners[..., 0].max(axis=1) - corners[..., 0].min(axis=1)
        keep    = span < 180.0
        if not keep.all():
            _log.warning(f'{int((~keep).sum())} of {keep.size} cells cross the antimeridian and are not drawn')

        if ax is None:
            ax = plt.gca()
        kwargs = {'edgecolor': self.edgecolor, 'linewidth': self.linewidth, **kwargs}
        if self.values is None:
            collection = PolyCollection(list(corners[keep]), facecolors='none', **kwargs)
        else:
            collection = PolyCollection(list(corners[keep]), cmap=cmap or self.cmap, **kwargs)
            collection.set_array(self.values[keep])
            collection.set_clim(vmin, vmax)
        ax.add_collection(collection)
        return collection

    def plot_raster(self, ax=None, fig=None, figsize=(8, 8), cmap=None, vmin=None, vmax=None, title=None, colorbar=True, shape=1024, interpolation='nearest', agg='mean', projection=None, view=None, **kwargs):
        """
        Resample the cells onto a regular grid and draw them as an image -- the fine-level counterpart of :meth:`plot_map`.

        Where :meth:`plot_map` outlines each cell, this hands the job to ``healpix-plot``,
        which resamples cell values onto a sampling grid and rasterises. That is the only
        workable route once there are more cells than a page has pixels: at level 15 a
        single Sentinel-3 granule is tens of millions of cells, and there is nothing to be
        gained from drawing each as a quadrilateral a thousandth of a pixel across.

        Parameters
        ----------
        shape : int or tuple of int
            Size of the sampling grid the cells are resampled onto.
        interpolation : {'nearest', 'bilinear'}
            How cell values reach the sampling grid. 'nearest' shows the cells as they
            are; 'bilinear' smooths across them.
        agg : str
            How duplicate cell ids are reduced before resampling.
        projection : str or cartopy CRS, optional
            Used only when creating a new axes. Defaults to the local-ish 'PlateCarree'
            rather than the whole-sphere Mollweide that ``healpix-plot`` prefers, since a
            granule is not a global map.

        Returns
        -------
        matplotlib axes

        Raises
        ------
        ValueError
            If the cells carry no values; there is nothing to rasterise.

        Examples
        --------
        >>> HealpixCells(cell_ids, level=15, values=radiance).plot_raster()

        """
        import healpix_plot

        if self.values is None:
            raise ValueError('plot_raster needs values; use plot_map to draw cell outlines alone')
        fill = self.fill_fraction
        if fill < SPARSE_LIMIT:
            _log.warning(f'only {fill:.2g} of the cells tiling this extent are present at level '
                         f'{self.level}: resampling will find almost none of them and the image '
                         f'will come out blank. plot_points scatters the centres instead')

        # Forwarded verbatim to healpix_geo, whose lookup is case-sensitive and rejects
        # the lower-cased spelling with "Operator 'wgs84' not found".
        grid = healpix_plot.HealpixGrid(level=self.level, indexing_scheme='nested',
                                        ellipsoid=self.ellipsoid.upper())
        if ax is None:
            import matplotlib.pyplot as plt

            fig = fig if fig is not None else plt.figure(figsize=figsize)
        _log.info(f'{self.cell_ids.size:,} cells at level {self.level} -> {shape} sampling grid')
        return healpix_plot.plot(
            self.cell_ids, np.asarray(self.values),
            healpix_grid   = grid,
            sampling_grid  = {'shape': shape},
            projection     = projection if projection is not None else 'PlateCarree',
            view           = view,
            interpolation  = interpolation,
            agg            = agg,
            ax             = ax,
            title          = title,
            colorbar       = colorbar,
            cmap           = cmap or self.cmap,
            vmin           = vmin,
            vmax           = vmax,
            **kwargs,
        )

    def plot_map(self, bbox=None, ax=None, fig=None, figsize=(8, 8), cmap=None, vmin=None, vmax=None, title=None, colorbar=True, basemap=None, **kwargs):
        """
        Draw the cells over a basemap with a colorbar, and return ``(fig, ax)``.

        The view defaults to the cells' own extent, padded, so they fill the frame.

        Parameters
        ----------
        basemap : dict, optional
            Passed to ``BaseMap``, e.g. ``{'graticule_step': 5}``.

        Examples
        --------
        >>> fig, ax = HealpixCells.from_lonlat(lon, lat, level=7).plot_map()
        >>> fig, ax = HealpixCells(ids, 7).plot_map(basemap={'graticule_step': 5})

        """
        from plotea.maps.base import BaseMap

        if bbox is None:
            lon_min, lon_max, lat_min, lat_max = self.extent
            pad  = 0.05 * max(lon_max - lon_min, lat_max - lat_min)
            bbox = [lon_min - pad, lat_min - pad, lon_max + pad, lat_max + pad]

        fig, ax    = BaseMap(bbox=bbox, **(basemap or {})).plot(ax=ax, fig=fig, figsize=figsize)
        collection = self.plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        if colorbar and self.values is not None:
            from plotea.maps.colorbar import Colorbar

            Colorbar.attach(collection, ax, label=self.label)
        if title is not None:
            ax.set_title(title, fontsize=9)
        return fig, ax
